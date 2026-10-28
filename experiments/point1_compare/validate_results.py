"""Reconcile artifact hashes, individual verdicts, coverage and measured records.

Valid artifacts can contain accepted attacks: this is not a security verdict.
Legacy aggregate-only E5 outputs cannot establish individual trial coverage.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from .artifacts import MESSAGE_FIELDS, RAW_FIELDS

REQUIRED_MANIFEST = ["run_id", "experiment", "command", "seed", "schemes", "git_commit",
                     "lib_versions", "quant", "protocol_mapping", "paper_versions",
                     "concurrency", "timing_scope", "verification", "aggregation"]
OUTPUTS = {"raw.jsonl", "messages.csv", "correctness.json", "summary.csv", "failures.jsonl"}


def _jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _inside(root, relative):
    target = (root / relative).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"Artifact path escapes run: {relative}")
    return target


def _e5(manifest, corr, raw, messages, summary, failures, root, problems):
    from collections import defaultdict
    honest_timing_keys = {(r["scheme"], r["group"], r["round"]) for r in raw
                          if r["phase"] != "tamper_verify"}
    message_keys = {(m["scheme"], m["group"], int(m["round"])) for m in messages}
    tamper_timings = defaultdict(list)
    for r in raw:
        if r["phase"] == "tamper_verify":
            tamper_timings[(r["scheme"], r["group"], r["detail"])].append(r)
    trials, honest, cases = (corr.get(k, []) for k in ["trials", "honest_rounds", "cases"])
    if not trials or not honest or not cases:
        problems.append("E5 requires individual trials, honest contexts and public transcripts")
        return
    schedule = {c["group"]: c for c in manifest["schedule"]}
    if len(schedule) != len(manifest["schedule"]):
        problems.append("Duplicate schedule group")
    expected = {(s, g) for s in manifest["schemes"] for g in schedule}
    observed = [(c["scheme"], c["case"]["group"]) for c in cases]
    if set(observed) != expected or len(observed) != len(expected):
        problems.append("Case coverage differs from schedule")
    for case in cases:
        if case["case"] != schedule.get(case["case"]["group"]):
            problems.append("Case metadata differs from schedule")
        path = _inside(root, case["public_transcript"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != case["public_transcript_sha256"]:
            problems.append("Public transcript hash mismatch: " + case["public_transcript"])
    expected_honest = {(s, g, c) for s, g in expected
                       for c in ["initial", "next_same_set", "next_changed_set"]}
    observed_honest = [(h["scheme"], h["group"], h["context"]) for h in honest]
    if set(observed_honest) != expected_honest or len(observed_honest) != len(expected_honest):
        problems.append("Honest context coverage differs from schedule")
    if corr.get("execution_errors") != len(failures):
        problems.append("execution_errors differs from failure records")
    for h in honest:
        case = schedule[h["group"]]
        if any(h[k] != case[k] for k in ["d", "N", "M"]):
            problems.append("Honest dimensions differ from schedule")
        if any(not isinstance(h[k], bool) for k in ["exact_match", "verified"]):
            problems.append("Honest verdict must be Boolean")
        identity = (h["scheme"], h["group"], h["round"])
        if identity not in honest_timing_keys:
            problems.append("Honest context has no raw timings")
        if identity not in message_keys:
            problems.append("Honest context has no measured messages")
    for scheme in manifest["schemes"]:
        inputs = [h["input_sha256"] for h in honest if h["scheme"] == scheme]
        if len(inputs) != len(set(inputs)):
            problems.append("Independent-input schedule contains repeated vector digests")
    expected_trials = {(s, g, t) for s, g in expected for t in manifest["tamper_types"][s]}
    actual_trials = [(t["scheme"], t["group"], t["tamper_type"]) for t in trials]
    if set(actual_trials) != expected_trials or len(actual_trials) != len(expected_trials):
        problems.append("Individual tamper coverage differs from schedule")
    counts = {}
    for t in trials:
        key = f"{t['scheme']}:{t['size']}:{t['tamper_type']}"
        c = counts.setdefault(key, {"trials": 0, "rejected": 0, "accepted": 0})
        if not isinstance(t["accepted"], bool) or t["honest_context_valid"] is not True:
            problems.append("Tamper trial needs Boolean verdict and valid honest context")
        if not _finite(t["verification_ms"]):
            problems.append("Invalid tamper verification time")
        case = schedule[t["group"]]
        if t["size"] != case["size"] or t["trial"] != case["trial"]:
            problems.append("Tamper metadata differs from schedule")
        c["trials"] += 1
        c["accepted" if t["accepted"] else "rejected"] += 1
        timing = tamper_timings[(t["scheme"], t["group"], t["tamper_type"])]
        if (len(timing) != 1 or timing[0]["verified"] != t["accepted"]
                or timing[0]["elapsed_ms"] != t["verification_ms"]):
            problems.append("Tamper verdict/time differs from raw record")
    if corr.get("types") != counts:
        problems.append("E5 aggregate counts differ from individual trials")
    summary_counts = {}
    for s in summary:
        key = f"{s['scheme']}:{s['size']}:{s['tamper_type']}"
        if key in summary_counts:
            problems.append("Duplicate summary row")
        summary_counts[key] = {k: int(s[k]) for k in ["trials", "rejected", "accepted"]}
    if summary_counts != counts:
        problems.append("Summary counts differ from individual trials")
    rejected_all = bool(trials) and not any(t["accepted"] for t in trials)
    if corr.get("all_tested_attacks_rejected") is not rejected_all:
        problems.append("Claim of all attacks rejected differs from individual verdicts")


def _validate(root):
    problems = []
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for key in REQUIRED_MANIFEST:
        if key not in manifest:
            problems.append("Manifest missing " + key)
    raw, failures = (_jsonl(root / name) for name in ["raw.jsonl", "failures.jsonl"])
    fields, messages = _csv(root / "messages.csv")
    _, summary = _csv(root / "summary.csv")
    corr = json.loads((root / "correctness.json").read_text(encoding="utf-8"))
    if corr["experiment"] != manifest["experiment"]:
        problems.append("Experiment mismatch")
    if not raw or not messages or not summary:
        problems.append("Measured raw/message/summary evidence must be nonempty")
    v2 = manifest.get("schema_version") == 2
    if v2:
        if manifest.get("execution_status") != "completed":
            problems.append("Run is not completed")
        if not manifest.get("source_sha256"):
            problems.append("Missing archived source hashes")
        for name, expected in manifest.get("source_sha256", {}).items():
            path = _inside(root, manifest["source_archive"] + "/" + name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                problems.append("Archived source hash mismatch: " + name)
        if set(manifest.get("output_sha256", {})) != OUTPUTS:
            problems.append("Output hash inventory differs from required outputs")
        for name, expected in manifest.get("output_sha256", {}).items():
            path = _inside(root, name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                problems.append("Output hash mismatch: " + name)
        if not set(MESSAGE_FIELDS) <= set(fields):
            problems.append("Message CSV missing schema-2 fields")
    for r in raw:
        if v2 and not set(RAW_FIELDS) <= set(r):
            problems.append("Raw row missing schema-2 fields")
        if not _finite(r.get("elapsed_ms")):
            problems.append("Raw elapsed_ms must be finite and non-negative")
        if r.get("scheme") not in manifest["schemes"] or r.get("run_id") != manifest["run_id"]:
            problems.append("Raw scheme/run_id mismatch")
        if any(not r.get(k) for k in (["phase", "role", "actor"] if v2 else ["phase", "role"])):
            problems.append("Raw row missing phase/role/actor")
    for m in messages:
        if int(m["num_bytes"]) <= 0:
            problems.append("Message bytes must be positive serialized lengths")
        if v2 and (not m["sender"] or not m["receiver"]):
            problems.append("Message requires concrete sender and receiver")
        if m["scheme"] not in manifest["schemes"] or m["run_id"] != manifest["run_id"]:
            problems.append("Message scheme/run_id mismatch")
    if corr["experiment"] == "decided_comparison":
        _performance(manifest, corr, raw, messages, summary, failures, problems)
        _decided_comparison(manifest, corr, raw, messages, root, problems)
    elif corr["experiment"] == "e5":
        if v2:
            _e5(manifest, corr, raw, messages, summary, failures, root, problems)
        else:
            problems.append("Legacy E5 aggregate-only evidence cannot establish individual trial coverage")
    elif corr["experiment"] == "e1":
        groups = corr.get("groups", [])
        identities = [(g["scheme"], str(g["group"])) for g in groups]
        if not groups or len(set(identities)) != len(identities):
            problems.append("E1 missing or duplicate group records")
        for scheme in manifest["schemes"]:
            records = [g for g in groups if g["scheme"] == scheme]
            if v2:
                inputs = [g.get("input_sha256") for g in records]
                if (any(not isinstance(h, str) or len(h) != 64 for h in inputs)
                        or len(inputs) != len(set(inputs))):
                    problems.append(f"{scheme}: E1 independent-input digests missing or repeated")
                for group in records:
                    if group.get("input_kind") == "training_snapshot":
                        path = _inside(root, group["input_archive"])
                        if hashlib.sha256(path.read_bytes()).hexdigest() != group["input_archive_sha256"]:
                            problems.append("E1 archived training snapshot hash mismatch")
                        import numpy as np
                        with np.load(path, allow_pickle=False) as snapshot:
                            vectors = {str(u): v for u, v in zip(snapshot["user_ids"], snapshot["vectors"])}
                            if manifest.get("input_digest_algorithm") == "uid_zero_delimited_int64le_v1":
                                from .adapters.common import vector_digest
                                measured = vector_digest(vectors)
                            else:
                                measured = hashlib.sha256(np.stack([vectors[u] for u in sorted(vectors)]).astype("<i8").tobytes()).hexdigest()
                            if measured != group["input_sha256"]:
                                problems.append("E1 training snapshot content differs from input digest")
            counts = {"groups": len(records), "exact_match": sum(bool(g["exact_match"]) for g in records),
                      "verified": sum(bool(g["verified"]) for g in records),
                      "failed": sum(not(g["exact_match"] and g["verified"]) for g in records)}
            rows = [s for s in summary if s["scheme"] == scheme]
            if len(rows) != 1 or any(int(rows[0][k]) != v for k, v in counts.items()):
                problems.append(f"{scheme}: E1 summary differs from correctness records")
            if any(corr.get("counts", {}).get(scheme, {}).get(k) != v for k, v in counts.items()):
                problems.append(f"{scheme}: E1 counts differ from correctness records")
    elif corr["experiment"] in ("e2", "e3", "e4", "e6") and v2:
        _performance(manifest, corr, raw, messages, summary, failures, problems)
    elif corr["experiment"] == "e7" and v2:
        _training(manifest, corr, raw, messages, summary, failures, root, problems)
    elif corr["experiment"] == "e4_boundary" and v2:
        _availability_boundary(manifest, corr, raw, messages, summary, failures, root, problems)
    elif corr["experiment"] in ("vlmrs_fullgraph_diagnostic", "saev_full_diagnostic", "ovp_full_diagnostic") and v2:
        _conditional_diagnostic(manifest, corr, raw, messages, summary, failures, problems)
    else:
        problems.append("Unsupported experiment contract: " + corr["experiment"])
    if manifest.get("artifact_kind") == "derived_complete_point":
        _derived_point(manifest, corr, raw, messages, root, problems)
    return list(dict.fromkeys(problems))


def _derived_point(manifest, corr, raw, messages, root, problems):
    import hashlib
    meta = manifest["derivation"]
    if manifest["experiment"] != "e6" or meta["new_protocol_execution"] is not False:
        problems.append("Completed-point view must be labelled as derived E6 evidence")
    origin = _inside(root, meta["origin_manifest"])
    tool = _inside(root, meta["tool_source"])
    if digest_file(origin) != meta["origin_manifest_sha256"] or digest_file(tool) != meta["tool_source_sha256"]:
        problems.append("Derivation origin/tool checksum mismatch")
        return
    original = json.loads(origin.read_text(encoding="utf-8"))
    selected = [s for s in original["settings"] if [s[k] for k in ["N", "M", "d"]] == meta["point"]]
    if (manifest["settings"] != selected or len(selected) != 3
            or {s["variant"] for s in selected} != {"hvsa", "hvsa_three_mask_control", "hvsa_verification_off"}
            or original["run_id"] != meta["origin_run_id"]
            or original["execution_status"] != meta["origin_execution_status"]
            or original["source_sha256"] != manifest["source_sha256"]):
        problems.append("Derived view differs from declared original full point")
    expected_files = {"origin/checkpoints/"+s["setting"]+".json" for s in selected}
    if set(meta["origin_checkpoints"]) != expected_files:
        problems.append("Derived view lacks exact origin checkpoint inventory")
        return
    original_raw, original_messages, original_obs, original_init = [], [], [], []
    for spec in selected:
        relative = "origin/checkpoints/"+spec["setting"]+".json"
        path = _inside(root, relative)
        if digest_file(path) != meta["origin_checkpoints"][relative]:
            problems.append("Origin checkpoint checksum mismatch")
            continue
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        expected = list(range(spec["warmup"]+spec["rounds"]))
        if (checkpoint["specification"] != spec or [o["round"] for o in checkpoint["observations"]] != expected):
            problems.append("Derived point origin was incomplete or had a different specification")
        original_raw.extend({**r, "run_id": manifest["run_id"]} for r in checkpoint["raw"])
        original_messages.extend({**r, "run_id": manifest["run_id"]} for r in checkpoint["messages"])
        original_obs.extend(checkpoint["observations"])
        original_init.append({k: checkpoint[k] for k in ["specification", "init_ms", "init_sent_bytes", "cache"]})
    if raw != original_raw or corr["observations"] != original_obs or corr["initialization"] != original_init:
        problems.append("Derived records differ from original checkpoints")
    # CSV encodes integer fields as text, including empty None values.
    normalized = [{k: "" if v is None else str(v) for k, v in r.items()} for r in original_messages]
    if messages != normalized:
        problems.append("Derived message records differ from original checkpoints")


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decided_comparison(manifest, corr, raw, messages, root, problems):
    import numpy as np
    from .adapters.common import reference_sum, vector_digest
    from collections import defaultdict
    if manifest.get("performance_qualified") is not False or corr.get("paper_reproduction_qualified") is not False:
        problems.append("Decided implementations must retain original-equivalence limitations")
    archive = root / manifest["source_archive"] / "experiments/point1_compare/configs/baseline_decisions_20261019.json"
    if (digest_file(archive) != manifest["decisions_sha256"]
            or json.loads(archive.read_text(encoding="utf-8")) != manifest["decisions"]):
        problems.append("Baseline decisions differ from launch archive")
    initialization = corr["initialization"]
    if sorted(i["scheme"] for i in initialization) != sorted(manifest["schemes"]):
        problems.append("Comparison initialization scheme coverage differs")
    for item in initialization:
        rows = [r for r in raw if r["group"]=="init:"+item["scheme"]]
        traffic = [r for r in messages if r["group"]=="init:"+item["scheme"]]
        if not rows or not traffic or sum(int(m["num_bytes"]) for m in traffic)!=item["sent_bytes"]:
            problems.append("Comparison initialization messages differ")
        if any(not math.isclose(sum(r["elapsed_ms"] for r in rows if r["role"]==role), elapsed,
                                rel_tol=1e-9,abs_tol=1e-7) for role,elapsed in item["role_ms"].items()):
            problems.append("Comparison initialization role times differ")
    paired = defaultdict(list)
    for o in corr["observations"]:
        snapshot = _inside(root,o["snapshot"])
        if digest_file(snapshot) != o["snapshot_sha256"]:
            problems.append("Comparison input snapshot hash mismatch")
        with np.load(snapshot,allow_pickle=False) as data:
            vectors = {str(u):v for u,v in zip(data["user_ids"],data["vectors"])}
            if (data["vectors"].shape != (o["N"],o["d"]) or data["vectors"].dtype != np.dtype("int64")
                    or sorted(vectors) != sorted(manifest["users"]) or vector_digest(vectors) != o["input_sha256"]):
                problems.append("Comparison input dimensions, identities or digest differ")
            reference = reference_sum([vectors[u] for u in o["uploaded"]]).tolist()
        if o["reference_sum"] != reference or o["exact_match"] is not (o["completed"] and o["returned_sum"]==reference):
            problems.append("Comparison result differs from archived plaintext reference")
        if (o["n_uploaded"] != len(o["uploaded"]) or not set(o["recipients"]) <= set(o["uploaded"]) <= set(manifest["users"])):
            problems.append("Comparison upload/recipient coverage differs")
        rows = [r for r in raw if r["group"]==o["group"]]
        traffic = [r for r in messages if r["group"]==o["group"]]
        if any(r["scheme"]!=o["scheme"] or r["round"]!=o["protocol_round"] or r["success"] is not o["completed"]
               or r["verified"] is not o["verified"] for r in rows):
            problems.append("Comparison raw round/verdict differs")
        kind = "aggregate_and_proof" if o["scheme"]=="vlmrs_fullgraph_public_d" else "verified_model"
        delivered = sorted(r["receiver"] for r in traffic if r["payload_kind"]==kind)
        if o["valid"] and delivered != o["recipients"]:
            problems.append("Comparison actual output recipients differ")
        paired[(o["case"],o["round"])].append(o)
    for obs in paired.values():
        if (sorted(o["scheme"] for o in obs)!=sorted(manifest["schemes"])
                or any(len({json.dumps(o[k],sort_keys=True) for o in obs})!=1 for k in
                       ["input_sha256","snapshot_sha256","uploaded","recipients","reference_sum"])):
            problems.append("Comparison paired schemes do not use the same input/availability")
    if corr["execution_errors"] != len(_jsonl(root/"failures.jsonl")):
        problems.append("Comparison execution error count differs")


def _conditional_diagnostic(manifest, corr, raw, messages, summary, failures, problems):
    cases = {c["case"]: c for c in manifest["cases"]}
    obs = corr.get("observations", [])
    if (len(obs) != len(cases) or {o["case"] for o in obs} != set(cases)
            or len(summary) != len(cases)):
        problems.append("Conditional baseline diagnostic case coverage incomplete")
        return
    if manifest.get("performance_qualified") is not False or corr.get("paper_reproduction_qualified") is not False:
        problems.append("Conditional baseline diagnostic cannot claim qualified paper performance")
    if len({o["input_sha256"] for o in obs}) != len(obs):
        problems.append("Conditional baseline diagnostic inputs are not independent")
    for o in obs:
        spec = cases[o["case"]]
        if any(o[k] != spec[k] for k in ("uploaded", "recovery") if k in spec):
            problems.append("Conditional baseline diagnostic availability differs from manifest")
        expected = o["completed"] and o["returned_sum"] == o["reference_sum"]
        if o["exact_match"] is not expected:
            problems.append("Conditional baseline diagnostic exact-match flag contradicts returned/reference sums")
        if len(o["reference_sum"]) != manifest["d"] or (o["completed"] and len(o["returned_sum"]) != manifest["d"]):
            problems.append("Conditional baseline diagnostic result dimension mismatch")
        rows = [r for r in raw if r["group"] == o["case"]]
        traffic = [m for m in messages if m["group"] == o["case"]]
        if not rows or not traffic:
            problems.append("Conditional baseline diagnostic case lacks measured events")
        if sum(int(m["num_bytes"]) for m in traffic) != o["sent_bytes"]:
            problems.append("Conditional baseline diagnostic traffic differs from measured records")
        for role, elapsed in o["role_ms"].items():
            if not math.isclose(sum(r["elapsed_ms"] for r in rows if r["role"] == role), elapsed,
                                rel_tol=1e-9, abs_tol=1e-7):
                problems.append("Conditional baseline diagnostic role time differs from records")
        summaries = [s for s in summary if s["case"] == o["case"]]
        if len(summaries) != 1 or any(summaries[0][k] != ("" if o[k] is None else str(o[k]))
                                     for k in ["completed", "exact_match", "verified"]):
            problems.append("Conditional baseline diagnostic summary differs from observations")
        if manifest["experiment"] == "ovp_full_diagnostic":
            details = o["details"]
            expected_online = sorted(set(spec["received_sets"][0]) & set(spec["received_sets"][1]))
            verdicts, counts = details["client_verdicts"], details["accepted_coordinates"]
            if (o["uploaded"] != expected_online or details["intersection"] != expected_online
                    or details["received_sets"] != [sorted(s) for s in spec["received_sets"]]
                    or set(verdicts) != set(expected_online) or set(counts) != set(expected_online)):
                problems.append("OVP full-chain uploader/verification coverage differs")
            if (any(not isinstance(c, int) or not 0 <= c <= manifest["d"] for c in counts.values())
                    or any(verdicts[u] is not (counts[u] == manifest["d"]) for u in verdicts)
                    or o["verified"] is not all(verdicts.values())
                    or details["tags_per_server"] != manifest["d"]*len(expected_online)
                    or details["attacker_uses_service_betas"] is not False):
                problems.append("OVP coordinate tag/verdict scope mismatch")
    if manifest["experiment"] == "vlmrs_fullgraph_diagnostic":
        reuse = corr["later_key_recovery"]
        if reuse["exact_match"] is not (reuse["reference"] == reuse["cloud_recovered"]):
            problems.append("VLMRS later-key recovery flag contradicts vectors")
        if (len(reuse["reference"]) != manifest["d"] or len(reuse["cloud_recovered"]) != manifest["d"]
                or reuse["attacker_uses_reference_vectors_or_verification_keys"] is not False):
            problems.append("VLMRS later-key diagnostic scope mismatch")
    if corr.get("execution_errors") != len(failures):
        problems.append("Conditional baseline diagnostic execution error count differs")


def _availability_boundary(manifest, corr, raw, messages, summary, failures, root, problems):
    import numpy as np
    from .adapters.common import reference_sum, vector_digest
    cases = {c["case"]:c for c in manifest["cases"]}
    observations = corr["observations"]
    groups = manifest["groups"]
    users = [u for group in groups for u in group]
    if (len(groups) != manifest["M"] or len(users) != manifest["N"]
            or len(set(users)) != len(users)):
        problems.append("Boundary registered groups are not a unique partition")
    if len(observations) != len(cases) or {o["case"] for o in observations} != set(cases):
        problems.append("Boundary case coverage differs from manifest")
    if len({o["input_sha256"] for o in observations}) != len(observations):
        problems.append("Boundary inputs are not independent")
    bad = []
    for o in observations:
        spec = cases[o["case"]]
        if any(o[k] != spec[k] for k in ["uploaded", "recipients"]):
            problems.append("Boundary availability differs from manifest")
        if (not set(spec["recipients"]) <= set(spec["uploaded"]) <= set(users)
                or o["per_group_uploaded"] != [len(set(g) & set(spec["uploaded"])) for g in groups]):
            problems.append("Boundary per-group upload coverage differs")
        path = _inside(root, o["snapshot"])
        if digest_file(path) != o["snapshot_sha256"]:
            problems.append("Boundary input snapshot changed")
        with np.load(path, allow_pickle=False) as snapshot:
            values, ids = snapshot["vectors"], snapshot["user_ids"]
            if (values.shape != (manifest["N"], manifest["d"]) or values.dtype != np.dtype("int64")
                    or ids.shape != (manifest["N"],) or sorted(str(u) for u in ids) != sorted(users)):
                problems.append("Boundary input shape/domain differs")
                continue
            vectors = {str(u):v for u,v in zip(ids, values)}
            if vector_digest(vectors) != o["input_sha256"]:
                problems.append("Boundary input digest differs")
            reference = reference_sum([vectors[u] for u in spec["uploaded"]])
            if reference.tolist() != o["reference_sum"]:
                problems.append("Boundary plaintext reference differs from archived inputs")
        exact = o["completed"] and o["returned_sum"] == o["reference_sum"]
        if o["exact_match"] is not exact:
            problems.append("Boundary exact-match flag differs")
        rows = [r for r in raw if r["group"] == o["case"]]
        traffic = [m for m in messages if m["group"] == o["case"]]
        if any(r["success"] is not o["completed"] or r["verified"] is not o["verified"]
               or any(r[k] != manifest[k] for k in ["M", "d"])
               or r["N_selected"] != manifest["N"] or r["N_uploaded"] != len(spec["uploaded"])
               or r["round"] != o["round"] for r in rows):
            problems.append("Boundary raw verdict or dimensions differ")
        receivers = sorted(m["receiver"] for m in traffic if m["payload_kind"] == "verified_model")
        matched = (o["completed"] is spec["expected_completed"] and o["verified"] is spec["expected_verified"]
                   and (exact if o["completed"] else o["returned_sum"] is None)
                   and receivers == (spec["recipients"] if o["completed"] else []))
        if o["matches_expected_boundary"] is not matched or o["model_receivers"] != receivers:
            problems.append("Boundary outcome or model delivery differs from actual messages")
        if not rows or not traffic or sum(int(m["num_bytes"]) for m in traffic) != o["sent_bytes"]:
            problems.append("Boundary lacks reconciled events/messages")
        for role, elapsed in o["role_ms"].items():
            if not math.isclose(sum(r["elapsed_ms"] for r in rows if r["role"] == role), elapsed, rel_tol=1e-9,abs_tol=1e-7):
                problems.append("Boundary role time differs from raw records")
        entries = [s for s in summary if s["case"] == o["case"]]
        if len(entries) != 1 or any(entries[0][k] != ("" if o[k] is None else str(o[k]))
            for k in ["completed", "verified", "matches_expected_boundary"]):
            problems.append("Boundary summary differs from observations")
        if not matched:
            bad.append(o["case"])
    if corr["execution_errors"] != len(failures) or sorted(bad) != sorted(f["case"] for f in failures):
        problems.append("Boundary execution-error records differ")


def _performance(manifest, corr, raw, messages, summary, failures, problems):
    from collections import defaultdict
    import numpy as np
    settings = {s["setting"]: s for s in manifest["settings"]}
    grouped_raw = defaultdict(list)
    grouped_links = defaultdict(lambda: defaultdict(int))
    for r in raw:
        grouped_raw[(r["scheme"], r["group"])].append(r)
    for m in messages:
        grouped_links[(m["scheme"], m["group"])][m["link"]] += int(m["num_bytes"])
    observations = corr.get("observations", [])
    expected = {(s["setting"], r) for s in settings.values() for r in range(s["warmup"]+s["rounds"])}
    observed = [(o["setting"], o["round"]) for o in observations]
    if not observations or set(observed) != expected or len(observed) != len(expected):
        problems.append("Performance observations differ from complete settings schedule")
    invalid = set()
    for o in observations:
        s = settings[o["setting"]]
        if (o["scheme"] != s["variant"] or any(o[k] != s[k] for k in ["N", "M", "d"])
                or o["warmup"] != (o["round"] < s["warmup"])):
            problems.append("Observation metadata differs from setting")
        valid = (o["completed"] and o["exact_match"] and
                 (o["verified"] is True if s["verification_enabled"] else o["verified"] is None))
        if valid != o["valid"]:
            problems.append("Performance valid flag differs from correctness/verifier verdict")
        if not valid:
            invalid.add((o["setting"], o["round"]))
        if not _finite(o["wall_ms"]) or any(not _finite(v) for v in o["role_ms"].values()):
            problems.append("Invalid performance time")
        rows = grouped_raw[(o["scheme"], o["group"])]
        if not rows:
            problems.append("Performance observation missing raw records")
        for role, elapsed in o["role_ms"].items():
            if not math.isclose(sum(r["elapsed_ms"] for r in rows if r["role"] == role), elapsed,
                                rel_tol=1e-9, abs_tol=1e-7):
                problems.append("Performance role time differs from raw records")
        links = grouped_links[(o["scheme"], o["group"])]
        if dict(links) != o["link_bytes"] or sum(links.values()) != o["total_sent_bytes"]:
            problems.append("Performance traffic differs from measured messages")
    if {(f["setting"], f["round"]) for f in failures} != invalid or len(failures) != len(invalid):
        problems.append("Performance failure records differ from invalid observations")
    expected_summary = {}
    for setting in settings:
        obs = [o for o in observations if o["setting"] == setting and not o["warmup"]]
        for metric in ["wall_ms", "total_sent_bytes", "user_ms", "edge_ms", "ta_ms", "cloud_ms"]:
            values = [o["role_ms"][metric[:-3]] if metric in ["user_ms", "edge_ms", "ta_ms", "cloud_ms"]
                      else o[metric] for o in obs]
            expected_summary[(setting, metric)] = {
                "median": float(np.median(values)), "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)), "count": len(values)}
    actual_keys = [(r["setting"], r["metric"]) for r in summary]
    if set(actual_keys) != set(expected_summary) or len(actual_keys) != len(expected_summary):
        problems.append("Performance summary metric coverage differs from observations")
    for r in summary:
        expected_stats = expected_summary[(r["setting"], r["metric"])]
        if any(not math.isclose(float(r[k]), v, rel_tol=1e-9, abs_tol=1e-7) for k, v in expected_stats.items()):
            problems.append("Performance summary statistics differ from observations")


def validate(run_dir):
    try:
        return _validate(Path(run_dir))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return [f"Unreadable or invalid artifact contract: {type(exc).__name__}: {exc}"]


def _training(manifest, corr, raw, messages, summary, failures, root, problems):
    import numpy as np
    from .run_integrity import vector_digest
    from .adapters.common import reference_sum
    obs = corr.get("observations", [])
    expected = {(s, r) for s in manifest["training_seeds"] for r in range(manifest["rounds"])}
    actual = [(o["seed"], o["round"]) for o in obs]
    if set(actual) != expected or len(actual) != len(expected):
        problems.append("Training coverage differs from scheduled seeds/rounds")
    timing_keys = {(r["seed"], r["group"]) for r in raw}
    message_groups = {m["group"] for m in messages}
    invalid = set()
    for o in obs:
        if (o["seed"], o["group"]) not in timing_keys or o["group"] not in message_groups:
            problems.append("Training observation lacks raw timing or messages")
        path = _inside(root, o["snapshot"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != o["snapshot_sha256"]:
            problems.append("Training snapshot hash mismatch")
            continue
        with np.load(path, allow_pickle=False) as snapshot:
            values, ids = snapshot["vectors"], snapshot["user_ids"]
            if values.shape != (manifest["N"], manifest["d"]) or values.dtype != np.int64:
                problems.append("Training snapshot has invalid shape/domain")
                continue
            vectors = {str(uid): vector for uid, vector in zip(ids, values)}
            if len(vectors) != manifest["N"] or vector_digest(vectors) != o["input_sha256"]:
                problems.append("Training input digest mismatch")
            reference = reference_sum(list(vectors.values()))
            exact = hashlib.sha256(reference.astype("<i8").tobytes()).hexdigest() == o["raw_sum_sha256"]
            if exact != o["exact_match"]:
                problems.append("Training exact-match flag differs from snapshot integer reference")
        valid = o["exact_match"] and o["verified"] is True
        if valid != o["valid"]:
            problems.append("Training validity differs from correctness/verification verdict")
        if not valid:
            invalid.add((o["seed"], o["round"]))
        elif not _finite(o["test_accuracy"]) or o["test_accuracy"] > 1:
            problems.append("Invalid test accuracy")
        if not _finite(o["training_ms"]) or not _finite(o["protocol_ms"]):
            problems.append("Invalid training/protocol time")
    if {(f["seed"], f["round"]) for f in failures} != invalid or len(failures) != len(invalid):
        problems.append("Training failures differ from invalid observations")
    if (len(summary) != len(manifest["training_seeds"])
            or sorted(int(r["seed"]) for r in summary) != sorted(manifest["training_seeds"])):
        problems.append("Training summary missing or duplicate seeds")
    for row in summary:
        seed_obs = [o for o in obs if o["seed"] == int(row["seed"])]
        if (int(row["rounds"]) != len(seed_obs)
                or int(row["exact_match"]) != sum(o["exact_match"] for o in seed_obs)
                or int(row["verified"]) != sum(o["verified"] is True for o in seed_obs)):
            problems.append("Training summary counts differ from observations")


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m experiments.point1_compare.validate_results RUN_DIR")
        return 2
    problems = validate(sys.argv[1])
    print("INVALID:\n" + "\n".join("- " + p for p in problems) if problems
          else "OK: artifacts reconcile; this does not assert protocol security")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
