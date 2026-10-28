"""Versioned run artifacts with archived source and output checksums."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


RAW_FIELDS = ["scheme", "run_id", "seed", "group", "repeat", "N_selected",
              "N_uploaded", "M", "d", "round", "phase", "role", "actor", "detail",
              "elapsed_ms", "sent_bytes", "success", "verified", "error_reason"]
MESSAGE_FIELDS = ["scheme", "run_id", "group", "round", "link", "direction", "phase",
                  "payload_kind", "sender", "receiver", "num_bytes", "note"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def create_run(directory: Path, manifest: dict, repository: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    manifest["schema_version"] = 2
    manifest["source_archive"] = "sources"
    # Snapshot at launch: later workspace edits cannot silently change a live run's provenance.
    for relative, expected in manifest["source_sha256"].items():
        source = repository / relative
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Source changed during snapshot: {relative}")
        target = directory / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    dump_json(directory / "manifest.json", manifest)


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def finalize_run(directory: Path, manifest: dict, raw_rows, messages,
                 correctness: dict, summary, failures) -> None:
    for name, rows in [("raw.jsonl", raw_rows), ("failures.jsonl", failures)]:
        with (directory / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(directory / "messages.csv", MESSAGE_FIELDS, messages)
    write_csv(directory / "summary.csv", list(summary[0]) if summary else
              ["scheme", "tamper_type", "trials", "rejected", "accepted"], summary)
    dump_json(directory / "correctness.json", correctness)
    manifest["output_sha256"] = {name: digest(directory / name) for name in
                                  ["raw.jsonl", "messages.csv", "correctness.json",
                                   "summary.csv", "failures.jsonl"]}
    manifest["execution_status"] = "completed"
    dump_json(directory / "manifest.json", manifest)


def round_rows(result, *, scheme, run_id, seed, group, dimension, m, round_t, repeat=0):
    return [{"scheme": scheme, "run_id": run_id, "seed": seed, "group": group,
             "repeat": repeat, "N_selected": result.n_selected, "N_uploaded": result.n_uploaded,
             "M": m, "d": dimension, "round": round_t, "phase": p.phase,
             "role": p.role, "actor": p.actor, "detail": p.detail,
             "elapsed_ms": p.elapsed_ms, "sent_bytes": None,
             "success": result.completed, "verified": result.verified,
             "error_reason": result.failure_reason} for p in result.phases]


def message_rows(result, *, scheme, run_id, group, round_t):
    return [{"scheme": scheme, "run_id": run_id, "group": group, "round": round_t,
             "link": x.link, "direction": x.direction, "phase": x.phase,
             "payload_kind": x.payload_kind, "sender": x.sender, "receiver": x.receiver,
             "num_bytes": x.num_bytes, "note": x.note} for x in result.messages]
