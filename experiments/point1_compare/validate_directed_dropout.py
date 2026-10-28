"""Reconcile directed diagnosis against inputs, full verifier/recipient ledgers."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from .directed_dropout_contract import (N, M, D, SEED, REPEATS, THRESHOLD, GROUPS,
    USERS, SCHEMES, GPVO, VLMRS, ROLES, STAGE, DECISIONS, specifications, tasks,
    vectors as input_vectors, round_id, expectation, expected_reveals, frozen_sources, summary)
from .adapters.common import vector_digest, reference_sum
from .artifacts import digest


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _safe(root, relative):
    target = (root/relative).resolve()
    _require(root.resolve() in target.parents, 'Path escapes run directory')
    return target


def _close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-6)


def validate(directory):
    try:
        _validate(Path(directory))
        return []
    except (AssertionError, KeyError, ValueError, TypeError, OSError, IndexError) as error:
        return [str(error) or type(error).__name__]


def _validate(root):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    _require(manifest['experiment'] == 'directed_dropout' and manifest['execution_status'] == 'completed', 'Incomplete or wrong experiment')
    _require(manifest['shape'] == dict(N=N, M=M, d=D, repeats=REPEATS, threshold=THRESHOLD), 'Changed shape')
    _require(manifest['seed'] == SEED and manifest['schemes'] == SCHEMES and manifest['groups'] == GROUPS and manifest['users'] == USERS, 'Changed participants')
    _require(manifest['cases'] == specifications() and manifest['planned_order'] == [list(t) for t in tasks()] and manifest['expected_calls'] == len(tasks()), 'Changed coverage plan')
    _require(manifest['performance_qualified'] is False and manifest['author_certified'] is False, 'Overstated qualification')
    _require(manifest['stage'] == STAGE and STAGE in manifest['source_sha256'], 'Missing archived stage')
    repo = Path(__file__).resolve().parents[2]
    frozen = frozen_sources(repo)
    _require(manifest['critical_source_sha256'] == frozen, 'Changed frozen source inventory')
    _require(all(manifest['source_sha256'].get(rel) == sha for rel, sha in frozen.items()), 'Frozen adapter mismatch')
    for relative, sha in manifest['source_sha256'].items():
        _require(digest(_safe(root, 'sources/'+relative)) == sha, 'Archived source mismatch: '+relative)
    _require(manifest['decisions_sha256'] == frozen[DECISIONS] and manifest['decisions'] == json.loads((root/'sources'/DECISIONS).read_text(encoding='utf-8')), 'Changed baseline decisions')
    expected_files = {'raw.jsonl', 'messages.csv', 'correctness.json', 'summary.csv', 'failures.jsonl'}
    _require(set(manifest['output_sha256']) == expected_files, 'Missing output digest')
    for name, sha in manifest['output_sha256'].items():
        _require(digest(root/name) == sha, 'Output digest mismatch: '+name)
    params = manifest['gpvo_public_parameters']
    import gmpy2
    _require(params['N'].bit_length() == params['modulus_bits'] == 3072 and params['e'].bit_length() == params['e_bits'], 'Wrong public parameter size')
    _require(params['e'] > params['N']**2 and bool(gmpy2.is_prime(params['e'])) and math.gcd(params['g'], params['N']) == 1, 'Wrong public parameter domain')
    _require(params['maximal_order_certified'] is False and params['e_is_prime'] is True and params['g_is_unit'] is True, 'Wrong public parameter claim')
    correctness = json.loads((root/'correctness.json').read_text(encoding='utf-8'))
    obs, inits = correctness['observations'], correctness['initialization']
    _require(correctness['experiment'] == 'directed_dropout', 'Wrong correctness artifact')
    _require([(o['case'], o['index'], o['scheme']) for o in obs] == tasks(), 'Missing or reordered observations')
    _require([i['scheme'] for i in inits] == SCHEMES, 'Missing or repeated initialization')
    raw = [json.loads(line) for line in (root/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    with (root/'messages.csv').open(encoding='utf-8', newline='') as stream:
        messages = list(csv.DictReader(stream))
    rg, mg = defaultdict(list), defaultdict(list)
    for row in raw:
        _require(row['run_id'] == manifest['run_id'] and row['seed'] == SEED and row['d'] == D and row['role'] in ROLES and row['actor'], 'Raw ownership mismatch')
        _require(math.isfinite(row['elapsed_ms']) and row['elapsed_ms'] >= 0, 'Invalid elapsed work')
        rg[row['group']].append(row)
    for row in messages:
        _require(row['run_id'] == manifest['run_id'] and row['direction'] == 'send' and row['sender'] and row['receiver'] and int(row['num_bytes']) > 0, 'Message ownership mismatch')
        mg[row['group']].append(row)
    groups = {o['group'] for o in obs}|{i['group'] for i in inits}
    # Zero-message groups are allowed structurally; per-call byte totals and
    # actual senders are still checked. This HVSA run sends one TA mask even
    # with no uploads, while the cloud's edge barrier remains unsatisfied.
    _require(set(rg) == groups and set(mg) <= groups, 'Missing raw groups or extra message groups')

    def metering(item, scheme, round_t, selected, uploaded, completed, verified, reason, repeat):
        rr, mm = rg[item['group']], mg[item['group']]
        for row in rr:
            _require(row['scheme'] == scheme and row['round'] == round_t and row['repeat'] == repeat and row['N_selected'] == selected and row['N_uploaded'] == uploaded and row['success'] == completed and row['verified'] == verified and row['error_reason'] == reason and row['M'] == (None if scheme == VLMRS else M), 'Phase result ownership mismatch')
        for row in mm:
            _require(row['scheme'] == scheme and int(row['round']) == round_t, 'Message round mismatch')
        roles = {r:sum(row['elapsed_ms'] for row in rr if row['role'] == r) for r in ROLES}
        _require(set(item['role_ms']) == set(ROLES) and all(_close(roles[r], item['role_ms'][r]) for r in ROLES), 'Role cost mismatch')
        _require(sum(int(row['num_bytes']) for row in mm) == item['total_sent_bytes'], 'Communication sum mismatch')
        return rr, mm

    for item in inits:
        _require(item['group'] == 'init:'+item['scheme'], 'Wrong initialization identity')
        metering(item, item['scheme'], -1, N, 0, True, None, '', -1)
    snapshots = {}
    for index in range(REPEATS):
        path = root/'inputs'/f'pool_{index}.npz'
        with np.load(path, allow_pickle=False) as data:
            _require(data.files == ['user_ids', 'vectors'] and data['user_ids'].tolist() == USERS and data['vectors'].shape == (N,D) and data['vectors'].dtype == np.dtype('int64'), 'Input pool shape mismatch')
            original = input_vectors(index)
            _require(np.array_equal(data['vectors'], np.stack([original[u] for u in USERS])), 'Input pool changed')
        snapshots[index] = (original, digest(path), vector_digest(original))
    expected_failures, outputs = [], set()
    for o in obs:
        sp = next(s for s in specifications() if s['case'] == o['case'])
        scheme, index = o['scheme'], o['index']
        _require(o['group'] == f"{o['case']}:{index}:{scheme}" and o['protocol_round'] == round_id(o['case'], index), 'Round identity mismatch')
        _require(o['uploaded'] == sp['uploaded'] and o['recipients'] == sp['recipients'], 'Changed dropout roster')
        _require(o['n_selected'] == N and o['n_uploaded'] == len(sp['uploaded']), 'Participant count mismatch')
        _require(o['group_upload_counts'] == [len(set(g)&set(sp['uploaded'])) for g in GROUPS], 'Group count mismatch')
        pool, snap_sha, pool_sha = snapshots[index]
        _require(Path(o['snapshot']).as_posix() == f'inputs/pool_{index}.npz' and o['snapshot_sha256'] == snap_sha and o['input_sha256'] == pool_sha, 'Input linkage mismatch')
        reference = reference_sum([pool[u] for u in sp['uploaded']]) if sp['uploaded'] else np.zeros(D, dtype=np.int64)
        _require(o['reference_sha256'] == hashlib.sha256(reference.astype('<i8').tobytes()).hexdigest(), 'Reference sum mismatch')
        exact = False
        if o['output'] is not None:
            _require(Path(o['output']).as_posix() == 'outputs/'+o['group'].replace(':','_')+'.npy', 'Output identity mismatch')
            path = _safe(root, o['output'])
            outputs.add(path.resolve())
            _require(digest(path) == o['output_sha256'], 'Output digest mismatch')
            actual = np.load(path, allow_pickle=False)
            _require(actual.shape == (D,) and actual.dtype == np.dtype('int64'), 'Output shape mismatch')
            exact = bool(np.array_equal(actual, reference))
        else:
            _require(o['output_sha256'] is None and not o['completed'], 'Missing returned sum')
        _require(o['exact_match'] is exact, 'Incorrect exactness claim')
        valid = bool(o['completed'] and exact and o['verified'] is True)
        _require(o['valid_aggregate'] is valid, 'Incorrect aggregate validity claim')
        expect_complete, reason = expectation(sp, scheme)
        matches = bool(valid if expect_complete else not o['completed'] and o['output'] is None and o['verified'] is None and o['failure_reason'] == reason)
        _require(o['expected_completed'] is expect_complete and o['matches_expected_outcome'] is matches, 'Expected/unexpected outcome hidden')
        rr, mm = metering(o, scheme, o['protocol_round'], N, len(sp['uploaded']), o['completed'], o['verified'], o['failure_reason'], index)
        _require(math.isfinite(o['wall_ms']) and o['wall_ms'] >= sum(o['role_ms'].values())-.1, 'Invalid round wall time')
        links = defaultdict(int)
        for row in mm:
            links[row['link']] += int(row['num_bytes'])
        _require(dict(links) == o['link_bytes'], 'Link byte mismatch')
        upload_kind = 'masked_vector_and_proof' if scheme == VLMRS else 'masked_upload'
        _require(Counter(row['sender'] for row in mm if row['payload_kind'] == upload_kind) == Counter(sp['uploaded']), 'Upload sender ledger mismatch')
        if scheme != VLMRS:
            evidence_kind = 'pre_aggregation_with_evidence' if scheme == 'hvsa' else 'group_sum_commitment'
            active_edges = [f'edge_{i}' for i,g in enumerate(GROUPS) if set(g)&set(sp['uploaded'])]
            _require(Counter(row['sender'] for row in mm if row['payload_kind'] == evidence_kind) == Counter(active_edges), 'Active-group evidence ledger mismatch')
        verifier_phase = 'verify_all_equations' if scheme == VLMRS else 'edge_verify'
        verifier_role = 'user' if scheme == VLMRS else 'edge'
        verifier_rows = [row for row in rr if row['phase'] == verifier_phase and row['role'] == verifier_role]
        expected_verifiers = sp['recipients'] if scheme == VLMRS else [f'edge_{i}' for i in range(M)]
        if o['completed']:
            _require(set(o['verdicts']) == set(expected_verifiers) and all(type(v) is bool for v in o['verdicts'].values()), 'Missing designated verifier verdict')
            _require(Counter(row['actor'] for row in verifier_rows) == Counter(expected_verifiers), 'Full verification not executed')
            _require(o['verified'] is all(o['verdicts'].values()), 'Wrong combined verdict')
        else:
            _require(o['verified'] is None and o['output'] is None and not o['verdicts'] and not verifier_rows, 'Unavailable run claimed verification')
        kind = 'aggregate_and_proof' if scheme == VLMRS else 'verified_model'
        received = sorted(row['receiver'] for row in mm if row['payload_kind'] == kind)
        _require(o['actual_recipients'] == received, 'Recipient ledger mismatch')
        delivery = sp['recipients'] if o['completed'] and o['verified'] is True else []
        _require(received == delivery, 'False model delivery claim')
        reveals = sum(row['payload_kind'] == 'dropout_mask_reveal' for row in mm)
        _require(o['reveal_count'] == reveals and o['expected_reveal_count'] == (expected_reveals(sp['uploaded']) if scheme == GPVO else 0), 'Reveal count mismatch')
        # Reveal deviations are observable protocol outcomes, not erased to force an expectation.
        if not valid:
            expected_failures.append(dict(group=o['group'], scheme=scheme, case=o['case'], index=index,
                reason=o['failure_reason'] or 'incorrect or unverified aggregate',
                expected_unavailability=not expect_complete, matches_expected_outcome=matches))
    _require({p.resolve() for p in (root/'outputs').glob('*.npy')} == outputs, 'Unreferenced returned sum')
    _require({p.name for p in (root/'inputs').glob('*.npz')} == {f'pool_{i}.npz' for i in range(REPEATS)}, 'Unreferenced input pool')
    failures = [json.loads(line) for line in (root/'failures.jsonl').read_text(encoding='utf-8').splitlines()]
    _require(failures == expected_failures, 'Unavailable or invalid outcomes hidden')
    _require(correctness['valid_aggregates'] == sum(o['valid_aggregate'] for o in obs) and correctness['unavailable'] == sum(not o['completed'] for o in obs) and correctness['unexpected_outcomes'] == sum(not o['matches_expected_outcome'] for o in obs), 'Incorrect aggregate totals')
    with (root/'summary.csv').open(encoding='utf-8', newline='') as stream:
        actual_summary = list(csv.DictReader(stream))
    _require(actual_summary == [{k:str(v) for k,v in row.items()} for row in summary(obs)], 'Summary mismatch')


if __name__ == '__main__':
    errors = validate(Path(sys.argv[1]))
    print(json.dumps(dict(valid=not errors, errors=errors), ensure_ascii=False))
    raise SystemExit(int(bool(errors)))
