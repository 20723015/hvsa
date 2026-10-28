"""Paired directed availability diagnosis; no performance ranking or re-sampling."""
from __future__ import annotations
import atexit
from collections import defaultdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from .directed_dropout_contract import (N, M, D, SEED, REPEATS, THRESHOLD, GROUPS,
    USERS, SCHEMES, GPVO, VLMRS, ROLES, STAGE, DECISIONS, specifications, tasks,
    vectors as input_vectors, round_id, expectation, expected_reveals, frozen_sources, summary)
from .adapters.common import RoundResult, reference_sum, vector_digest
from .adapters.hvsa_point1 import HVSAProtocol
from .adapters.gpvo_decided import GPVODecidedProtocol
from .artifacts import create_run, digest, dump_json, finalize_run, round_rows, message_rows
from .run_compare import ROOT, _git_commit, _lib_versions, _new_run_id, _review_metadata
from .watchdog import RoundDeadline
from utils import LHH


def main():
    import psutil
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process.pid != os.getpid() and (process.info['name'] or '').lower() in ['python.exe', 'python']:
                command = process.info['cmdline'] or []
                if any(x.startswith('experiments.point1_compare.run_') or x == 'experiments.point1_compare.continue_decided_trajectories' for x in command):
                    raise RuntimeError('Another protocol worker is active: '+str(process.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    frozen = frozen_sources(ROOT)
    for rel, sha in frozen.items():
        if digest(Path(ROOT)/rel) != sha:
            raise ValueError('Frozen source changed: '+rel)
    sys.path.insert(0, str(Path(ROOT)/'tmp/point1_pairing_deps'))
    import py_ecc
    import gmpy2
    from .adapters.vlmrs_fullgraph import VLMRSFullGraphProtocol
    logging.disable(logging.CRITICAL)
    run_id = _new_run_id('directed_dropout')
    manifest = dict(run_id=run_id, experiment='directed_dropout',
        command=' '.join(sys.argv), git_commit=_git_commit(), seed=SEED,
        shape=dict(N=N, M=M, d=D, repeats=REPEATS, threshold=THRESHOLD),
        schemes=SCHEMES, groups=GROUPS, users=USERS, cases=specifications(),
        planned_order=tasks(), expected_calls=len(tasks()),
        lib_versions={**_lib_versions(), 'py_ecc':py_ecc.__version__, 'gmpy2':gmpy2.version()},
        decisions=json.loads((Path(ROOT)/DECISIONS).read_text(encoding='utf-8')),
        decisions_sha256=digest(Path(ROOT)/DECISIONS), critical_source_sha256=frozen,
        stage=STAGE, round_timeout_seconds=1800, execution_status='running',
        performance_qualified=False, author_certified=False,
        scope='functional availability diagnosis, one initialization per scheme, all designated verifiers',
        inputs_scope='three common input pools, reused across twelve cases, not 36 independent pools',
        dropout_stage='before_upload occurs after required roster/share distribution; TA/edges stay online',
        timing_scope='local sequential replay; costs retained for evidence, no performance ranking')
    manifest.update(_review_metadata(manifest['shape']))
    manifest['implementation_variants'] = {s:s for s in SCHEMES}
    manifest['measurement_limitations'] = [
        'Declared baseline instances, not author code or security-equivalence certification.',
        'One long-lived initialized state per scheme; three input pools are reused across twelve cases.',
        'Functional diagnosis only; local timings are not a controlled performance distribution.',
        'Before-upload dropout follows roster/share distribution; edges and TA remain available.',
        'GPVO skips empty groups in this adapter; no original-paper availability theorem is inferred.',
        'Previously recorded conditional integrity/privacy behavior and backend differences remain.']
    manifest['source_sha256'][STAGE] = digest(Path(ROOT)/STAGE)
    test_file = 'tests/test_point1_directed_dropout.py'
    manifest['source_sha256'][test_file] = digest(Path(ROOT)/test_file)
    directory = Path(ROOT)/'results/point1_compare'/run_id
    create_run(directory, manifest, Path(ROOT))
    for name in ['inputs', 'outputs', 'checkpoints']:
        (directory/name).mkdir()
    print('RUN_DIR='+str(directory), flush=True)
    raw, messages, observations, failures, initialization = [], [], [], [], []
    context = {'phase':'setup'}

    def save_prefix():
        dump_json(directory/'checkpoints/completed_prefix.json', dict(raw=raw, messages=messages,
            observations=observations, failures=failures, initialization=initialization, context=context))

    def stop(status, reason):
        if manifest['execution_status'] == 'running':
            save_prefix()
            manifest.update(execution_status=status, termination=dict(context=context, reason=reason))
            dump_json(directory/'manifest.json', manifest)

    atexit.register(lambda: stop('interrupted', 'driver exited before finalization'))
    try:
        states = {}
        LHH._generator_for.cache_clear()
        for scheme in SCHEMES:
            context = dict(phase='initialization', scheme=scheme)
            with RoundDeadline(1800, lambda: stop('timed_out', 'initialization deadline')):
                state = (HVSAProtocol(D, GROUPS) if scheme == 'hvsa' else
                         GPVODecidedProtocol(D, GROUPS) if scheme == GPVO else
                         VLMRSFullGraphProtocol(D, USERS, threshold=THRESHOLD, root_exponent=7))
            states[scheme] = state
            result = RoundResult(scheme, True, None, None, N, 0, state.init_phases, state.init_messages)
            group = 'init:'+scheme
            raw += round_rows(result, scheme=scheme, run_id=run_id, seed=SEED, group=group,
                              dimension=D, m=None if scheme == VLMRS else M, round_t=-1, repeat=-1)
            messages += message_rows(result, scheme=scheme, run_id=run_id, group=group, round_t=-1)
            initialization.append(dict(scheme=scheme, group=group,
                role_ms={r:result.phase_ms(r) for r in ROLES},
                total_sent_bytes=sum(m.num_bytes for m in result.messages)))
            save_prefix()
            print('INITIALIZED='+scheme, flush=True)
        params = states[GPVO].params
        manifest['gpvo_public_parameters'] = dict(N=params.n_mod, g=params.g, e=params.e,
            modulus_bits=params.n_mod.bit_length(), e_bits=params.e.bit_length(),
            e_is_prime=bool(gmpy2.is_prime(params.e)), g_is_unit=math.gcd(params.g, params.n_mod)==1,
            maximal_order_certified=False)
        dump_json(directory/'manifest.json', manifest)
        snapshots = {}
        for index in range(REPEATS):
            pool = input_vectors(index)
            snapshot = directory/'inputs'/f'pool_{index}.npz'
            np.savez(snapshot, user_ids=np.array(USERS), vectors=np.stack([pool[u] for u in USERS]))
            snapshots[index] = dict(snapshot=str(snapshot.relative_to(directory)),
                snapshot_sha256=digest(snapshot), input_sha256=vector_digest(pool))
        for case, index, scheme in tasks():
            sp = next(s for s in specifications() if s['case'] == case)
            pool = input_vectors(index)
            uploaded, recipients = set(sp['uploaded']), set(sp['recipients'])
            reference = reference_sum([pool[u] for u in sorted(uploaded)]) if uploaded else np.zeros(D, dtype=np.int64)
            context = dict(phase='round', case=case, index=index, scheme=scheme,
                           protocol_round=round_id(case, index), **snapshots[index])
            tick = time.perf_counter()
            with RoundDeadline(1800, lambda: stop('timed_out', 'round deadline')):
                result = (states[scheme].run_round(pool, uploaded, context['protocol_round'], recovery_clients=recipients)
                          if scheme == VLMRS else states[scheme].run_round(pool, uploaded, context['protocol_round'], model_recipients=recipients))
            wall_ms = (time.perf_counter()-tick)*1000
            exact = bool(result.raw_sum is not None and np.array_equal(result.raw_sum, reference))
            valid = result.completed and exact and result.verified is True
            expected_complete, expected_reason = expectation(sp, scheme)
            matches = bool(valid if expected_complete else
                (not result.completed and result.raw_sum is None and result.verified is None and result.failure_reason == expected_reason))
            group = f'{case}:{index}:{scheme}'
            raw += round_rows(result, scheme=scheme, run_id=run_id, seed=SEED, group=group,
                              dimension=D, m=None if scheme == VLMRS else M,
                              round_t=context['protocol_round'], repeat=index)
            messages += message_rows(result, scheme=scheme, run_id=run_id, group=group, round_t=context['protocol_round'])
            verdicts = result.extra.get('verification_verdicts' if scheme == VLMRS else 'per_edge_verified', {})
            if scheme == GPVO:
                verdicts = {'edge_'+key:value for key, value in verdicts.items()}
            else:
                verdicts = {key:bool(value) for key, value in verdicts.items()}
            output = None
            if result.raw_sum is not None:
                output = directory/'outputs'/f'{group.replace(":", "_")}.npy'
                np.save(output, result.raw_sum, allow_pickle=False)
            kind = 'aggregate_and_proof' if scheme == VLMRS else 'verified_model'
            actual_recipients = sorted(m.receiver for m in result.messages if m.payload_kind == kind)
            links = defaultdict(int)
            for message in result.messages:
                links[message.link] += message.num_bytes
            reveal_count = sum(m.payload_kind == 'dropout_mask_reveal' for m in result.messages)
            observation = dict(scheme=scheme, case=case, index=index, group=group,
                protocol_round=context['protocol_round'], **snapshots[index],
                uploaded=sorted(uploaded), recipients=sorted(recipients),
                group_upload_counts=[len(set(g)&uploaded) for g in GROUPS],
                n_selected=result.n_selected, n_uploaded=result.n_uploaded,
                completed=result.completed, verified=result.verified, exact_match=exact,
                valid_aggregate=valid, expected_completed=expected_complete,
                matches_expected_outcome=matches, failure_reason=result.failure_reason,
                verdicts=verdicts, actual_recipients=actual_recipients,
                output=str(output.relative_to(directory)) if output else None,
                output_sha256=digest(output) if output else None,
                reference_sha256=hashlib.sha256(reference.astype('<i8').tobytes()).hexdigest(),
                reveal_count=reveal_count,
                expected_reveal_count=expected_reveals(uploaded) if scheme == GPVO else 0,
                wall_ms=wall_ms, role_ms={r:result.phase_ms(r) for r in ROLES},
                link_bytes=dict(links), total_sent_bytes=sum(links.values()))
            observations.append(observation)
            if not valid:
                failures.append(dict(group=group, scheme=scheme, case=case, index=index,
                    reason=result.failure_reason or 'incorrect or unverified aggregate',
                    expected_unavailability=not expected_complete, matches_expected_outcome=matches))
            save_prefix()
            print(f'PROGRESS={len(observations)}/{len(tasks())} {scheme} {case} index={index} completed={result.completed} exact={exact} verified={result.verified} expected={matches}', flush=True)
        correctness = dict(experiment='directed_dropout', observations=observations, initialization=initialization,
            valid_aggregates=sum(o['valid_aggregate'] for o in observations),
            unavailable=sum(not o['completed'] for o in observations),
            unexpected_outcomes=sum(not o['matches_expected_outcome'] for o in observations))
        finalize_run(directory, manifest, raw, messages, correctness, summary(observations), failures)
        from .validate_directed_dropout import validate
        problems = validate(directory)
        if problems:
            raise ValueError('Evidence reconciliation failed: '+repr(problems))
        print('COMPLETED='+str(directory), flush=True)
        return int(bool(correctness['unexpected_outcomes']))
    except BaseException as error:
        stop('exception', repr(error))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
