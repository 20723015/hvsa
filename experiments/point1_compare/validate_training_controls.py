"""Recompute control transitions, predictions and a declared SGD sample."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .training_control_contract import (ROOT, SOURCE, SOURCE_SHA, N, D, SCALE,
    sha, design, exact_sum, load_source, source_arrays)


def check_payload(a, entry, previous, historical, historical_row, initial):
    """Pure checks also exercised against rehashed semantic forgeries."""
    required = {'encoded_base', 'float_base', 'encoded_locals', 'float_locals', 'integers',
        'integer_sum', 'encoded_mean', 'float_mean', 'encoded_next', 'float_next',
        'encoded_predictions', 'float_predictions'}
    if set(a) != required:
        raise ValueError('Unexpected training payload fields')
    for k in ['encoded_base', 'float_base', 'encoded_next', 'float_next']:
        if a[k].shape != (D,) or a[k].dtype != np.float32 or not np.isfinite(a[k]).all():
            raise ValueError('Invalid model domain: '+k)
    for k in ['encoded_locals', 'float_locals']:
        if a[k].shape != (N, D) or a[k].dtype != np.float32 or not np.isfinite(a[k]).all():
            raise ValueError('Invalid local model domain: '+k)
    seed, r = entry['seed'], entry['round']
    if entry['batch_seeds'] != [seed*1000000+r*N+i for i in range(N)]:
        raise ValueError('Batch seed schedule differs')
    order = ['encoded', 'float'] if (seed+r) % 2 == 0 else ['float', 'encoded']
    if entry['order'] != order:
        raise ValueError('Branch execution order differs')
    for branch in ['encoded', 'float']:
        expected_base = initial if previous is None else previous[branch+'_next']
        if not np.array_equal(a[branch+'_base'], expected_base):
            raise ValueError('Branch state was reset or cross-fed: '+branch)
        if not np.isfinite(entry['training_ms'][branch]) or entry['training_ms'][branch] < 0:
            raise ValueError('Invalid training time')
    if not np.array_equal(a['encoded_base'], historical['base_model']):
        raise ValueError('Historical base differs')
    integers = (a['encoded_locals']*SCALE).astype(np.int64)
    if a['integers'].dtype != np.int64 or not np.array_equal(integers, a['integers']):
        raise ValueError('Quantization differs')
    if not np.array_equal(integers, historical['vectors']):
        raise ValueError('Historical encoded locals differ')
    total = exact_sum(integers)
    if a['integer_sum'].dtype != np.int64 or not np.array_equal(total, a['integer_sum']):
        raise ValueError('Integer aggregate differs')
    if hashlib.sha256(total.astype('<i8').tobytes()).hexdigest() != historical_row['raw_sum_sha256']:
        raise ValueError('Historical raw sum differs')
    encoded_mean = total.astype(np.float64)/SCALE/N
    float_mean = np.mean(a['float_locals'].astype(np.float64), axis=0)
    for branch, mean in [('encoded', encoded_mean), ('float', float_mean)]:
        if a[branch+'_mean'].dtype != np.float64 or not np.array_equal(mean, a[branch+'_mean']):
            raise ValueError('Branch aggregation differs: '+branch)
        if not np.array_equal(mean.astype(np.float32), a[branch+'_next']):
            raise ValueError('Next model differs: '+branch)
        if a[branch+'_predictions'].shape != (10000,) or a[branch+'_predictions'].dtype != np.uint8:
            raise ValueError('Prediction domain differs')
    if r == 0 and not np.array_equal(a['encoded_locals'], a['float_locals']):
        raise ValueError('Identical initial training inputs did not reproduce identical locals')
    if not entry['historical_base_match'] or not entry['historical_encoded_match']:
        raise ValueError('Historical match flags differ')
    delta = a['float_next'].astype(np.float64)-a['encoded_next'].astype(np.float64)
    expected = dict(parameter_linf=float(np.max(np.abs(delta))),
        parameter_l2=float(np.linalg.norm(delta)),
        prediction_disagreements=int(np.count_nonzero(a['encoded_predictions'] != a['float_predictions'])),
        own_local_models_identical=bool(np.array_equal(a['encoded_locals'], a['float_locals'])))
    if any(entry[k] != v for k, v in expected.items()):
        raise ValueError('Reported branch difference differs from actual models')


def replay_client(base, dataset, partition, batch_seed):
    """Independent implementation, one predeclared client per branch/round."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    model = nn.Linear(784, 10)
    torch.nn.utils.vector_to_parameters(torch.tensor(base, dtype=torch.float32), model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    generator = torch.Generator().manual_seed(batch_seed)
    for images, labels in DataLoader(Subset(dataset, partition.tolist()), batch_size=64,
                                    shuffle=True, generator=generator):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(images.flatten(1)), labels).backward()
        optimizer.step()
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy()


def evaluate_vector(vector, testing):
    """Same 512-image evaluation batches, independent of stored predictions."""
    import torch
    from torch import nn
    model = nn.Linear(784, 10)
    torch.nn.utils.vector_to_parameters(torch.tensor(vector, dtype=torch.float32), model.parameters())
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(testing), 512):
            images = testing.data[start:start+512].to(torch.float32).div(255).flatten(1)
            output.append(model(images).argmax(1).numpy().astype(np.uint8))
    return np.concatenate(output)


def validate(directory, data_root=None, progress=None):
    import torch
    from torch import nn
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    root = Path(directory)
    try:
        m = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if m['design'] != design(m['design']['smoke']) or m['execution_status'] != 'completed':
            raise ValueError('Design or completion status differs')
        if m['historical_source'] != SOURCE or m['historical_manifest_sha256'] != SOURCE_SHA:
            raise ValueError('Historical source was replaced')
        if m['new_secure_protocol_calls'] != 0 or m['historical_full_edge_checks'] != 300:
            raise ValueError('Secure execution scope differs')
        source, old, historical_rows = load_source()
        if m['historical_outputs'] != old['output_sha256']:
            raise ValueError('Historical output inventory differs')
        expected_sources = {'docs/POINT1_TRAINING_CONTROL_STAGE_20261026.md',
            'experiments/point1_compare/training_control_contract.py',
            'experiments/point1_compare/run_training_controls.py',
            'experiments/point1_compare/validate_training_controls.py',
            'experiments/point1_compare/report_training_controls.py',
            'experiments/point1_compare/artifacts.py',
            'experiments/point1_compare/multirun_environment.py',
            'tests/test_point1_training_controls.py'}
        if set(m['source_sha256']) != expected_sources:
            raise ValueError('Source inventory differs')
        for name, digest in m['source_sha256'].items():
            if sha(root/'sources'/name) != digest:
                raise ValueError('Archived source changed')
        actual_files = {p.relative_to(root).as_posix() for sub in ['initial', 'rounds']
                        for p in (root/sub).iterdir()} | {'metrics.json'}
        if actual_files != set(m['output_sha256']):
            raise ValueError('Unexpected or omitted output files')
        for name, digest in m['output_sha256'].items():
            if sha(root/name) != digest:
                raise ValueError('Output hash mismatch: '+name)
        data_root = Path(data_root or ROOT/'data')
        if m['dataset_sha256'] != old['dataset_sha256']:
            raise ValueError('Dataset inventory differs')
        for name, digest in m['dataset_sha256'].items():
            if sha(data_root/name) != digest:
                raise ValueError('Dataset changed')
        torch.set_num_threads(1)
        training = MNIST(str(data_root), train=True, download=False, transform=ToTensor())
        testing = MNIST(str(data_root), train=False, download=False, transform=ToTensor())
        metrics = json.loads((root/'metrics.json').read_text(encoding='utf-8'))
        spec = m['design']
        expected = [(s, r) for s in spec['seeds'] for r in range(spec['rounds'])]
        if [(e['seed'], e['round']) for e in metrics] != expected:
            raise ValueError('Training coverage differs')
        expected_names = {'metrics.json'} | {f'initial/seed_{s}.npz' for s in spec['seeds']}
        for seed, r in expected:
            expected_names.update(f'rounds/seed_{seed}_round_{r:03d}.'+suffix for suffix in ['npz', 'json'])
        if actual_files != expected_names:
            raise ValueError('Training payload names differ')
        previous = {}
        for count, e in enumerate(metrics, 1):
            seed, r = e['seed'], e['round']
            with np.load(root/'initial'/f'seed_{seed}.npz', allow_pickle=False) as archive:
                initial, partitions = archive['model'], archive['partitions']
            torch.manual_seed(seed)
            model = nn.Linear(784, 10)
            expected_initial = torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy()
            expected_partitions = np.stack(np.array_split(np.random.default_rng(seed).permutation(60000), N))
            if not np.array_equal(initial, expected_initial) or not np.array_equal(partitions, expected_partitions):
                raise ValueError('Initial model or IID partition differs')
            payload = f'rounds/seed_{seed}_round_{r:03d}.npz'
            if e['payload'] != payload or e['payload_sha256'] != sha(root/payload):
                raise ValueError('Entry payload ownership differs')
            sidecar = json.loads((root/'rounds'/f'seed_{seed}_round_{r:03d}.json').read_text(encoding='utf-8'))
            if sidecar != e:
                raise ValueError('Metrics differ from immutable round entry')
            old_row = historical_rows[(seed, r)]
            if e['historical_snapshot'] != old_row['snapshot'] or e['historical_snapshot_sha256'] != old_row['snapshot_sha256']:
                raise ValueError('Historical round ownership differs')
            with np.load(root/payload, allow_pickle=False) as archive:
                a = {k: archive[k] for k in archive.files}
            historical = source_arrays(source, old_row)
            check_payload(a, e, previous.get(seed), historical, old_row, initial)
            for branch in ['encoded', 'float']:
                pred = evaluate_vector(a[branch+'_next'], testing)
                if not np.array_equal(pred, a[branch+'_predictions']):
                    raise ValueError('Stored predictions differ from actual next model')
                correct = int(np.count_nonzero(pred == testing.targets.numpy()))
                if correct != e[branch+'_correct'] or e[branch+'_accuracy'] != correct/10000:
                    raise ValueError('Test correct count differs')
                client = (seed+r) % N
                replayed = replay_client(a[branch+'_base'], training, partitions[client], e['batch_seeds'][client])
                if not np.array_equal(replayed, a[branch+'_locals'][client]):
                    raise ValueError('Independent sampled SGD differs: '+branch)
            if e['encoded_accuracy'] != old_row['test_accuracy']:
                raise ValueError('Historical accuracy differs')
            if e['float_minus_encoded_pp'] != (e['float_correct']-e['encoded_correct'])/100:
                raise ValueError('Accuracy difference differs')
            previous[seed] = a
            if progress and (count % 10 == 0 or count == len(metrics)):
                progress(f'VALIDATED_PAIRS={count}/{len(metrics)} sampled_sgd={count*2}')
        return dict(valid=True, errors=[], paired_rounds=len(metrics),
            historical_snapshot_matches=len(metrics), predictions_recomputed=2*len(metrics),
            independent_sampled_sgd_clients=2*len(metrics), sample_rule='client=(seed+round)%20 per branch',
            full_local_sgd_reexecution=False, new_secure_protocol_calls=0,
            historical_manifest_sha256=SOURCE_SHA, manifest_sha256=sha(root/'manifest.json'))
    except (ValueError, KeyError, OSError, TypeError, AssertionError) as exc:
        return dict(valid=False, errors=[str(exc)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    result = validate(args.directory, progress=lambda s: print(s, flush=True))
    print(json.dumps(result, indent=2))
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
