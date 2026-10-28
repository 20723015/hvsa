"""Independent plaintext training loops paired with archived full-HVSA E7."""
from __future__ import annotations
import argparse
import copy
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from .artifacts import dump_json
from .multirun_environment import snapshot
from .training_control_contract import (ROOT, SOURCE, SOURCE_SHA, STAGE, N, D,
    SCALE, design, sha, exact_sum, load_source, source_arrays)


def parameters(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy().copy()


def set_parameters(model, values):
    torch.nn.utils.vector_to_parameters(torch.tensor(values, dtype=torch.float32), model.parameters())


def train_locals(model, dataset, partitions, seed, round_t):
    """Each call actually trains from this branch's own incoming model."""
    values = []
    for i in range(N):
        local = copy.deepcopy(model)
        local.train()
        optimizer = torch.optim.SGD(local.parameters(), lr=.01)
        generator = torch.Generator().manual_seed(seed*1000000+round_t*N+i)
        loader = DataLoader(Subset(dataset, partitions[i].tolist()), batch_size=64,
                            shuffle=True, generator=generator)
        for images, labels in loader:
            optimizer.zero_grad()
            nn.functional.cross_entropy(local(images.flatten(1)), labels).backward()
            optimizer.step()
        values.append(parameters(local))
    return np.stack(values)


def predictions(model, dataset):
    model.eval()
    outputs = []
    with torch.no_grad():
        for images, _ in DataLoader(dataset, batch_size=512, shuffle=False):
            outputs.append(model(images.flatten(1)).argmax(1).numpy().astype(np.uint8))
    return np.concatenate(outputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    spec = design(args.smoke)
    torch.set_num_threads(1)
    source, old, observations = load_source()
    if torch.__version__ != old['lib_versions']['torch'] or np.__version__ != old['lib_versions']['numpy']:
        raise ValueError('Training libraries differ from archived source')
    data_root = ROOT/'data'
    for name, digest in old['dataset_sha256'].items():
        if sha(data_root/name) != digest:
            raise ValueError('Training dataset changed: '+name)
    training = MNIST(str(data_root), train=True, download=False, transform=ToTensor())
    testing = MNIST(str(data_root), train=False, download=False, transform=ToTensor())
    if len(training) != 60000 or len(testing) != 10000:
        raise ValueError('Unexpected MNIST sizes')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    run = ROOT/'results/point1_compare'/(stamp+'_training_controls'+('_smoke' if args.smoke else ''))
    run.mkdir(parents=True, exist_ok=False)
    (run/'rounds').mkdir()
    (run/'initial').mkdir()
    sources = [STAGE, 'experiments/point1_compare/training_control_contract.py',
        'experiments/point1_compare/run_training_controls.py',
        'experiments/point1_compare/validate_training_controls.py',
        'experiments/point1_compare/report_training_controls.py',
        'experiments/point1_compare/artifacts.py',
        'experiments/point1_compare/multirun_environment.py',
        'tests/test_point1_training_controls.py']
    source_sha = {}
    for rel in sources:
        data = (ROOT/rel).read_bytes()
        target = run/'sources'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        source_sha[rel] = sha(target)
    m = dict(experiment='point1_independent_training_controls', schema_version=1,
        design=spec, historical_source=SOURCE, historical_manifest_sha256=SOURCE_SHA,
        historical_outputs=old['output_sha256'], historical_full_edge_checks=300,
        dataset_sha256=old['dataset_sha256'], source_sha256=source_sha,
        new_secure_protocol_calls=0, execution_status='running', pid=os.getpid(),
        command=sys.argv, python=platform.python_version(),
        versions={k: importlib.metadata.version(k) for k in ['torch', 'torchvision', 'numpy']},
        limitations=['Reuses archived 60 full-HVSA calls, not new cryptographic execution.',
            'Only IID MNIST/linear model/full participation; three training seeds.',
            'Float64 mean is one specified FedAvg implementation, not all implementations.',
            'Stored local models are aggregation evidence; offline validation does not rerun all SGD.',
            'No protocol-performance or precision-SOTA claim.'])
    dump_json(run/'manifest.json', m)
    print('RUN_DIR='+str(run), flush=True)
    metrics = []
    for seed in spec['seeds']:
        torch.manual_seed(seed)
        encoded_model = nn.Linear(784, 10)
        float_model = copy.deepcopy(encoded_model)
        partitions = np.stack(np.array_split(np.random.default_rng(seed).permutation(len(training)), N))
        initial = parameters(encoded_model)
        np.savez_compressed(run/'initial'/f'seed_{seed}.npz', model=initial, partitions=partitions)
        for r in range(spec['rounds']):
            environment_before = snapshot()
            quant_base, float_base = parameters(encoded_model), parameters(float_model)
            old_row = observations[(seed, r)]
            old_arrays = source_arrays(source, old_row)
            if not np.array_equal(quant_base, old_arrays['base_model']):
                raise ValueError(f'Historical base not reproduced: seed={seed} round={r}')
            locals_by_branch, training_ms = {}, {}
            order = ['encoded', 'float'] if (seed+r) % 2 == 0 else ['float', 'encoded']
            for branch in order:
                tick = time.perf_counter()
                locals_by_branch[branch] = train_locals(
                    encoded_model if branch == 'encoded' else float_model,
                    training, partitions, seed, r)
                training_ms[branch] = (time.perf_counter()-tick)*1000
            encoded_local, float_local = locals_by_branch['encoded'], locals_by_branch['float']
            integers = (encoded_local*SCALE).astype(np.int64)
            if not np.array_equal(integers, old_arrays['vectors']):
                raise ValueError(f'Historical encoded updates not reproduced: seed={seed} round={r}')
            integer_sum = exact_sum(integers)
            encoded_mean = integer_sum.astype(np.float64)/SCALE/N
            float_mean = np.mean(float_local.astype(np.float64), axis=0)
            set_parameters(encoded_model, encoded_mean)
            set_parameters(float_model, float_mean)
            quant_next, float_next = parameters(encoded_model), parameters(float_model)
            quant_pred, float_pred = predictions(encoded_model, testing), predictions(float_model, testing)
            labels = testing.targets.numpy()
            quant_correct = int(np.count_nonzero(quant_pred == labels))
            float_correct = int(np.count_nonzero(float_pred == labels))
            if quant_correct/10000 != old_row['test_accuracy']:
                raise ValueError('Historical test accuracy not reproduced')
            delta = float_next.astype(np.float64)-quant_next.astype(np.float64)
            entry = dict(seed=seed, round=r, order=order, training_ms=training_ms,
                batch_seeds=[seed*1000000+r*N+i for i in range(N)],
                historical_snapshot=old_row['snapshot'], historical_snapshot_sha256=old_row['snapshot_sha256'],
                historical_encoded_match=True, historical_base_match=True,
                encoded_correct=quant_correct, float_correct=float_correct,
                encoded_accuracy=quant_correct/10000, float_accuracy=float_correct/10000,
                float_minus_encoded_pp=(float_correct-quant_correct)/100,
                parameter_linf=float(np.max(np.abs(delta))), parameter_l2=float(np.linalg.norm(delta)),
                prediction_disagreements=int(np.count_nonzero(quant_pred != float_pred)),
                own_local_models_identical=bool(np.array_equal(encoded_local, float_local)),
                environment_before=environment_before, environment_after=snapshot())
            stem = f'seed_{seed}_round_{r:03d}'
            path = run/'rounds'/(stem+'.npz')
            with path.open('xb') as stream:
                np.savez_compressed(stream, encoded_base=quant_base, float_base=float_base,
                    encoded_locals=encoded_local, float_locals=float_local, integers=integers,
                    integer_sum=integer_sum, encoded_mean=encoded_mean, float_mean=float_mean,
                    encoded_next=quant_next, float_next=float_next,
                    encoded_predictions=quant_pred, float_predictions=float_pred)
            entry['payload'] = path.relative_to(run).as_posix()
            entry['payload_sha256'] = sha(path)
            with (run/'rounds'/(stem+'.json')).open('x', encoding='utf-8') as stream:
                json.dump(entry, stream, indent=2, allow_nan=False)
            metrics.append(entry)
            print(f'seed={seed} round={r} historical_match=True encoded={quant_correct}/10000 '
                  f'float={float_correct}/10000 linf={entry["parameter_linf"]:.9g} '
                  f'disagree={entry["prediction_disagreements"]} committed={len(metrics)}', flush=True)
    dump_json(run/'metrics.json', metrics)
    m['output_sha256'] = {p.relative_to(run).as_posix(): sha(p) for sub in ['initial', 'rounds']
                         for p in sorted((run/sub).iterdir())}
    m['output_sha256']['metrics.json'] = sha(run/'metrics.json')
    m['execution_status'] = 'completed'
    dump_json(run/'manifest.json', m)
    from .validate_training_controls import validate
    receipt = validate(run)
    dump_json(run/'validation.json', receipt)
    if not receipt['valid']:
        raise ValueError(str(receipt))
    print(f'COMPLETED={run} pairs={len(metrics)} validation=passed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
