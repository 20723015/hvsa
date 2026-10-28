"""Presentation-only axis-unit repair; preserve first export and data bytes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .artifacts import dump_json
from .training_control_contract import sha


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    args = p.parse_args(argv)
    out = Path(args.directory).resolve()
    a = json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    run = Path(a['run'])
    if sha(run/'manifest.json') != a['manifest_sha256']:
        raise ValueError('Original data changed')
    m = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    for rel, digest in m['output_sha256'].items():
        if sha(run/rel) != digest:
            raise ValueError('Training output changed')
    history = out/'review_history/before_axis_unit_repair'
    history.mkdir(parents=True, exist_ok=False)
    for name in ['training_differences.png', 'training_differences.svg', 'analysis_manifest.json']:
        shutil.copyfile(out/name, history/name)
    shutil.copyfile(__file__, out/'repair_source.py')
    rows = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7))
    for seed, color in zip(m['design']['seeds'], ['#165a96', '#bd5e1e', '#477d43']):
        part = [e for e in rows if e['seed'] == seed]
        for ax, key, scale in zip(axes,
                ['float_minus_encoded_pp', 'parameter_linf', 'prediction_disagreements'], [1, 1e6, 1]):
            ax.plot(np.arange(1, 21), [e[key]*scale for e in part], marker='o',
                    ms=3.2, lw=1.5, color=color, label=f'Seed {seed}')
    titles = ['Float minus encoded accuracy', 'Maximum coordinate difference', 'Prediction disagreements']
    labels = ['Accuracy difference (percentage points)', 'Parameter absolute difference (×10⁻⁶)',
              'Count out of 10,000 test images']
    for ax, title, label in zip(axes, titles, labels):
        ax.set(title=title, xlabel='Communication round', ylabel=label, xticks=[1, 5, 10, 15, 20])
        ax.grid(alpha=.2)
    axes[0].axhline(0, color='gray', lw=.8)
    axes[0].set_ylim(-.012, .012)
    axes[1].set_ylim(0, max(e['parameter_linf'] for e in rows)*1e6*1.1)
    axes[2].set_ylim(0, max(1, max(e['prediction_disagreements'] for e in rows))*1.12)
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .01))
    fig.suptitle('All paired rounds retained; 3 seeds are descriptive repetitions', y=.98)
    fig.tight_layout(rect=[0, .12, 1, .92])
    for suffix in ['png', 'svg']:
        fig.savefig(out/('training_differences.'+suffix), dpi=180)
    plt.close(fig)
    manifest = json.loads((out/'analysis_manifest.json').read_text(encoding='utf-8'))
    manifest.update(figure_sha256={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.png', '.svg']},
        plot_repair=dict(source='repair_source.py', source_sha256=sha(out/'repair_source.py'),
            reason='Replace crowded scientific offset with explicit axis units; integer disagreement ticks.',
            data_changed=False, previous='review_history/before_axis_unit_repair'),
        visual_review='pending actual final PNG inspection')
    dump_json(out/'analysis_manifest.json', manifest)
    print('Repaired axis labels only; training inputs and analysis unchanged')


if __name__ == '__main__':
    main()
