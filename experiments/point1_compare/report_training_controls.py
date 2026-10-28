"""Source-bound figures and descriptive paired E7 control results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .artifacts import dump_json
from .training_control_contract import ROOT, SOURCE, SOURCE_SHA, sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--document', required=True)
    args = parser.parse_args(argv)
    run, out, doc = Path(args.directory), Path(args.output), Path(args.document)
    m = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    v = json.loads((run/'validation.json').read_text(encoding='utf-8'))
    if m['design']['smoke'] or not v['valid'] or v['manifest_sha256'] != sha(run/'manifest.json'):
        raise ValueError('Report requires validated formal data')
    for rel, digest in m['output_sha256'].items():
        if sha(run/rel) != digest:
            raise ValueError('Validated output changed')
    if out.exists() or doc.exists():
        raise FileExistsError('Report cannot overwrite an existing delivery')
    out.mkdir(parents=True)
    obs = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    rows = []
    for seed in m['design']['seeds']:
        last = [e for e in obs if e['seed'] == seed][-1]
        rows.append({k: last[k] for k in ['seed', 'encoded_accuracy', 'float_accuracy',
            'float_minus_encoded_pp', 'parameter_linf', 'parameter_l2', 'prediction_disagreements']})
    summary = dict(run=str(run), manifest_sha256=sha(run/'manifest.json'),
        validation_sha256=sha(run/'validation.json'), historical_manifest_sha256=SOURCE_SHA,
        pairs=len(obs), trained_branches=2*len(obs), new_secure_protocol_calls=0,
        rows=rows, encoded_mean_pct=float(np.mean([r['encoded_accuracy'] for r in rows])*100),
        float_mean_pct=float(np.mean([r['float_accuracy'] for r in rows])*100),
        encoded_sample_sd_pp=float(np.std([r['encoded_accuracy'] for r in rows], ddof=1)*100),
        float_sample_sd_pp=float(np.std([r['float_accuracy'] for r in rows], ddof=1)*100),
        max_parameter_linf=max(e['parameter_linf'] for e in obs),
        max_prediction_disagreements=max(e['prediction_disagreements'] for e in obs),
        archived_matches=sum(e['historical_encoded_match'] and e['historical_base_match'] for e in obs),
        environment_observations=[e[k] for e in obs for k in ['environment_before', 'environment_after']])
    dump_json(out/'analysis.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), sharey=True)
    for ax, seed in zip(axes, m['design']['seeds']):
        part = [e for e in obs if e['seed'] == seed]
        x = np.arange(1, 21)
        ax.plot(x, [e['encoded_accuracy']*100 for e in part], color='#165a96', lw=2,
                marker='o', ms=4, label='Encoded plaintext = archived HVSA')
        ax.plot(x, [e['float_accuracy']*100 for e in part], color='#c56e14', lw=1.6,
                ls='--', marker='x', ms=5, label='Independent float FedAvg')
        ax.set(title=f'Seed {seed}', xlabel='Communication round', xticks=[1, 5, 10, 15, 20])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('MNIST test accuracy (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .01))
    fig.suptitle('Independent training trajectories; shared initialization, partition and batch schedule', y=.98)
    fig.tight_layout(rect=[0, .13, 1, .92])
    for suffix in ['png', 'svg']:
        fig.savefig(out/('training_accuracy.'+suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7))
    colors = ['#165a96', '#bd5e1e', '#477d43']
    for seed, color in zip(m['design']['seeds'], colors):
        part = [e for e in obs if e['seed'] == seed]
        x = np.arange(1, 21)
        for ax, key in zip(axes, ['float_minus_encoded_pp', 'parameter_linf', 'prediction_disagreements']):
            ax.plot(x, [e[key] for e in part], marker='o', ms=3.2, lw=1.5, color=color, label=f'Seed {seed}')
    titles = ['Float minus encoded accuracy', 'Maximum coordinate difference', 'Prediction disagreements']
    ys = ['Accuracy difference (percentage points)', 'Model parameter absolute difference', 'Count out of 10,000 test images']
    for ax, title, ylabel in zip(axes, titles, ys):
        ax.set(title=title, xlabel='Communication round', ylabel=ylabel, xticks=[1, 5, 10, 15, 20])
        ax.grid(alpha=.2)
    axes[0].axhline(0, color='gray', lw=.8)
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[2].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .01))
    fig.suptitle('All paired rounds retained; 3 seeds are descriptive repetitions', y=.98)
    fig.tight_layout(rect=[0, .12, 1, .92])
    for suffix in ['png', 'svg']:
        fig.savefig(out/('training_differences.'+suffix), dpi=180)
    plt.close(fig)
    sources = {}
    for rel in m['source_sha256']:
        target = out/'sources'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run/'sources'/rel, target)
        sources[rel] = sha(target)
    dump_json(out/'analysis_manifest.json', dict(run_manifest_sha256=sha(run/'manifest.json'),
        analysis_sha256=sha(out/'analysis.json'), source_sha256=sources,
        figure_sha256={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.png', '.svg']},
        visual_review='pending actual PNG inspection'))
    table = '\n'.join(f'|{r["seed"]}|{r["encoded_accuracy"]*100:.2f}|{r["float_accuracy"]*100:.2f}|'
        f'{r["float_minus_encoded_pp"]:+.2f}|{r["parameter_linf"]:.8g}|{r["prediction_disagreements"]}|'
        for r in rows)
    figure_rel = '../'+out.relative_to(ROOT).as_posix()
    text = ('# 研究点一：独立明文FedAvg训练对照\n\n'
        '日期：2026-10-26。新增两个实际闭环：同编码明文与独立浮点FedAvg，种子42/43/44各20轮，'
        '共60组配对、120个分支训练回合。同编码分支60/60逐轮重现旧HVSA的基模型、20个量化本地更新、'
        '整数和及测试正确数；旧来源的60次完整安全聚合和300次边缘验证不重复计作新执行。\n\n'
        '## 设置与对照含义\n\n'
        'MNIST训练60000/测试10000，Linear(784,10)、7850参数、20客户端IID等量划分、全员参与，'
        '本地SGD学习率0.01、1epoch、batch64，CPU单线程。两分支共享初始参数、划分和batch随机种子，'
        '后续各自从自己的上轮模型训练。浮点分支实际重新训练本地模型，不复用量化分支更新。\n\n'
        '量化为float32乘10^6后截断int64，精确整数和除10^6再除20，再转float32写回。'
        '浮点对照将自己的float32本地模型转float64求均值，再转float32写回。\n\n'
        '## 末轮结果\n\n'
        '|种子|同编码明文/HVSA准确率%|独立浮点准确率%|浮点减同编码/百分点|参数最大坐标差|预测分歧/10000|\n'
        '|---|---:|---:|---:|---:|---:|\n'+table+'\n\n'
        f'末轮同编码均值{summary["encoded_mean_pct"]:.3f}%，样本标准差{summary["encoded_sample_sd_pp"]:.3f}个百分点；'
        f'浮点均值{summary["float_mean_pct"]:.3f}%，样本标准差{summary["float_sample_sd_pp"]:.3f}个百分点。'
        f'全部配对轮的最大参数坐标差为{summary["max_parameter_linf"]:.9g}，最大预测分歧为'
        f'{summary["max_prediction_disagreements"]}/10000。三种子的全部轨迹保留。\n\n'
        f'![逐种子训练准确率]({figure_rel}/training_accuracy.png)\n\n'
        f'![逐轮配对差值]({figure_rel}/training_differences.png)\n\n'
        '## 核验与解释\n\n'
        '独立核验重新计算全部60组聚合、两分支连续性及120组测试预测；按预声明规则每轮每分支'
        '选客户端(seed+round)%20，另外重新执行120次本地SGD并逐参数核对。它是本地SGD抽样复算，'
        '不表示离线重跑了全部2400个客户端训练。驱动执行了全部客户端训练，实际本地模型及随机数均归档。\n\n'
        '测试准确率的接近只支持本设置下未观察到明显精度损失；有限种子不能证明统计等价或精度领先。'
        'HVSA与同编码明文的逐轮一致性和独立浮点分支的微小量化差异分别解释。'
        '未新增GPVO/VLMRS训练曲线，未更改已录用论文/PPT，不提供训练速度排名。\n\n'
        '## 来源\n\n'
        f'- 新运行：`{run.relative_to(ROOT).as_posix()}`，manifest SHA-256 `{sha(run/"manifest.json")}`。\n'
        f'- 旧安全聚合来源：`{SOURCE}`，manifest SHA-256 `{SOURCE_SHA}`，归档字节保持不变。\n'
        '- [运行前规格](POINT1_TRAINING_CONTROL_STAGE_20261026.md)。\n'
        f'- [分析数据]({figure_rel}/analysis.json)、[分析与图摘要]({figure_rel}/analysis_manifest.json)。\n\n'
        '图的实际视觉核查另存凭据，生成报告本身不表示已经查看。\n')
    doc.write_text(text, encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'environment_observations'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
