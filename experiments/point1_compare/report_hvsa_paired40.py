# -*- coding: utf-8 -*-
"""Measured cumulative and amortized curves for paired contributor counts."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CODEX.md').is_file())
sys.path.insert(0,str(ROOT))
from experiments.point1_compare.artifacts import digest,dump_json
from experiments.point1_compare.run_hvsa_paired40 import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run');p.add_argument('--output',required=True);p.add_argument('--document',required=True)
    args=p.parse_args();source=Path(args.run).resolve()
    problems=validate(source)
    if problems:raise ValueError(repr(problems))
    m=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    c=json.loads((source/'correctness.json').read_text(encoding='utf-8'))
    shape=m['paired_shape'];rounds=shape['rounds']
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    init={i['specification']['setting']:i for i in c['initialization']}
    curves={};all_records={}
    for label in ['fixed_count','changing_count']:
        obs=sorted([o for o in c['observations'] if o['setting']==label],key=lambda o:o['round'])
        work=np.array([sum(o['role_ms'].values())/1000 for o in obs])
        cumulative=np.cumsum(work)+init[label]['init_ms']/1000
        traffic=np.cumsum([o['total_sent_bytes'] for o in obs])+init[label]['init_sent_bytes']
        curves[label]={'round_work_s':work.tolist(),'cumulative_work_s':cumulative.tolist(),
            'amortized_work_s':(cumulative/np.arange(1,rounds+1)).tolist(),
            'cumulative_bytes':traffic.tolist(),'init_ms':init[label]['init_ms'],
            'init_bytes':init[label]['init_sent_bytes']}
        all_records[label]=obs
    dump_json(output/'analysis.json',{'kind':'analysis_of_existing_evidence','shape':shape,'curves':curves,
        'source_manifest':{'path':str(source/'manifest.json'),'sha256':digest(source/'manifest.json')},
        'definition':'Initialization once plus real round work; same vector pool and uploaded count, alternating trajectory order.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    names={'fixed_count':'Fixed members, 40 contributors','changing_count':'Changing members, 40 contributors'}
    for label in names:names[label]=names[label].replace('40',str(shape['uploaded']))
    chart_specs=[('cumulative_work','cumulative_work_s','Cumulative protocol computation','Total role work including setup (s)'),
                 ('amortized_work','amortized_work_s','Amortized protocol computation','Cumulative role work / completed rounds (s)')]
    for stem,metric,title,ylabel in chart_specs:
        fig,ax=plt.subplots(figsize=(9,5.4))
        x=np.arange(1,rounds+1);marks=sorted(set(i for i in [0,9,rounds-1] if i<rounds))
        for label,color,style,marker in [('fixed_count','#2474A6','-','o'),('changing_count','#DE8A25','--','s')]:
            ax.plot(x,curves[label][metric],label=names[label],color=color,linestyle=style,
                    marker=marker,markevery=marks,markersize=4,linewidth=1.7)
        ax.set_xlabel('Completed rounds');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        ax.set_xlim(.5,rounds+.5)
        ax.set_ylim(0,max(max(values[metric]) for values in curves.values())*1.06)
        ax.set_xticks(sorted(set([1,rounds]+[n for n in [10,20,30,40,50] if n<=rounds])))
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=2,frameon=False,fontsize=9)
        fig.suptitle(title,y=.98,fontsize=13)
        fig.text(.5,.075,f"Registered N={shape['n']}, M={shape['m']}, d={shape['d']}; same inputs and contributor count.",ha='center',fontsize=9)
        fig.text(.5,.035,'Actual trajectories; setup counted once, no warmup exclusion. Local work is not deployed latency.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.13,right=.985,bottom=.20,top=.82)
        fig.savefig(output/(stem+'.png'),dpi=200);fig.savefig(output/(stem+'.svg'));plt.close(fig)
    lines=['# HVSA同人数多轮对照结果','',
        f"日期：2026-10-23。注册{shape['n']}人、{shape['m']}个边缘、维度{shape['d']}，固定与变化两条轨迹每轮均上传{shape['uploaded']}人，各{rounds}轮。共{2*rounds}/{2*rounds}次原始整数和逐维正确且全部边缘完整验证通过；全部实际模型接收者与对应上传集合一致。",'',
        '本轮控制了输入向量池、上传人数和协议代码，并每轮轮换两条轨迹的执行先后。固定轨迹始终使用变化轨迹第0轮的名单；变化轨迹每轮重新确定名单。各自参考和按实际上传者求和，因此两条轨迹不要求聚合结果相同。更新为独立合成整数输入，这是多轮协议回放，不是训练收敛实验。','',
        '## 实际累计与摊销成本','',
        '下表为一次初始化加上真实已执行前缀，没有将单轮耗时乘以轮数。无预热剔除；摊销值为累计工作/已完成轮数，不是中位数。系统工作是全部角色互斥工作之和，不等于多机部署时延。','',
        '| 轨迹 | 前缀轮数 | 累计工作/s | 摊销工作/s每轮 | 累计协议封装/MiB |','|---|---:|---:|---:|---:|']
    cutoffs=[i for i in [1,10,50] if i<=rounds]
    if rounds not in cutoffs:cutoffs.append(rounds)
    for label,name in [('fixed_count','固定成员'),('changing_count','变化成员')]:
        for n in cutoffs:
            v=curves[label]
            lines.append(f"| {name}，{shape['uploaded']}人上传 | {n} | {v['cumulative_work_s'][n-1]:.3f} | {v['amortized_work_s'][n-1]:.3f} | {v['cumulative_bytes'][n-1]/2**20:.4f} |")
    for stem,metric,title,ylabel in chart_specs:lines += ['',f'![{title}]({stem}.png)']
    lines += ['','## 每轮实际角色工作','',
        '各列分别取实际轮次中位数。角色中位数相加不必等于系统工作中位数；全体客户端列没有除以注册人数或上传人数。','',
        '| 轨迹 | 全体客户端/ms | 全体边缘/s | TA/s | 云/ms | 回放/s，中位数[IQR] | 回放范围/s |','|---|---:|---:|---:|---:|---:|---:|']
    for label,name in [('fixed_count','固定成员'),('changing_count','变化成员')]:
        obs=all_records[label]
        med={role:float(np.median([o['role_ms'][role] for o in obs])) for role in ['user','edge','ta','cloud']}
        values=np.array([o['wall_ms']/1000 for o in obs]);q25,wall,q75=np.percentile(values,[25,50,75])
        lines.append(f"| {name} | {med['user']:.4f} | {med['edge']/1000:.4f} | {med['ta']/1000:.4f} | {med['cloud']:.4f} | {wall:.4f} [{q25:.4f},{q75:.4f}] | {min(values):.4f}—{max(values):.4f} |")
    maxima=[]
    for label,name in [('fixed_count','固定轨迹'),('changing_count','变化轨迹')]:
        worst=max(all_records[label],key=lambda o:o['wall_ms'])
        maxima.append(f"{name}第{worst['round']+1}轮为{worst['wall_ms']/1000:.3f}秒")
    lines += ['', '最大本地回放观测：'+'；'.join(maxima)+'。全部观测原样保留，未另行定位运行波动来源，不把单个慢回合归因于成员变化。']
    fixed=curves['fixed_count']['cumulative_work_s'][-1];changing=curves['changing_count']['cumulative_work_s'][-1]
    lines += ['',
        f"本次变化/固定轨迹的末尾累计工作比值为{changing/fixed:.4f}。这是本次配对轨迹的实测值，差异仍包括系统运行波动；不以一条轨迹证明成员变化必然加速或减速。通信、累计曲线及波动均保留原值，不平滑为预期形状。",'',
        '## 与此前记录的关系','',
        '旧多轮中固定50人和变化40人的两条50轮数据保留；本次新采固定40人/变化40人，不将旧跨日时间混入本对照。初始化每条轨迹一次，轮次PRG及完整验证每轮真实执行。输入、原始返回和及接收者经重算核验；两个被动观察器只保留实际返回对象，未改包、掩码或验证。','',
        '本项仅补HVSA同人数对照。GPVO/VLMRS的跨方案50轮、完整分阶段掉线及其他规模扩展仍待补，不将本项写成三方案多轮实验已完成。单机回放、密码库及缓存行为的限制沿用各自归档，不支持SOTA结论。','',
        '## 可追溯来源','',f"原始目录：`{source}`；manifest SHA-256 `{digest(source/'manifest.json')}`。",'',
        f"分析目录：`{output}`；analysis.json保存完整{rounds}轮实际曲线与初始化单列数据。PNG/SVG由实际曲线导出，PNG生成后另做目视核查。",'',
        '执行规格见[同人数多轮规格](POINT1_HVSA_PAIRED40_STAGE_20261023.md)。','']
    text='\n'.join(lines);(output/'RESULTS_SUMMARY.md').write_text(text,encoding='utf-8')
    shutil.copy2(__file__,output/'analysis_source.py')
    document=Path(args.document).resolve()
    relative=os.path.relpath(output,document.parent).replace('\\','/')
    for stem,*_ in chart_specs:text=text.replace(']('+stem+'.png)',']('+relative+'/'+stem+'.png)')
    document.write_text(text,encoding='utf-8')
    dump_json(output/'analysis_manifest.json',{'kind':'analysis_of_existing_evidence',
        'created_at':datetime.datetime.now().astimezone().isoformat(),'visual_review':'pending',
        'matplotlib':matplotlib.__version__,'numpy':np.__version__,
        'source_manifest':{'path':str(source/'manifest.json'),'sha256':digest(source/'manifest.json')},
        'document':{'path':str(document),'sha256':digest(document)},
        'output_sha256':{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    print('REPORT='+str(document),flush=True)


if __name__=='__main__':main()
