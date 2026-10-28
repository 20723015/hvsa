"""Original full N/d/M grid: source-verified 3+10 costs, no interpolated cells."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .artifacts import dump_json
from .full_scale_contract import ROOT,POINTS,SCANS,GPVO,VLMRS,schemes,reused_inventory
from .round_shards import file_sha
from .validate_full_scale import validate_batch

NAMES={'hvsa':'HVSA',GPVO:'GPVO (declared)',VLMRS:'VLMRS (declared)'}


def stats(values):
    q=np.percentile(values,[25,50,75])
    return dict(count=len(values),median=float(q[1]),q25=float(q[0]),q75=float(q[2]),
                minimum=float(min(values)),maximum=float(max(values)))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('batch');p.add_argument('--output',required=True);p.add_argument('--document',required=True)
    args=p.parse_args(argv);batch=Path(args.batch).resolve();out=Path(args.output).resolve();doc=Path(args.document).resolve()
    errors=validate_batch(batch)
    if errors:raise ValueError('Full original grid is not complete: '+repr(errors))
    if out.exists() or doc.exists():raise FileExistsError('Do not overwrite report/figures')
    b=json.loads(batch.read_text(encoding='utf-8'));records=[];setups=[];sources={}
    jobs=b['reused']+b['jobs']
    for point in POINTS:
        job=next(j for j in jobs if tuple(j['point'])==point);root=Path(job['run_dir'])
        m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        c=json.loads((root/'correctness.json').read_text(encoding='utf-8'))
        sources[str(root)]=dict(manifest_sha256=file_sha(root/'manifest.json'),
            output_sha256=m['output_sha256'],source_sha256=m['source_sha256'],lib_versions=m['lib_versions'],
            observed_conditions=m.get('observed_conditions'),reused=job in b['reused'])
        for scheme in schemes(point):
            obs=[e for e in c['observations'] if e['scheme']==scheme]
            measured=[e for e in obs if not e['warmup']]
            if len(obs)!=13 or len(measured)!=10 or not all(e['valid'] for e in obs):
                raise ValueError('Wrong formal coverage')
            metrics=dict(client_ms=[e['role_ms']['user']/point[0] for e in measured],
                system_work_s=[sum(e['role_ms'].values())/1000 for e in measured],
                traffic_mib=[e.get('total_sent_bytes',0)/2**20 for e in measured],
                wall_s=[e['wall_ms']/1000 for e in measured])
            for role in ['user','edge','ta','cloud']:metrics[role+'_work_s']=[e['role_ms'][role]/1000 for e in measured]
            records.append(dict(point=list(point),scheme=scheme,M=None if scheme==VLMRS else point[1],
                source_run=str(root),reused=job in b['reused'],warmup_calls=3,measured_calls=10,
                metrics={k:stats(v) for k,v in metrics.items()},values=metrics,
                input_sha256=[e['input_sha256'] for e in measured]))
            init=next(e for e in c['initialization'] if e['scheme']==scheme)
            setups.append(dict(point=list(point),scheme=scheme,
                role_ms=init['role_ms'],total_work_s=sum(init['role_ms'].values())/1000,
                sent_bytes=init.get('total_sent_bytes',init.get('sent_bytes')),source_run=str(root)))
    if len(records)!=28:raise ValueError('Missing cells in report')
    analysis=dict(artifact_kind='analysis_of_full_E2_evidence',batch=str(batch),batch_sha256=file_sha(batch),
        records=records,initialization=setups,sources=sources,unique_points=10,point_scheme_cells=28,
        actual_calls=364,warmup_calls=84,measured_calls=280,reused_calls=117,new_calls=247,
        VLMRS_M='not applicable; no duplicate M-dependent measurements',
        interpretation='Local replay; source, time, trust, backend and key-epoch limits retained.')
    out.mkdir(parents=True);dump_json(out/'analysis.json',analysis)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'hvsa':'#246a9a',GPVO:'#c47c22',VLMRS:'#41825a'}
    chart_metrics=[('client_ms','All-client work / selected N (ms)',True),
        ('system_work_s','Sum of exclusive role work (s)',True),
        ('traffic_mib','Serialized sent envelopes (MiB)',False)]
    for scan,points in SCANS.items():
        selected=['hvsa',GPVO] if scan=='M' else ['hvsa',GPVO,VLMRS]
        fig,axes=plt.subplots(1,3,figsize=(15,5.1));positions=np.arange(len(points))
        for si,scheme in enumerate(selected):
            rows=[next(r for r in records if tuple(r['point'])==point and r['scheme']==scheme) for point in points]
            for ax,(key,ylabel,log) in zip(axes,chart_metrics):
                data=[r['metrics'][key] for r in rows];med=np.array([v['median'] for v in data])
                ax.errorbar(positions,med,yerr=np.array([med-np.array([v['q25'] for v in data]),
                    np.array([v['q75'] for v in data])-med]),label=NAMES[scheme],
                    color=colors[scheme],marker=['o','s','^'][si],lw=1.5,capsize=3)
                for x,row in zip(positions,rows):
                    jitter=np.linspace(-.065,.065,10)+(si-(len(selected)-1)/2)*.035
                    ax.scatter(x+jitter,row['values'][key],color=colors[scheme],s=8,alpha=.25,zorder=2)
                if log:ax.set_yscale('log')
                else:ax.set_ylim(bottom=0)
                ax.set_ylabel(ylabel);ax.grid(alpha=.2,axis='y')
        idx={'N':0,'M':1,'d':2}[scan];ticks=[str(point[idx]) for point in points]
        for ax in axes:
            ax.set_xticks(positions,ticks);ax.set_xlabel({'N':'Registered clients N','d':'Vector dimension d','M':'Edge servers M'}[scan])
        handles,legend=axes[0].get_legend_handles_labels()
        fig.legend(handles,legend,loc='lower center',ncol=len(selected),frameon=False,bbox_to_anchor=(.5,.045))
        fixed={'N':'M=5, d=7850','d':'N=50, M=5','M':'N=100, d=7850; VLMRS: M not applicable'}[scan]
        fig.suptitle('Full '+scan+' scan: '+fixed,y=.98)
        fig.text(.5,.01,'10 measured rounds/cell; 3 warmups excluded. All points retained. Capture periods and backends differ.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.17,1,.92])
        for ext in ['png','svg']:fig.savefig(out/(scan.lower()+'_scan.'+ext),dpi=180)
        plt.close(fig)
    shutil.copyfile(__file__,out/'report_source.py')
    dump_json(out/'analysis_manifest.json',dict(batch_sha256=file_sha(batch),analysis_sha256=file_sha(out/'analysis.json'),
        report_source_sha256=file_sha(out/'report_source.py'),
        figure_sha256={p.name:file_sha(p) for p in out.iterdir() if p.suffix in ['.png','.svg']},visual_review='pending actual PNG inspection'))
    lines=['# 跨方案完整规模扫描结果','',
        '完整覆盖原E2的N/d/M网格：10个几何工作点，HVSA/GPVO各10点、VLMRS为8个平面N/d点。'
        '共28个方案/点、364次真实调用（84预热、280正式），每格3预热+10正式，全部正确且完整验证。'
        '复用117次原合格记录，新增247次；旧N20/d1000与多轮/掉线记录不混入本分母。','',
        '## 设置与来源','',
        '同点三方案共享真实整数输入和全员在线名单；初始化单列、方案执行次序轮换。'
        '密码源码与20261019实例化决策冻结；GPVO逐轮新会话，VLMRS保持全邻居/公开根指数7/长期密钥复用。'
        '旧三点和新七点的采集时段不同，只读环境观察不等于锁定硬件，不能将跨点耗时差全部归因于规模。'
        'VLMRS没有边缘数参数，M图仅展示HVSA和GPVO。','',
        '|N|M（分层）|d|方案|正式次数|客户端ms/人 中位数[IQR]|全角色工作s 中位数[IQR]|通信MiB 中位数[IQR]|来源批次|',
        '|---:|---:|---:|---|---:|---:|---:|---:|---|']
    for r in records:
        n,m,d=r['point'];parts=[]
        for key in ['client_ms','system_work_s','traffic_mib']:
            v=r['metrics'][key];parts.append(f'{v["median"]:.4f} [{v["q25"]:.4f},{v["q75"]:.4f}]')
        lines.append(f'|{n}|{m if r["scheme"]!=VLMRS else "不适用"}|{d}|{NAMES[r["scheme"]]}|10|'+ '|'.join(parts)+'|'+Path(r['source_run']).name+'|')
    rel='../'+out.relative_to(ROOT).as_posix()
    for scan,title in [('n','人数扫描'),('d','维度扫描'),('m','边缘数扫描')]:lines+=['',f'![{title}]({rel}/{scan}_scan.png)']
    lines+=['','## 初始化与角色分解','',
        '初始化工作不并入每轮中位数；随机公参生成时间不可归因于规模。逐角色正式观测与全部逐轮值在analysis.json中保留。',
        '|N|M（分层）|d|方案|初始化全角色工作s|初始化发送字节|','|---:|---:|---:|---|---:|---:|']
    for r in setups:
        n,m,d=r['point'];lines.append(f'|{n}|{m if r["scheme"]!=VLMRS else "不适用"}|{d}|{NAMES[r["scheme"]]}|{r["total_work_s"]:.4f}|{r["sent_bytes"]}|')
    lines+=['','## 结论范围','',
        '这些结果覆盖既定工作点的完整实测，不构成原文安全等价、独立平台/密钥重复或SOTA认证。'
        '客户端是全部客户端工作除以所选N；全角色工作为互斥计时之和，通信为实际逐接收者封装字节，均非部署网络时延。'
        'IQR只描述本格10次观测，不作为置信区间。所有有利/不利成本方向保留。',
        f'原始批次：`{batch}`，SHA-256 `{file_sha(batch)}`。',
        f'[分析与全部值]({rel}/analysis.json)、[图与源码摘要]({rel}/analysis_manifest.json)。',
        '图的视觉核查需另存实际查看凭据，本脚本运行不自动表示已查看。','']
    doc.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(report=str(doc),figures=str(out),cells=28,calls=364),indent=2))


if __name__=='__main__':main()
