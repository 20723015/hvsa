# -*- coding: utf-8 -*-
"""Actual cumulative/steady-round E3 evidence; smoke and incomplete runs excluded."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import shutil
import numpy as np
from .artifacts import dump_json
from .large_multirun_contract import SCHEMES,GPVO,VLMRS,LABELS,summaries
from .multirun_environment import summarize as condition_summary
from .round_shards import load,file_sha
from .validate_large_multirun import validate

NAMES={'hvsa':'HVSA',GPVO:'GPVO declared',VLMRS:'VLMRS declared'}
ZH={'hvsa':'HVSA',GPVO:'GPVO明确实例化',VLMRS:'VLMRS明确实例化'}
COLORS={'hvsa':'#2474a6',GPVO:'#de8a25',VLMRS:'#3b9866'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run');parser.add_argument('--output',required=True)
    parser.add_argument('--document',required=True)
    args=parser.parse_args(argv)
    root=Path(args.run).resolve();out=Path(args.output).resolve();doc=Path(args.document).resolve()
    errors=validate(root)
    if errors:raise ValueError(errors)
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if m['shape']['smoke']:raise ValueError('Smoke cannot be reported as formal E3')
    if doc.exists():raise FileExistsError(doc)
    out.mkdir(parents=True,exist_ok=False)
    journal,entries=load(root);obs=[e for e in entries if e['kind']=='round'];s=m['shape']
    curves=[]
    for label in LABELS:
        for scheme in SCHEMES:
            init=next(e for e in entries if e['kind']=='init' and e['label']==label and e['scheme']==scheme)
            rows=[e for e in obs if e['label']==label and e['scheme']==scheme]
            work=np.array([sum(e['role_ms'].values())/1000 for e in rows])
            traffic=np.array([e['total_sent_bytes'] for e in rows])
            setup=sum(init['role_ms'].values())/1000
            cumulative=setup+np.cumsum(work)
            curves.append(dict(label=label,scheme=scheme,rounds=[e['round']+1 for e in rows],
                setup_work_s=setup,setup_bytes=init['total_sent_bytes'],
                role_setup_ms=init['role_ms'],round_work_s=work.tolist(),
                role_round_ms=[e['role_ms'] for e in rows],round_wire_bytes=traffic.tolist(),
                cumulative_work_s=cumulative.tolist(),amortized_work_s=(cumulative/np.arange(1,len(rows)+1)).tolist(),
                cumulative_wire_bytes=(init['total_sent_bytes']+np.cumsum(traffic)).tolist(),
                valid=[e['valid'] for e in rows],completed=[e['completed'] for e in rows],
                first_round_s=float(work[0]),later_round_work_s=dict(zip(['q25','median','q75'],np.percentile(work[1:],[25,50,75]).tolist()))))
    analysis=dict(kind='analysis_of_completed_large_multirun',shape=s,run=str(root),
        manifest_sha256=file_sha(root/'manifest.json'),journal_sha256=file_sha(root/'journal.json'),
        observations=len(obs),valid=sum(e['valid'] for e in obs),unavailable=sum(not e['completed'] for e in obs),
        initialization_count=6,curves=curves,prefixes=summaries(entries,s),
        conditions=condition_summary(entries),
        interpretation='Actual paired traces; one key epoch each, initialization included once, no extrapolation or independent-key repetitions.')
    dump_json(out/'analysis.json',analysis)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    for stem,metric,ylabel,title in [
        ('cumulative_work','cumulative_work_s','All-role work including setup (min)','Measured cumulative protocol work'),
        ('amortized_work','amortized_work_s','Cumulative work / actual rounds (s)','Measured amortized protocol work')]:
        fig,ax=plt.subplots(figsize=(10.5,6))
        for c in curves:
            y=np.array(c[metric])/(60 if metric=='cumulative_work_s' else 1)
            ax.plot(c['rounds'],y,'-' if c['label']=='fixed' else '--',color=COLORS[c['scheme']],
                linewidth=1.6,marker='o' if c['label']=='fixed' else 's',markevery=[0,9,49],markersize=3,
                label=NAMES[c['scheme']]+' / '+c['label'])
        ax.set_title(title);ax.set_xlabel('Actual completed calls in the trajectory');ax.set_ylabel(ylabel)
        ax.set_xlim(.5,50.5);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
        ax.legend(ncol=2,frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.15))
        fig.text(.5,.045,'N=50, d=7850; hierarchical M=5; 40 contributors in both trajectories, paired inputs and rotating order.',ha='center',fontsize=8.5)
        fig.text(.5,.02,'Setup once per trajectory; all returned calls retained. Declared instances, not deployed latency or equal-security ranking.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.105,right=.98,bottom=.26,top=.9)
        fig.savefig(out/(stem+'.png'),dpi=200);fig.savefig(out/(stem+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
    max_work=max(max(c['round_work_s']) for c in curves)
    max_wire=max(max(c['round_wire_bytes'])/2**20 for c in curves)
    for row,scheme in enumerate(SCHEMES):
        for label,style in [('fixed','-'),('changing','--')]:
            c=next(v for v in curves if v['scheme']==scheme and v['label']==label)
            for column,y in [(0,c['round_work_s']),(1,np.array(c['round_wire_bytes'])/2**20)]:
                axes[row,column].plot(c['rounds'],y,style,color=COLORS[scheme],linewidth=1.4,
                    marker='o' if label=='fixed' else 's',markersize=2,
                    markerfacecolor=COLORS[scheme] if label=='fixed' else 'white',label=label)
                failed=[i for i,v in enumerate(c['valid']) if not v]
                if failed:axes[row,column].scatter(np.array(c['rounds'])[failed],np.array(y)[failed],marker='x',color='red',zorder=4)
        for column in range(2):
            ax=axes[row,column];ax.set_title(NAMES[scheme]+(' / round work' if column==0 else ' / wire envelopes'))
            ax.set_ylabel('All-role work (s)' if column==0 else 'Protocol envelopes (MiB)')
            ax.set_xlim(.5,50.5);ax.set_ylim(0,1.1*(max_work if column==0 else max_wire))
            ax.grid(alpha=.2);ax.legend(frameon=False,ncol=2,fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Actual round index + 1')
    fig.text(.5,.035,'Every observed round marked; shared y limits within each column. First-round effects and outliers retained; setup excluded.',ha='center',fontsize=8.5)
    fig.text(.5,.015,'50 registered, 40 uploaded, 7850 dimensions; local sequential replay. Red x (if any) marks incomplete/invalid returned calls.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.085,right=.98,bottom=.1,top=.95,hspace=.36,wspace=.23)
    fig.savefig(out/'roundwise_work_wire.png',dpi=200);fig.savefig(out/'roundwise_work_wire.svg');plt.close(fig)
    conditions=analysis['conditions']
    text=['# 三方案N50/d7850同人数多轮结果（2026-10-24启动）','',
        f'固定40人/变化40人各真实执行50轮，三方案共{len(obs)}次调用，{analysis["valid"]}次原始整数和正确且全部指定验证者通过，{analysis["unavailable"]}次未完成。六个状态各初始化一次；此前HVSA单方案100轮和N20分段补充独立保留，本报告不混合其成本。','',
        '## 1. 设置与计量','',
        '注册50人，分层方案5个边缘，7850维；VLMRS为平面全邻居图，恢复阈值34。每轮固定/变化轨迹都上传40人，固定名单取变化轨迹第0轮。第i轮使用种子100042+i的共同整数向量池，名单使用10042+i的排列；两轨迹及三方案共用输入。奇数轮反转两轨迹先后，轮换三方案次序。无有利重采样，上传前缺席发生于名单/必要分享完成之后。','',
        '每个方案每条轨迹初始化一次，共6个连续状态。HVSA/VLMRS记录长期公开密钥指纹并逐轮核对，GPVO记录公参指纹和每轮新会话公开指纹。公开指纹检查与冻结源码、实际执行账本共同使用，不作为独立密码学证明。两个GPVO状态独立生成公参，初始化耗时差异单列。','',
        '协议计算采用互斥角色计时，通信为逐接收者实际封装；输入生成、明文参考、环境采集、归档与核验在协议计时之外。累计值包含一次初始化及实际前缀，首轮不排除。系统角色工作与本机串行回放均不解释成多机部署时延。','',
        '## 2. 初始化、首轮与后续轮','',
        '后续轮为第2—50轮的描述性中位数/IQR，同一状态内的序列不作为49套独立密钥或独立机器重复。所有异常高值保留在逐轮图中。','',
        '| 方案 | 轨迹 | 初始化/s | 首轮工作/s | 第2—50轮中位数[IQR]/s | 初始化封装/MiB |',
        '|---|---|---:|---:|---:|---:|']
    for c in curves:
        later=c['later_round_work_s']
        text.append(f'| {ZH[c["scheme"]]} | '+('固定40人' if c['label']=='fixed' else '变化40人')+f' | {c["setup_work_s"]:.3f} | {c["first_round_s"]:.3f} | {later["median"]:.3f} [{later["q25"]:.3f},{later["q75"]:.3f}] | {c["setup_bytes"]/2**20:.5f} |')
    text+=['','## 3. 实际累计与摊销','','| 方案 | 轨迹 | 真实前缀轮数 | 累计工作/s | 摊销/s每轮 | 累计封装/MiB |','|---|---|---:|---:|---:|---:|']
    for p in analysis['prefixes']:
        text.append(f'| {ZH[p["scheme"]]} | '+('固定40人' if p['label']=='fixed' else '变化40人')+f' | {p["rounds"]} | {p["cumulative_work_ms"]/1000:.3f} | {p["amortized_work_ms"]/1000:.3f} | {p["cumulative_bytes"]/2**20:.4f} |')
    text+=['',
        '![实际累计工作](../figures/point1_compare/20261024_large_multirun/cumulative_work.png)','',
        '![实际摊销工作](../figures/point1_compare/20261024_large_multirun/amortized_work.png)','',
        '![全部逐轮工作与通信](../figures/point1_compare/20261024_large_multirun/roundwise_work_wire.png)','',
        '## 4. 环境观察、解释与边界','',
        f'共记录{conditions["snapshots"]}份初始化/聚合前后环境快照；全部观察均接通电源：{conditions["all_observed_power_plugged"]}；观察到供电状态转换{conditions["observed_power_transitions"]}次；其他协议worker出现于{conditions["snapshots_with_other_protocol_workers"]}份快照。报告频率范围为{conditions["reported_frequency_min_mhz"]}—{conditions["reported_frequency_max_mhz"]}MHz。完整快照保留在各分片。','',
        '该记录支持同进程、每轨迹单一初始化/密钥状态的连续执行，不能证明快照之间无短时供电变化，也不等于锁定频率、温度、背景负载或热状态。所有时段数据保留；不能把固定/变化的耗时差全部归因于成员变化、密钥复用或协议结构。它只有一组配对轨迹，不提供独立机器重复的因果效应估计。','',
        'GPVO仍为本项目模N_RSA²乘法LHF、3072位模数、素数e>N_RSA²及逐轮新会话实例化；生成元最大阶未认证，整数群接口与原文P-256表述的差异保留。VLMRS仍采用公开根指数7、长期密钥复用和全邻居恢复，既有条件完整性/历史恢复现象不被诚实正确性掩盖。后端、缓存容量和信任设施差异继续限定成本解释，不宣称原文安全等价或SOTA。','',
        'N50/d7850三方案各两条50轮轨迹完成不等于N100/d7850掉线矩阵或E4统一环境单状态10条已经完成；这些仍是下一阶段任务。未返回的在途工作若发生中断，不会捏造为完整协议成本。','',
        '## 证据入口','',
        '- [运行前规格](POINT1_LARGE_MULTIRUN_STAGE_20261024.md)。',
        f'- [原始manifest](../results/point1_compare/{root.name}/manifest.json)、[分片journal](../results/point1_compare/{root.name}/journal.json)、[逐次结果](../results/point1_compare/{root.name}/correctness.json)。',
        '- [分析数据](../figures/point1_compare/20261024_large_multirun/analysis.json)、[分析来源清单](../figures/point1_compare/20261024_large_multirun/analysis_manifest.json)。',
        '- 最终三张PNG需实际目视核查后保存独立凭据；当前报告生成本身不表示该步骤已完成。','']
    doc.write_text('\n'.join(text),encoding='utf-8')
    for name in ['report_large_multirun.py','large_multirun_contract.py','round_shards.py','multirun_environment.py','validate_large_multirun.py']:
        shutil.copyfile(Path(__file__).with_name(name),out/name)
    files=['analysis.json',*[x+'.'+ext for x in ['cumulative_work','amortized_work','roundwise_work_wire'] for ext in ['png','svg']],
        'report_large_multirun.py','large_multirun_contract.py','round_shards.py','multirun_environment.py','validate_large_multirun.py']
    dump_json(out/'analysis_manifest.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run=str(root),manifest_sha256=file_sha(root/'manifest.json'),outputs={name:file_sha(out/name) for name in files},visual_review='pending_actual_png_views'))
    print(json.dumps(dict(report=str(doc),observations=len(obs),valid=analysis['valid']),ensure_ascii=False))


if __name__=='__main__':main()
