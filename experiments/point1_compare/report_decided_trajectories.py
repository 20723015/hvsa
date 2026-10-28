# -*- coding: utf-8 -*-
"""Measured E3/E4 tables and scientific figures, with source reconciliation."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CODEX.md').is_file())
sys.path.insert(0,str(ROOT))
from experiments.point1_compare.artifacts import digest,dump_json
from experiments.point1_compare.validate_decided_trajectories import validate
from experiments.point1_compare.run_decided_comparison import SCHEMES,GPVO,VLMRS

NAMES={'hvsa':'HVSA',GPVO:'GPVO declared',VLMRS:'VLMRS declared'}
ZH={'hvsa':'HVSA',GPVO:'GPVO明确实例化',VLMRS:'VLMRS明确实例化'}
COLORS={'hvsa':'#2474a6',GPVO:'#de8a25',VLMRS:'#3b9866'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run');p.add_argument('--output',required=True);p.add_argument('--document',required=True)
    args=p.parse_args(argv);source=Path(args.run).resolve();output=Path(args.output).resolve()
    problems=validate(source)
    if problems:raise ValueError(repr(problems))
    m=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    c=json.loads((source/'correctness.json').read_text(encoding='utf-8'))
    if m['shape']['smoke']:raise ValueError('Smoke is excluded from formal reporting')
    output.mkdir(parents=True,exist_ok=False)
    observations=c['observations'];init={(i['setting'],i['scheme']):i for i in c['initialization']}
    curves={};cells=[]
    for scheme in SCHEMES:
        curves[scheme]={}
        for membership in ['fixed','changing']:
            label='e3_'+membership;obs=[o for o in observations if o['setting']==label and o['scheme']==scheme]
            values=np.array([sum(o['role_ms'].values())/1000 for o in obs])
            setup=init[label,scheme];cumulative=np.cumsum(values)+sum(setup['role_ms'].values())/1000
            curves[scheme][membership]=dict(round_work_s=values.tolist(),cumulative_work_s=cumulative.tolist(),
                amortized_work_s=(cumulative/np.arange(1,len(obs)+1)).tolist(),
                cumulative_bytes=(np.cumsum([o['total_sent_bytes'] for o in obs])+setup['sent_bytes']).tolist(),
                initialization_role_ms=setup['role_ms'],initialization_bytes=setup['sent_bytes'])
    for sp in m['settings']:
        if sp['experiment']!='e4':continue
        for scheme in SCHEMES:
            obs=[o for o in observations if o['setting']==sp['setting'] and o['scheme']==scheme]
            work=[sum(o['role_ms'].values())/1000 for o in obs]
            traffic=[o['total_sent_bytes']/2**20 for o in obs]
            cells.append(dict(setting=sp['setting'],stage=sp['stage'],rate=sp['rate'],scheme=scheme,
                trials=len(obs),completed=sum(o['completed'] for o in obs),valid=sum(o['valid'] for o in obs),
                n_uploaded=sorted({o['n_uploaded'] for o in obs}),n_recipients=sorted({len(o['recipients']) for o in obs}),
                work_s=dict(zip(['q25','median','q75'],np.percentile(work,[25,50,75]).tolist())),
                traffic_mib=dict(zip(['q25','median','q75'],np.percentile(traffic,[25,50,75]).tolist())),
                failed_rounds=[dict(round=o['round'],reason=o['failure_reason'],completed=o['completed'],verified=o['verified']) for o in obs if not o['valid']]))
    analysis=dict(kind='analysis_of_existing_evidence',shape=m['shape'],curves=curves,dropout_cells=cells,
        source_manifest=dict(path=str(source/'manifest.json'),sha256=digest(source/'manifest.json')),
        total_observations=len(observations),valid=sum(o['valid'] for o in observations),
        note='E3 setup counted once; E4 round distributions include failed partial work and exclude separately recorded setup.')
    dump_json(output/'analysis.json',analysis)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    rounds=m['shape']['rounds'];n=m['shape']['n'];d=m['shape']['d'];uploaded=n-int(n*.2)
    fig,axes=plt.subplots(1,2,figsize=(12,5.4))
    for scheme in SCHEMES:
        for member,style in [('fixed','-'),('changing','--')]:
            for ax,metric in zip(axes,['cumulative_work_s','amortized_work_s']):
                ax.plot(np.arange(1,rounds+1),curves[scheme][member][metric],style,color=COLORS[scheme],
                    label=NAMES[scheme]+' / '+member,linewidth=1.6,marker='o' if member=='fixed' else 's',
                    markevery=[0,rounds-1],markersize=3)
    for ax,title,ylabel in zip(axes,['Measured cumulative work','Measured amortized work'],
            ['All-role work including setup (s)','Cumulative work / completed rounds (s)']):
        ax.set_title(title);ax.set_xlabel('Completed rounds');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        ax.set_xlim(.6,rounds+.4);ax.set_ylim(bottom=0,top=ax.get_ylim()[1]*1.05)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False,fontsize=8.5)
    fig.text(.5,.06,f'N={n}, d={d}; hierarchical M={m["shape"]["m"]}, no edges in VLMRS; {uploaded} contributors per trajectory, paired inputs.',ha='center',fontsize=9)
    fig.text(.5,.025,'Setup once per trajectory; no warmup exclusion. Declared implementations; local work is not deployed latency.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.075,right=.985,bottom=.18,top=.78,wspace=.3)
    fig.savefig(output/'multiround_work.png',dpi=200);fig.savefig(output/'multiround_work.svg');plt.close(fig)
    for metric,stem,ylabel in [('work_s','dropout_work','All-role computation per attempted round (s)'),
                              ('traffic_mib','dropout_communication','Serialized protocol envelopes per attempted round (MiB)')]:
        fig,axes=plt.subplots(1,2,figsize=(12,5.4))
        for ax,stage in zip(axes,['before_upload','after_upload']):
            for scheme in SCHEMES:
                data=sorted([x for x in cells if x['stage']==stage and x['scheme']==scheme],key=lambda x:x['rate'])
                x=np.array([o['rate']*100 for o in data]);y=np.array([o[metric]['median'] for o in data])
                lo=np.array([o[metric]['q25'] for o in data]);hi=np.array([o[metric]['q75'] for o in data])
                ax.errorbar(x,y,yerr=[y-lo,hi-y],color=COLORS[scheme],label=NAMES[scheme],marker='o',markersize=3,capsize=3,linewidth=1.5)
            ax.set_title('Before upload' if stage=='before_upload' else 'After upload')
            ax.set_xlabel('Unavailable clients (%)');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
            ax.set_xlim(-1.5,31.5);ax.set_xticks([0,10,20,30]);ax.set_ylim(bottom=0,top=ax.get_ylim()[1]*1.05)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False)
        fig.text(.5,.06,f'N={n}, d={d}; 10 common independent traces per cell; median and IQR, all attempts retained.',ha='center',fontsize=9)
        fig.text(.5,.025,'Before: sum received uploads. After: sum all uploads, fewer recovery/model recipients. Setup reported separately.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.085,right=.985,bottom=.18,top=.82,wspace=.32)
        fig.savefig(output/(stem+'.png'),dpi=200);fig.savefig(output/(stem+'.svg'));plt.close(fig)
    lines=['# 三方案多轮与分阶段掉线：小规模正式补充结果','',
        f"注册{n}人、分层方案M={m['shape']['m']}、维度{d}。正式共{len(observations)}次尝试，{analysis['valid']}次原始整数和逐维正确且全部指定验证者通过。输入、返回整数和、实际上传/恢复/模型接收名单、初始化、执行次序、原始计时与实际字节均已归档并独立重算。",'',
        '这是N20/d1000的正式补充，原计划N50/d7850跨方案50轮及N100/d7850掉线仍待扩展。已完成的HVSA同人数100轮保留，但没有混入本批三方案成本。GPVO/VLMRS沿用已声明的实例化及条件安全局限，未增加认证或刷新机制，不作原文安全等价或SOTA声明。','',
        '## 同人数多轮','',
        f'固定{uploaded}人和变化{uploaded}人各10轮，同轮向量池相同、名单不一定相同；各轨迹初始化一次，交替轨迹及轮换方案次序，无预热剔除。下表为真实前缀累计，摊销为累计除以轮数，包含一次初始化。','',
        '| 方案 | 成员 | 前缀轮数 | 累计工作/s | 摊销/s每轮 | 累计封装/MiB |','|---|---|---:|---:|---:|---:|']
    for scheme in SCHEMES:
        for membership in ['fixed','changing']:
            v=curves[scheme][membership]
            for k in [1,5,10]:
                lines.append(f"| {ZH[scheme]} | {'固定' if membership=='fixed' else '变化'} | {k} | {v['cumulative_work_s'][k-1]:.4f} | {v['amortized_work_s'][k-1]:.4f} | {v['cumulative_bytes'][k-1]/2**20:.4f} |")
    lines+=['','两条成员轨迹各自初始化，GPVO各产生一套公共参数。公参生成时间随随机参数变化，可能显著改变短轨迹的累计及摊销值；不能把固定/变化曲线差异全部归因于成员变化。为区分来源，列出真实初始化与后续实际轮次工作：','',
        '| 方案 | 成员 | 初始化/s | 实际10轮角色工作之和/s |','|---|---|---:|---:|']
    for scheme in SCHEMES:
        for membership in ['fixed','changing']:
            v=curves[scheme][membership]
            lines.append(f"| {ZH[scheme]} | {'固定' if membership=='fixed' else '变化'} | {sum(v['initialization_role_ms'].values())/1000:.4f} | {sum(v['round_work_s']):.4f} |")
    lines+=['',f'![实际多轮累计与摊销]({(output/"multiround_work.png").as_posix()})','',
        '多轮收益只能结合实测生命周期解释：HVSA注册种子复用；GPVO公参复用、每轮重新产生会话密钥与随机量；VLMRS长期密钥复用，但每轮仍有分享、恢复和剩余客户端的完整配对验证。运行次数增加不等于每轮成本趋近零。两条轨迹的差异包含运行波动；不是训练收敛或独立多机重复实验。','',
        '## 分阶段掉线','',
        '每个阶段/比例10条共同独立轨迹，四档缺席名单为同一随机排列的嵌套前缀，上传前/后共用相同输入与名单，不为成功率重采样。上传前只聚合完整上传者；上传后仍聚合全部20人，但只对仍在线者恢复/交付。VLMRS的“上传前”是在名单及分享已分发之后；恢复阈值14，30%缺席后仍有14人，本阶段未测试越阈值。','',
        '| 阶段 | 缺席率 | 方案 | 上传/仍在线 | 正确且完整验证/尝试 | 全角色工作/s，中位数[IQR] | 封装/MiB，中位数 |','|---|---:|---|---|---:|---:|---:|']
    for x in cells:
        w=x['work_s'];t=x['traffic_mib']
        lines.append(f"| {'上传前' if x['stage']=='before_upload' else '上传后'} | {int(x['rate']*100)}% | {ZH[x['scheme']]} | {x['n_uploaded'][0]}/{x['n_recipients'][0]} | {x['valid']}/{x['trials']} | {w['median']:.4f} [{w['q25']:.4f},{w['q75']:.4f}] | {t['median']:.4f} |")
    for stem in ['dropout_work','dropout_communication']:lines+=['',f'![{stem}]({(output/(stem+".png")).as_posix()})']
    lines+=['','失败及部分工作均包含在上述分布中；如出现未完成，应结合下列原因与成功率阅读，不能将失败回合的较低成本解释为优化。两阶段的聚合人数/验证接收人数不同，曲线不是纯恢复模块的增量成本。初始注册与公参成本另列，不重复计入每条恢复轨迹。','',
        '| 方案/情形 | 初始全角色工作/ms | 初始封装/字节 |','|---|---:|---:|']
    for item in c['initialization']:
        lines.append(f"| {ZH[item['scheme']]} / {item['setting']} | {sum(item['role_ms'].values()):.4f} | {item['sent_bytes']} |")
    failed=[o for o in observations if not o['valid']]
    lines+=['','## 失败记录','']
    if failed:
        lines+=['| 方案 | 情形 | 轨迹序号，从0开始 | 完成 | 原因 |','|---|---|---:|---|---|']
        for o in failed:lines.append(f"| {ZH[o['scheme']]} | {o['setting']} | {o['round']} | {o['completed']} | {o['failure_reason']} |")
    else:lines+=['本批没有观察到失败；不外推为任意分布、越阈值或任意离线阶段都可完成。']
    lines+=['','## 证据与边界','',
        '所有工作为真实密码操作、封装和本地完整回放，不是部署网络时延；计量后端及缓存差异沿用成本报告，不能把全部差异归因于协议。三方案具有不同可信设施与验证责任，成本表不表示同等安全排名。轮次是同一状态的轨迹，不是独立密钥重复。E4为10条独立输入/名单种子，每个格子只有一个初始化状态。', '',
        f"运行：[{m['run_id']}]({source.as_posix()})；manifest SHA256：`{digest(source/'manifest.json')}`。",'',
        '图像生成后须实际检查最终PNG；此报告生成动作本身不证明已目视核查。']
    document=Path(args.document).resolve()
    if document.exists():raise FileExistsError(document)
    document.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copyfile(__file__,output/'analysis_source.py')
    dump_json(output/'analysis_manifest.json',dict(source_run=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        libraries=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),
        files_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        report=dict(path=str(document),sha256=digest(document)),visual_review='pending_actual_png_inspection'))
    print('REPORT='+str(document))


if __name__=='__main__':main()
