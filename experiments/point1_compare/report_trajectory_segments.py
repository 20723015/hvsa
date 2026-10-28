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
from experiments.point1_compare.trajectory_bundle import collect
from experiments.point1_compare.run_decided_comparison import SCHEMES,GPVO,VLMRS

NAMES={'hvsa':'HVSA',GPVO:'GPVO declared',VLMRS:'VLMRS declared'}
ZH={'hvsa':'HVSA',GPVO:'GPVO明确实例化',VLMRS:'VLMRS明确实例化'}
COLORS={'hvsa':'#2474a6',GPVO:'#de8a25',VLMRS:'#3b9866'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('prefix');p.add_argument('suffix');p.add_argument('--output',required=True);p.add_argument('--document',required=True)
    args=p.parse_args(argv);output=Path(args.output).resolve()
    bundle,m,c=collect(args.prefix,args.suffix)
    if m['shape']['smoke']:raise ValueError('Smoke is excluded from formal reporting')
    output.mkdir(parents=True,exist_ok=False)
    dump_json(output/'bundle.json',bundle)
    observations=c['observations'];init={(i['setting'],i['scheme']):i for i in c['initialization'] if i['setting'].startswith('e3_')}
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
            epochs=[]
            for epoch in [0,1]:
                eo=[o for o in obs if o['epoch']==epoch]
                setup=next(i for i in c['initialization'] if i['setting']==sp['setting'] and i['scheme']==scheme and i['epoch']==epoch)
                ew=[sum(o['role_ms'].values())/1000 for o in eo]
                et=[o['total_sent_bytes']/2**20 for o in eo]
                epochs.append(dict(epoch=epoch,trials=len(eo),valid=sum(o['valid'] for o in eo),
                    round_indices=[o['round'] for o in eo],work_s=dict(zip(['q25','median','q75'],np.percentile(ew,[25,50,75]).tolist())),
                    traffic_mib=dict(zip(['q25','median','q75'],np.percentile(et,[25,50,75]).tolist())),
                    initialization_work_s=sum(setup['role_ms'].values())/1000,initialization_bytes=setup['sent_bytes']))
            cells.append(dict(setting=sp['setting'],stage=sp['stage'],rate=sp['rate'],scheme=scheme,epochs=epochs,
                trials=len(obs),completed=sum(o['completed'] for o in obs),valid=sum(o['valid'] for o in obs),
                n_uploaded=sorted({o['n_uploaded'] for o in obs}),n_recipients=sorted({len(o['recipients']) for o in obs}),
                work_s=dict(zip(['q25','median','q75'],np.percentile(work,[25,50,75]).tolist())),
                traffic_mib=dict(zip(['q25','median','q75'],np.percentile(traffic,[25,50,75]).tolist())),
                failed_rounds=[dict(round=o['round'],reason=o['failure_reason'],completed=o['completed'],verified=o['verified']) for o in obs if not o['valid']]))
    analysis=dict(kind='analysis_of_existing_evidence',shape=m['shape'],curves=curves,dropout_cells=cells,
        source_bundle=dict(path=str(output/'bundle.json'),sha256=digest(output/'bundle.json')),
        total_observations=len(observations),valid=sum(o['valid'] for o in observations),
        note='E3 is the preserved continuous 10-round trajectory. E4 is 8+2 traces in two independent key-state epochs; both actual setups recorded separately.')
    environment=ROOT/'docs/POINT1_TRAJECTORY_CONTINUATION_ENV_20261024.json'
    analysis['environment_observation']=dict(path=environment.as_posix(),sha256=digest(environment),
        observation=json.loads(environment.read_text(encoding='utf-8')))
    ratios=[x['epochs'][1]['work_s']['median']/x['epochs'][0]['work_s']['median'] for x in cells]
    analysis['epoch_work_ratio_new_over_original']=dict(minimum=min(ratios),maximum=max(ratios),
        interpretation='Observed cross-period variation, not a causal protocol or membership effect.')
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
    for metric,stem,ylabel in [('work_s','dropout_work','All-role round computation, excluding setup (s)'),
                              ('traffic_mib','dropout_communication','Round protocol envelopes, excluding setup (MiB)')]:
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        all_values=[sum(o['role_ms'].values())/1000 if metric=='work_s' else o['total_sent_bytes']/2**20 for o in observations if o['experiment']=='e4']
        for epoch in [0,1]:
            for column,stage in enumerate(['before_upload','after_upload']):
                ax=axes[epoch,column]
                for si,scheme in enumerate(SCHEMES):
                    data=sorted([x for x in cells if x['stage']==stage and x['scheme']==scheme],key=lambda x:x['rate'])
                    x=np.array([o['rate']*100 for o in data])
                    y=np.array([o['epochs'][epoch][metric]['median'] for o in data])
                    lo=np.array([o['epochs'][epoch][metric]['q25'] for o in data]);hi=np.array([o['epochs'][epoch][metric]['q75'] for o in data])
                    ax.errorbar(x,y,yerr=[y-lo,hi-y],color=COLORS[scheme],label=NAMES[scheme],marker='o',markersize=3,capsize=3,linewidth=1.5)
                    for cell in data:
                        oo=[o for o in observations if o['setting']==cell['setting'] and o['scheme']==scheme and o['epoch']==epoch]
                        vv=[sum(o['role_ms'].values())/1000 if metric=='work_s' else o['total_sent_bytes']/2**20 for o in oo]
                        jitter=np.linspace(-.22,.22,len(vv))+(si-1)*.65
                        ax.scatter(cell['rate']*100+jitter,vv,s=9,color=COLORS[scheme],alpha=.35,zorder=3)
                ax.set_title(('Original process, 8 traces: ' if epoch==0 else 'Fresh process, 2 traces: ')+('before upload' if stage=='before_upload' else 'after upload'),fontsize=10)
                ax.set_xlabel('Unavailable clients (%)');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
                ax.set_xlim(-2,32);ax.set_xticks([0,10,20,30]);ax.set_ylim(0,max(all_values)*1.08)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False)
        fig.text(.5,.055,f'N={n}, d={d}; points: all observed rounds; lines/whiskers: within-process median/IQR; shared vertical scale.',ha='center',fontsize=9)
        fig.text(.5,.025,'Local replay; dots have display-only horizontal jitter; two setups/key states and process-period timing are reported separately.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.085,right=.985,bottom=.13,top=.91,wspace=.32,hspace=.42)
        fig.savefig(output/(stem+'.png'),dpi=200);fig.savefig(output/(stem+'.svg'));plt.close(fig)
    lines=['# 三方案多轮与分阶段掉线：小规模正式补充结果','',
        f"注册{n}人、分层方案M={m['shape']['m']}、维度{d}。汇总{len(observations)}项带完整观测记录的用例，{analysis['valid']}项原始整数和逐维正确且全部指定验证者通过。输入、返回整数和、实际上传/恢复/模型接收名单、初始化、执行次序、原始计时与实际字节均已归档并独立重算。",'',
        '这是252个已保存回合加48个新进程回合的分段汇总。E3旧段连续10轮不重跑；E4各格子为8+2条输入/名单轨迹，段间重建密钥和协议状态，两次真实初始化分别计量。不能称为同一长期密钥状态连续10轮。原计划N50/d7850跨方案50轮及N100/d7850掉线仍待扩展。既有HVSA同人数100轮和试跑不混入。GPVO/VLMRS保留已声明实例化及其条件局限，不作原文安全等价或SOTA声明。','',
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
        '每个阶段/比例合计10条共同输入/名单轨迹，前8条与后2条属于两个独立初始化状态。功能用例计数合计，成本列按8/2分开。图按原8条与新2条分行，显示全部原始点、各段中位数/IQR，并共用纵轴。四档缺席名单为同一随机排列的嵌套前缀，上传前/后共用相同输入与名单，不重采样。上传前只聚合完整上传者；上传后仍聚合全部20人，但只对仍在线者恢复/交付。VLMRS的“上传前”在名单及分享已分发之后；恢复阈值14，30%缺席后仍有14人，本阶段未测试越阈值。','',
        '| 阶段 | 缺席率 | 方案 | 上传/仍在线 | 正确且完整验证/尝试 | 旧8条工作/s中位数 | 新2条工作/s中位数 | 封装/MiB旧/新中位数 |','|---|---:|---|---|---:|---:|---:|---:|']
    for x in cells:
        old,new=x['epochs']
        lines.append(f"| {'上传前' if x['stage']=='before_upload' else '上传后'} | {int(x['rate']*100)}% | {ZH[x['scheme']]} | {x['n_uploaded'][0]}/{x['n_recipients'][0]} | {x['valid']}/{x['trials']} | {old['work_s']['median']:.4f} | {new['work_s']['median']:.4f} | {old['traffic_mib']['median']:.4f}/{new['traffic_mib']['median']:.4f} |")
    for stem in ['dropout_work','dropout_communication']:lines+=['',f'![{stem}]({(output/(stem+".png")).as_posix()})']
    lines+=['','协议返回的不完成/失败记录及其已测工作若有，均保留；应结合原因与成功率阅读。用户暂停时未形成完整记录的在途调用不计入这300项，也未估算其耗时。因此本汇总不声称涵盖全部已消耗CPU开销。两阶段的聚合人数/验证接收人数不同，曲线不是纯恢复模块的增量成本。初始注册与公参成本另列，不重复计入每条恢复轨迹。','',
        '## 各段实际初始化','',
        '| 方案/情形 | 段（0旧，1新） | 初始全角色工作/ms | 初始封装/字节 |','|---|---:|---:|---:|']
    for item in c['initialization']:
        lines.append(f"| {ZH[item['scheme']]} / {item['setting']} | {item['epoch']} | {sum(item['role_ms'].values()):.4f} | {item['sent_bytes']} |")
    lines+=['','## 掉线分段统计','',
        '两段分别列出，用于识别密钥重置及跨时段运行波动；不将两段平均差直接归因于协议或掉线率。每段的原始记录全部保留。','',
        '| 方案/情形 | 段 | 样本数 | 正确且完整验证 | 工作/s，中位数[IQR] | 封装/MiB，中位数 |','|---|---:|---:|---:|---:|---:|']
    for x in cells:
        for e in x['epochs']:
            w=e['work_s'];t=e['traffic_mib']
            lines.append(f"| {ZH[x['scheme']]} / {x['setting']} | {e['epoch']} | {e['trials']} | {e['valid']} | {w['median']:.4f} [{w['q25']:.4f},{w['q75']:.4f}] | {t['median']:.4f} |")
    lines+=['',f'同一格子新段/旧段工作中位数之比范围为{min(ratios):.3f}—{max(ratios):.3f}；协议关键源码与库版本相同，时段差异的原因尚未定位，因此不把这个倍数解释为协议或成员变化效果。新段的一次只读环境快照记录为电池供电；旧段没有对应快照，无法据此单独归因。新段内部也有波动，供电/频率没有锁定；分段结果属于描述性观测，不能作严格同条件性能排名。',
        '',f'[新段环境快照]({environment.as_posix()})记录了观测时刻及供电/频率信息，没有更改机器设置。']
    failed=[o for o in observations if not o['valid']]
    lines+=['','## 失败记录','']
    if failed:
        lines+=['| 方案 | 情形 | 轨迹序号，从0开始 | 完成 | 原因 |','|---|---|---:|---|---|']
        for o in failed:lines.append(f"| {ZH[o['scheme']]} | {o['setting']} | {o['round']} | {o['completed']} | {o['failure_reason']} |")
    else:lines+=['本批没有观察到失败；不外推为任意分布、越阈值或任意离线阶段都可完成。']
    lines+=['','## 证据与边界','',
        '所有工作为真实密码操作、封装和本地完整回放，不是部署网络时延；计量后端及缓存差异沿用成本报告，不能把全部差异归因于协议。三方案可信设施与验证责任不同，不表示同等安全排名。E3是旧段连续10轮；E4为两段8+2轨迹，每个格子两次初始化。输入/返回和、来源摘要与计量账本已独立核验，未使用归档中的证明载荷重新执行当时的全部密码验证。', '',
        f"[来源bundle]({(output/'bundle.json').as_posix()})，SHA256：`{digest(output/'bundle.json')}`。",'',
        f"保留前缀：[{Path(args.prefix).name}]({Path(args.prefix).resolve().as_posix()})；新续段：[{Path(args.suffix).name}]({Path(args.suffix).resolve().as_posix()})。",'',
        '图像生成后须实际检查最终PNG；此报告生成动作本身不证明已目视核查。']
    document=Path(args.document).resolve()
    if document.exists():raise FileExistsError(document)
    document.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copyfile(__file__,output/'analysis_source.py')
    for name in ['trajectory_bundle.py','validate_trajectory_segments.py','trajectory_segment_contract.py']:
        target=output/'analysis_dependencies'/name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'experiments/point1_compare'/name,target)
    dump_json(output/'analysis_manifest.json',dict(source_bundle=str(output/'bundle.json'),source_bundle_sha256=digest(output/'bundle.json'),
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        libraries=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),
        files_sha256={p.relative_to(output).as_posix():digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        report=dict(path=str(document),sha256=digest(document)),visual_review='pending_actual_png_inspection'))
    print('REPORT='+str(document))


if __name__=='__main__':main()
