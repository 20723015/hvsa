# -*- coding: utf-8 -*-
"""Report complete N100 E4 archives, retaining every returned cost point."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import shutil
import numpy as np
from .artifacts import dump_json
from .large_dropout_contract import SCHEMES,GPVO,VLMRS,labels,summaries
from .multirun_environment import summarize as conditions
from .round_shards import load,file_sha
from .validate_large_dropout import validate

NAMES={'hvsa':'HVSA',GPVO:'GPVO declared',VLMRS:'VLMRS declared'}
SHORT={'hvsa':'H',GPVO:'G',VLMRS:'V'}
COLORS={'hvsa':'#2474a6',GPVO:'#de8a25',VLMRS:'#3b9866'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run');parser.add_argument('--output',required=True);parser.add_argument('--document',required=True)
    a=parser.parse_args(argv);root=Path(a.run).resolve();out=Path(a.output).resolve();doc=Path(a.document).resolve()
    errors=validate(root)
    if errors:raise ValueError(errors)
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'));s=m['shape']
    if s['smoke']:raise ValueError('Smoke excluded from formal results')
    if doc.exists():raise FileExistsError(doc)
    out.mkdir(parents=True,exist_ok=False)
    journal,entries=load(root);obs=[e for e in entries if e['kind']=='round'];cells=[]
    for label in labels(s):
        stage,rate=label.rsplit('_',1)
        for scheme in SCHEMES:
            init=next(e for e in entries if e['kind']=='init' and e['label']==label and e['scheme']==scheme)
            part=[e for e in obs if e['label']==label and e['scheme']==scheme]
            work=[sum(e['role_ms'].values())/1000 for e in part];wire=[e['total_sent_bytes'] for e in part]
            cells.append(dict(label=label,stage=stage,rate=int(rate),scheme=scheme,
                rounds=[e['round'] for e in part],setup_s=sum(init['role_ms'].values())/1000,
                setup_bytes=init['total_sent_bytes'],round_work_s=work,round_wire_bytes=wire,
                role_round_ms=[e['role_ms'] for e in part],
                work_q25_median_q75=np.percentile(work,[25,50,75]).tolist(),
                wire_q25_median_q75=np.percentile(wire,[25,50,75]).tolist(),
                completed=sum(e['completed'] for e in part),valid=sum(e['valid'] for e in part),
                matches_expected=sum(e['matches_expected'] for e in part),
                recipient_counts=[len(e['recipients']) for e in part],uploaded_counts=[len(e['uploaded']) for e in part],
                verifier_calls=sum(len(e['verdicts']) for e in part)))
    analysis=dict(kind='complete_N100_staged_dropout',run=str(root),shape=s,
        manifest_sha256=file_sha(root/'manifest.json'),journal_sha256=file_sha(root/'journal.json'),
        calls=len(obs),valid=sum(e['valid'] for e in obs),unavailable=sum(not e['completed'] for e in obs),
        matches_expected=sum(e['matches_expected'] for e in obs),verifier_calls=sum(len(e['verdicts']) for e in obs),
        initialization_count=sum(e['kind']=='init' for e in entries),cells=cells,
        summaries=summaries(entries,s),conditions=conditions(entries))
    dump_json(out/'analysis.json',analysis)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    stages=['before_upload','after_upload'];titles=['Before upload','After upload (all 100 contributions retained)']
    for stem,key,scale,ylabel,title in [
        ('work_by_dropout','round_work_s',1,'All-role exclusive work per call (s)','Measured protocol work by dropout stage'),
        ('wire_by_dropout','round_wire_bytes',2**20,'Serialized sent envelopes per call (MiB)','Measured protocol communication by dropout stage')]:
        fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
        ymax=max(max(c[key])/scale for c in cells)*1.12
        for ax,stage,panel in zip(axes,stages,titles):
            for si,scheme in enumerate(SCHEMES):
                part=[c for c in cells if c['stage']==stage and c['scheme']==scheme]
                offset=(si-1)*1.5;xs=[];ys=[];lo=[];hi=[]
                for c in part:
                    x=c['rate']+offset;y=np.array(c[key])/scale
                    ax.scatter(x+np.linspace(-.45,.45,len(y)),y,s=16,alpha=.42,color=COLORS[scheme],zorder=2)
                    q=np.percentile(y,[25,50,75]);xs.append(x);ys.append(q[1]);lo.append(q[1]-q[0]);hi.append(q[2]-q[1])
                ax.errorbar(xs,ys,yerr=[lo,hi],fmt='o-',capsize=4,linewidth=1.5,color=COLORS[scheme],label=NAMES[scheme],zorder=3)
            ax.set_title(panel);ax.set_xlabel('Dropped clients (%)');ax.set_xticks([0,10,20,30]);ax.set_xlim(-4,34)
            ax.set_ylim(0,ymax);ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel(ylabel);axes[1].legend(frameon=False,loc='upper right')
        fig.suptitle(title,fontsize=14)
        fig.text(.5,.065,'Dots: all 10 returned calls per cell; lines: median; whiskers: interquartile range. No first-round removal.',ha='center',fontsize=9)
        fig.text(.5,.025,'N=100, d=7850, hierarchical M=5; setup shown separately. Declared implementations; local replay, not deployed latency.',ha='center',fontsize=8.7)
        fig.subplots_adjust(left=.085,right=.985,top=.84,bottom=.19,wspace=.1)
        fig.savefig(out/(stem+'.png'),dpi=180);fig.savefig(out/(stem+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    role_colors={'user':'#5f9ed1','edge':'#dd8452','ta':'#55a868','cloud':'#9a7bb8'}
    ymax=max(np.mean(c['round_work_s']) for c in cells)*1.15
    for ax,stage,title in zip(axes,stages,titles):
        part=[c for c in cells if c['stage']==stage];bottom=np.zeros(len(part));x=np.arange(len(part))
        for role,color in role_colors.items():
            y=np.array([np.mean([r[role] for r in c['role_round_ms']])/1000 for c in part])
            ax.bar(x,y,bottom=bottom,label=role,color=color,width=.73);bottom+=y
        ax.set_xticks(x);ax.set_xticklabels([SHORT[c['scheme']] for c in part]);ax.set_title(title)
        ax.set_ylim(0,ymax);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for j,rate in enumerate([0,10,20,30]):ax.text(j*3+1,-.12,f'{rate}%',transform=ax.get_xaxis_transform(),ha='center')
    axes[0].set_ylabel('Mean exclusive work per returned call (s)');axes[1].legend(frameon=False)
    fig.suptitle('Where protocol computation is performed',fontsize=14)
    fig.text(.5,.065,'H = HVSA; G = GPVO declared; V = VLMRS declared. Mean stacks include all 10 calls; initialization is separate.',ha='center',fontsize=9)
    fig.text(.5,.025,'All designated edges or recovery clients verify; different trust, cryptographic backends and caches remain disclosed.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.075,right=.985,top=.84,bottom=.24,wspace=.1)
    fig.savefig(out/'roles_by_dropout.png',dpi=180);fig.savefig(out/'roles_by_dropout.svg');plt.close(fig)
    for name in ['report_large_dropout.py','validate_large_dropout.py','large_dropout_contract.py']:
        shutil.copy2(Path(__file__).with_name(name),out/name)
    generated=[p for p in out.iterdir() if p.is_file()]
    dump_json(out/'analysis_manifest.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run=str(root),run_manifest_sha256=file_sha(root/'manifest.json'),files={p.name:file_sha(p) for p in generated},
        visual_review='pending_actual_PNG_inspection'))
    env=analysis['conditions'];lines=[
        '# 三方案N100分阶段掉线实验结果','',
        f'正式运行：`{root.name}`。N100/M5/d7850，上传前/后×四档比例×10条共同输入×三方案，共{len(obs)}次真实调用。',
        f"逐维正确且全部指定验证通过{analysis['valid']}次；未完成{analysis['unavailable']}次；符合预声明可用性{analysis['matches_expected']}次；完整验证者调用{analysis['verifier_calls']}次。24个状态各初始化一次。",'',
        '## 阶段语义与证据','',
        '上传前退出不计入贡献；上传后退出仍保留其已上传向量，全部100人参与参考整数和。两阶段共用嵌套缺席名单与共同输入。独立核验实际NPZ、原始返回和、全部指定验证者/实际接收者、环边界补偿消息及分片/汇总摘要；不以脚本未报错代替这些检查。',
        '这是合成整数协议回放，HVSA使用默认零基模型；不代表联邦训练收敛。10条共同输入/名单对应每格同一初始化状态，不是10次独立密钥/机器重复。', '',
        '## 全部格子与成本','',
        '角色工作和通信均为每次返回调用的实际记录，首轮和异常耗时均保留。中位数及四分位范围只描述本批10条；初始化单列，不把随机公参生成差异归因于掉线。', '',
        '|阶段|掉线率|方案|正确且完整验证/调用|初始化工作秒|每次工作秒 中位数[Q1,Q3]|每次通信MiB 中位数[Q1,Q3]|',
        '|---|---:|---|---:|---:|---:|---:|']
    for c in cells:
        q=c['work_q25_median_q75'];w=np.array(c['wire_q25_median_q75'])/2**20
        lines.append(f"|{'上传前' if c['stage']=='before_upload' else '上传后'}|{c['rate']}%|{NAMES[c['scheme']]}|{c['valid']}/10|{c['setup_s']:.3f}|{q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]|{w[1]:.3f} [{w[0]:.3f}, {w[2]:.3f}]|")
    lines.extend(['','## 图与运行环境',''])
    for stem in ['work_by_dropout','wire_by_dropout','roles_by_dropout']:
        lines.append(f'![{stem}](../figures/point1_compare/{out.name}/{stem}.png)');lines.append('')
    lines.extend([
        f"共{env['snapshots']}份只读快照，全部观察接通电源={env['all_observed_power_plugged']}，观察到供电变化{env['observed_power_transitions']}次；含其他协议worker的快照{env['snapshots_with_other_protocol_workers']}份。报告CPU频率{env['reported_frequency_min_mhz']}—{env['reported_frequency_max_mhz']}MHz。",'',
        '连续单初始化解决此前8+2分段状态问题；温度/频率没有锁定，快照无法证明两次观察之间条件不变，不能称完全统一硬件条件。串行全角色工作不是实际网络完成时间；序列化字节不是TCP/TLS流量。', '',
        'GPVO使用已声明的整数群LHF、素数e>N²、每轮新DH/承诺随机量，生成元未提供最大阶证书；VLMRS使用已声明的全邻居图、公开根接口及长期密钥复用。保留既有条件诊断，不将这些实例称作者唯一原文实现、安全等价或SOTA。', '',
        f'[运行前规格](POINT1_LARGE_DROPOUT_STAGE_20261024.md) · [原始清单](../results/point1_compare/{root.name}/manifest.json) · [完整核验](../results/point1_compare/{root.name}/checkpoints/final_validation.json)',
        f'[逐调用分析](../figures/point1_compare/{out.name}/analysis.json) · [分析摘要](../figures/point1_compare/{out.name}/analysis_manifest.json)', '',
        '图形视觉核查状态以独立凭据为准，本报告生成不自动表示已目视核查。'])
    doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(document=str(doc),output=str(out),calls=len(obs),valid=analysis['valid'])))


if __name__=='__main__':main()
