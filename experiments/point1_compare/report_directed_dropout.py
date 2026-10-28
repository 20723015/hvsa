# -*- coding: utf-8 -*-
"""Report actual availability boundaries without turning failures into successes."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import shutil
from .artifacts import digest, dump_json
from .directed_dropout_contract import SCHEMES, GPVO, VLMRS, specifications, summary
from .validate_directed_dropout import validate

LABELS = {
    'control':'无缺席对照', 'whole_before':'整组4人上传前缺席',
    'whole_after':'同组4人上传后离线', 'adjacent_before':'同环连续2人上传前缺席',
    'alternating_before':'同环交替2人上传前缺席', 'spread_before':'跨组2人上传前缺席',
    'at_threshold_before':'剩14人，上传前缺席', 'at_threshold_after':'剩14人，上传后离线',
    'below_threshold_before':'剩13人，上传前缺席', 'below_threshold_after':'剩13人，上传后离线',
    'all_after':'全体上传后离线', 'none_before':'无人上传'}
ENGLISH = ['No dropout', 'Whole group / before upload', 'Whole group / after upload',
    'Adjacent ring peers / before', 'Alternating ring peers / before', 'Across groups / before',
    'At threshold (14) / before', 'At threshold (14) / after',
    'Below threshold (13) / before', 'Below threshold (13) / after',
    'All offline / after upload', 'No uploads']
NAMES = {'hvsa':'HVSA', GPVO:'GPVO明确实例化', VLMRS:'VLMRS明确实例化'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run');parser.add_argument('--output', required=True)
    parser.add_argument('--document', required=True)
    args = parser.parse_args(argv)
    root, out, document = Path(args.run).resolve(), Path(args.output).resolve(), Path(args.document).resolve()
    problems = validate(root)
    if problems:
        raise ValueError(problems)
    out.mkdir(parents=True, exist_ok=False)
    if document.exists():
        raise FileExistsError(document)
    corr = json.loads((root/'correctness.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    obs = corr['observations']
    cells = []
    for sp in specifications():
        for scheme in SCHEMES:
            rows = [o for o in obs if o['case']==sp['case'] and o['scheme']==scheme]
            cells.append(dict(case=sp['case'], scheme=scheme, calls=len(rows),
                completed=sum(o['completed'] for o in rows), valid=sum(o['valid_aggregate'] for o in rows),
                unavailable=sum(not o['completed'] for o in rows),
                matches_expected=sum(o['matches_expected_outcome'] for o in rows),
                uploaded=len(sp['uploaded']), recipients=len(sp['recipients']),
                actual_recipient_counts=[len(o['actual_recipients']) for o in rows],
                reveal_counts=[o['reveal_count'] for o in rows],
                reasons=[o['failure_reason'] for o in rows]))
    analysis = dict(kind='functional_availability_diagnosis', run=str(root),
        source_manifest_sha256=digest(root/'manifest.json'), source_correctness_sha256=digest(root/'correctness.json'),
        shape=manifest['shape'], observations=len(obs), valid_aggregates=corr['valid_aggregates'],
        unavailable=corr['unavailable'], unexpected_outcomes=corr['unexpected_outcomes'],
        common_input_pools=3, initializations=len(corr['initialization']), cells=cells,
        per_scheme=[dict(scheme=s, calls=sum(o['scheme']==s for o in obs),
            valid=sum(o['scheme']==s and o['valid_aggregate'] for o in obs),
            unavailable=sum(o['scheme']==s and not o['completed'] for o in obs)) for s in SCHEMES],
        note='Counts describe predeclared functional cases, not random-dropout probabilities or a security ranking.')
    dump_json(out/'analysis.json', analysis)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
    fig, ax = plt.subplots(figsize=(11,8.5))
    colors = {'verified':'#d4e9df', 'unavailable':'#f5e1ca', 'no_receivers':'#d8e5f3', 'mixed':'#f3cccc'}
    for i, sp in enumerate(specifications()):
        for j, scheme in enumerate(SCHEMES):
            cell = next(x for x in cells if x['case']==sp['case'] and x['scheme']==scheme)
            count, valid, unavail = cell['calls'], cell['valid'], cell['unavailable']
            status = 'verified' if valid==count else 'unavailable' if unavail==count else 'mixed'
            if status=='verified' and cell['recipients']==0:
                status='no_receivers'
            ax.add_patch(Rectangle((j,i),1,1,facecolor=colors[status],edgecolor='white',linewidth=3))
            text = f'{valid}/{count} verified' if valid else f'{unavail}/{count} unavailable'
            if status=='no_receivers':text+='\n0 recipients'
            ax.text(j+.5,i+.5,text,ha='center',va='center',fontsize=10,color='#203b44')
    ax.set_xlim(0,3);ax.set_ylim(12,0)
    ax.set_xticks([.5,1.5,2.5]);ax.set_xticklabels(['HVSA','GPVO declared','VLMRS declared'])
    ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=8)
    ax.set_yticks([i+.5 for i in range(12)])
    ax.set_yticklabels([f'{name}  [{len(sp["uploaded"])}/{len(sp["recipients"])}]' for name,sp in zip(ENGLISH,specifications())],fontsize=9)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title('Directed dropout: measured completion and full verification',y=1.085,fontsize=13)
    fig.legend(handles=[Patch(facecolor=colors['verified'],label='Correct sum + all verifiers'),
        Patch(facecolor=colors['unavailable'],label='Unavailable (not tamper rejection)'),
        Patch(facecolor=colors['no_receivers'],label='Verified, no client delivery')],
        loc='lower center',bbox_to_anchor=(.5,.067),ncol=1,frameon=False,fontsize=9)
    fig.text(.5,.045,'N=20, d=257; hierarchical M=5; VLMRS threshold=14. Brackets: uploads / recovery recipients.',ha='center',fontsize=8.5)
    fig.text(.5,.022,'3 shared input pools; one initialized state per scheme. Functional cases, not success probabilities or a security ranking.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.39,right=.98,bottom=.20,top=.86)
    fig.savefig(out/'directed_availability.png',dpi=200)
    fig.savefig(out/'directed_availability.svg');plt.close(fig)
    text = [
        '# 三方案定向掉线边界结果（2026-10-24）', '',
        f'固定12种情形、三方案共{len(obs)}次真实协议调用；其中{corr["valid_aggregates"]}次生成正确原始整数和且全部指定验证者通过，{corr["unavailable"]}次未完成聚合。与运行前预期不符的结果为{corr["unexpected_outcomes"]}次。未完成全部单独保留，不称为108次聚合成功，也不称为篡改拒绝。', '',
        '## 1. 范围与共同条件', '',
        '注册20人；HVSA/GPVO分5组，每组4人，组内按数值顺序成环；VLMRS为全邻居平面方案、恢复阈值14。维度257，种子42，输入为幅度小于10^9的有符号整数。三份独立向量池在所有情形/方案间复用；同一重复编号使用同一输入和同一名单，不声称36份独立输入或3套独立密钥。每方案仅初始化一次，每次调用轮次唯一，GPVO依冻结生命周期逐轮更新DH会话。', '',
        '本轮是功能边界诊断，无预热或正式性能分布；真实初始化、部分工作和逐条通信保留在归档，但不按这些耗时排名。边缘与TA始终在线。上传前缺席发生在本轮必要名单/秘密分享完成之后；上传后离线仍计入已上传贡献，减少在线恢复者/接收者。', '',
        '## 2. 实际结果', '',
        '表内“通过”均表示原始和逐维正确且全部指定验证者实际执行并通过；“未完成”表示尚无已验证聚合，不是验证返回False。每格3次，共用该方案的一次初始化。', '',
        '| 情形 | 上传/在线恢复或接收人数 | HVSA | GPVO明确实例化 | VLMRS明确实例化 |',
        '|---|---:|---|---|---|']
    for sp in specifications():
        row = []
        for scheme in SCHEMES:
            cell = next(c for c in cells if c['case']==sp['case'] and c['scheme']==scheme)
            value = f'{cell["valid"]}/3通过；{cell["unavailable"]}/3未完成'
            if cell['valid']==3:value='3/3通过'
            elif cell['unavailable']==3:value='3/3未完成'
            if cell['valid']==3 and not sp['recipients']:value+='，0接收者'
            row.append(value)
        text.append(f'| {LABELS[sp["case"]]} | {len(sp["uploaded"])}/{len(sp["recipients"])} | '+ ' | '.join(row)+' |')
    text += ['', '![定向掉线完成与验证](../figures/point1_compare/20261024_directed_dropout/directed_availability.png)', '',
        '## 3. 结果解释', '',
        '整组4人上传前缺席时，HVSA实际仅有4组上报，3/3次停在点一同步屏障；相同4人上传后离线时，20人的贡献仍计入，3/3次完整验证通过并向16人下发。该边界与此前N100/d7850的HVSA单方案诊断一致，不能由分散/随机掉线成功推出任意集中掉线均可完成。GPVO本项目适配器跳过空组，并按实际活跃组构造承诺/掩码辅助，VLMRS仍有16个恢复者；两者在所测整组上传前情形均完成。此处不将适配器空组处理策略提升为原论文任意掉线定理。', '',
        '环邻居位置补充观察如下。连续两个缺席者之间的掩码无需从在线客户端揭示；只处理缺席/在线的边界。交替缺席产生更多边界。该计数说明当前GPVO环恢复路径的实际执行，不证明邻居掩码的额外隐私价值或原文安全性质。', '',
        '| GPVO上传前两人缺席位置 | 三次实际邻居掩码揭示消息数 | 正确且完整验证 |', '|---|---|---|']
    for case in ['adjacent_before','alternating_before','spread_before']:
        cell = next(c for c in cells if c['case']==case and c['scheme']==GPVO)
        text.append(f'| {LABELS[case]} | '+ '、'.join(map(str,cell['reveal_counts']))+f' | {cell["valid"]}/3 |')
    text += ['',
        '恢复者恰好14人时，三方案在上传前/后两阶段均完成；降至13人时，VLMRS本项目实例化两阶段各3/3次因恢复份额不足而未完成，verified=null，未进入完整配对验证。HVSA/GPVO在同一名单下均完成。两阶段上传人数分别13和20，不能把两者参考和混为同一向量。该结论限于已固定14人阈值，不能用其作为不同阈值配置的普遍优劣判断。', '',
        '全体20人上传后离线时，HVSA/GPVO各3/3次由全部5个边缘完成正确性验证，实际模型接收者均为0；VLMRS无足够恢复份额，3/3次未完成。边缘验完不等于客户端已收到模型。无人上传时，三方案各3/3次均未完成；数学零参考不作为已发布的空聚合。', '',
        '## 4. 核验与限制', '',
        '独立核验重新生成3份共同输入、逐次求实际上传者的明文整数和，并读取全部实际返回和NPY；逐验证者判定与实际验证阶段/接收消息相互核对。所有未完成调用与已消耗的部分工作保存在failures及原始计量中，不删除不利情形，不重采样。每方案合计：', '',
        '| 方案 | 实际调用 | 正确且完整验证 | 未完成 |', '|---|---:|---:|---:|']
    for row in analysis['per_scheme']:
        text.append(f'| {NAMES[row["scheme"]]} | {row["calls"]} | {row["valid"]} | {row["unavailable"]} |')
    text += ['',
        '这些比例是人为指定边界覆盖数，不是随机掉线成功概率或安全性排名。协议关键源码与10月20日成本阶段一致；GPVO整数群LHF、生成元分布与后端，VLMRS公开根指数与长期密钥复用的条件现象均保留，不因本轮诚实结果正确而变成原文等价或SOTA。既有300项分段多轮/掉线与本轮108次调用分开统计。', '',
        '本轮补齐小规模跨方案定向边界；统一条件连续性能、N50/d7850的三方案50轮与N100/d7850掉线仍待扩展。结果纳入新论文稿v4，旧v3、暂停源及已录用论文/PPT不改。', '',
        '## 证据入口', '',
        '- [运行前规格](POINT1_DIRECTED_DROPOUT_STAGE_20261024.md)。',
        f'- [原始manifest](../results/point1_compare/{root.name}/manifest.json)、[逐次结果](../results/point1_compare/{root.name}/correctness.json)、[未完成记录](../results/point1_compare/{root.name}/failures.jsonl)。',
        '- [分析数据](../figures/point1_compare/20261024_directed_dropout/analysis.json)、[图表来源清单](../figures/point1_compare/20261024_directed_dropout/analysis_manifest.json)。',
        '- 最终图片需实际目视核查后单独保存凭据；核验/拒伪测试及论文版本摘要在最终证据凭据中记录。', '']
    document.write_text('\n'.join(text), encoding='utf-8')
    shutil.copyfile(__file__, out/'analysis_source.py')
    for filename in ['directed_dropout_contract.py','validate_directed_dropout.py']:
        shutil.copyfile(Path(__file__).with_name(filename), out/filename)
    outputs = ['analysis.json','analysis_source.py','directed_dropout_contract.py',
               'validate_directed_dropout.py','directed_availability.png','directed_availability.svg']
    dump_json(out/'analysis_manifest.json', dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_run=str(root), source_manifest_sha256=digest(root/'manifest.json'),
        outputs={name:digest(out/name) for name in outputs},
        visual_review_status='pending_actual_png_view'))
    print(json.dumps(dict(report=str(document), figure=str(out/'directed_availability.png'),
        observations=len(obs), valid=corr['valid_aggregates'], unavailable=corr['unavailable']), ensure_ascii=False))


if __name__ == '__main__':
    main()
