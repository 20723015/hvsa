"""Derive a bounded behavior report only from a completed declared E5 run."""
from __future__ import annotations
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import shutil
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CODEX.md').is_file())
sys.path.insert(0,str(ROOT))
from experiments.point1_compare.artifacts import digest,dump_json
from experiments.point1_compare.run_decided_integrity import GPVO,VLMRS,TYPES
from experiments.point1_compare.validate_decided_integrity import validate

NAMES={GPVO:'GPVO本项目实例化',VLMRS:'VLMRS本项目实例化'}
LABELS={'result_only':'仅改结果','proof_only':'仅改证明',
    'coordinated_result_proof':'协调修改结果与证明','replay_same_set':'相同上传集合跨轮重放',
    'replay_changed_set':'不同上传集合跨轮重放','zero_extended_dimension':'末尾补零扩维（接口诊断）'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--output',required=True)
    parser.add_argument('--document',required=True)
    args=parser.parse_args()
    source=Path(args.run).resolve();problems=validate(source)
    if problems:raise ValueError(repr(problems))
    m=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    c=json.loads((source/'correctness.json').read_text(encoding='utf-8'))
    counts=Counter((t['scheme'],t['tamper_type'],t['accepted']) for t in c['trials'])
    records=[]
    for s in m['schemes']:
        for kind in TYPES:
            ts=[t for t in c['trials'] if t['scheme']==s and t['tamper_type']==kind]
            records.append({'scheme':s,'tamper_type':kind,'trials':len(ts),
                'accepted':counts[s,kind,True],'rejected':counts[s,kind,False],
                'input_guard_rejections':sum(bool(t['input_rejection']) for t in ts),
                'accepted_wrong_value':sum(t['accepted'] and t['same_shape'] and not t['exact_match'] for t in ts),
                'accepted_wrong_dimension':sum(t['accepted'] and not t['same_shape'] for t in ts)})
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    dump_json(output/'analysis.json',{'kind':'analysis_of_existing_evidence','records':records,
        'honest_contexts':len(c['honest_contexts']),'tamper_verifier_calls':len(c['trials']),
        'lifecycle_diagnostics':c['lifecycle_diagnostics'],
        'source_manifest':{'path':str(source/'manifest.json'),'sha256':digest(source/'manifest.json')},
        'interpretation':'Counts of real verifier calls, not independent end-to-end malicious broadcasts or a security probability.'})
    sizes={s['setting']:s for s in m['settings']}
    per=sum(s['count'] for s in m['settings'])
    lines=['# GPVO与VLMRS明确实例化：生产参数异常行为结果','',
        '日期：2026-10-23。本报告接续四点成本矩阵，使用相同协议核心源码和实例化决定，新增真实异常验证记录。它不改写此前成本数据，也不证明与作者实现或原文安全等价。','',
        '## 实际执行范围','',
        f"两种明确实例化各测试{per}份独立当前输入；每份覆盖六类，共{len(c['trials'])}次代表验证者调用。每方案每个维度另执行两个诚实重放锚点，合计{len(c['honest_contexts'])}个完整诚实上下文，全部原始整数和逐维正确、全部指定诚实验证者执行并通过。",'',
        '正式设置为N10/M2下32维和257维各50份，N20/M5下7850维10份。VLMRS无边缘角色。GPVO固定3072位RSA模数、素数e>N_RSA²、Z*_(N_RSA²)乘法LHF及逐轮新DH会话；VLMRS固定真实BLS12-381、全邻居、ceil(2N/3)阈值、公开根指数7及长期DH密钥复用。若本报告输入为缩小试跑，以文末逐设置记录为实际范围，不能冒称100+10正式矩阵。','',
        '每类篡改调用一个代表边缘或客户端的实际验证函数。其输入来自实际诚实协议交付的公开结果与证明，构造不读取诚实向量、HF/PRF密钥或其他验证秘密。它是验证行为检查，不是恶意云向所有验证者广播的端到端时延实验，也不并入此前120次正式成本观测。','',
        'GPVO的诊断上下文另存验证者所需的TA辅助量，供实际验证函数使用；攻击构造只读取candidate、proof和公开params，不读取这些辅助量，不将本地归档字段误写成云实际收到的信息。','',
        '## 各类行为','',
        '表中格式为“拒绝数 / 调用数（接受数）”。这些比例描述有限用例，不是安全概率，也不能按总拒绝率排列协议安全性。','',
        '| 用例 | GPVO本项目实例化 | VLMRS本项目实例化 |','|---|---:|---:|']
    for kind in TYPES:
        cells=[]
        for s in [GPVO,VLMRS]:
            r=next((r for r in records if r['scheme']==s and r['tamper_type']==kind),None)
            cells.append(f"{r['rejected']} / {r['trials']}（{r['accepted']}）" if r else '未执行')
        lines.append('| '+LABELS[kind]+' | '+' | '.join(cells)+' |')
    lines += ['',
        '协调修改与重放的结果必须结合当前实例化解释：GPVO的回传聚合承诺没有另外增加独立认证锚点；VLMRS的根指数公开且证明标签为静态sum。记录中出现的错误接受保留为该实现条件下的行为，不直接命名为原论文漏洞。相同集合重放比较的是旧向量/旧证明与新输入的参考和；不同集合重放使用另一个真实上传名单的旧锚点。','',
        '此处重放仅指旧结果与旧证明原样重放，不包含利用当前结果和证明重新协调构造一个旧值；后者属于协调修改类。因此直接重放被拒绝不等于所有形式的旧值替换均被阻断。','',
        '每个维度的旧锚点在该维度各当前输入中复用，因此每类有110份独立当前输入，并不等于110份独立旧重放载荷或110套独立密钥。公参与长期密钥按各方案策略复用，GPVO逐轮DH会话更新。两方案当前输入配对，输入NPZ、实际名单和来源摘要均已核对。','',
        '### 输出维度接口问题单列','',
        '末尾补零不改变不含长度约束的线性哈希值。当前GPVO适配器的verify_candidate只限制一维int64，没有显式检查公开维度；该接口是否接受由表中实测记录给出。VLMRS的HF入口检查公开维度，异常时抛ValueError，记录为输入拒绝，与验证等式返回False分开保存，不声称完整链路已优雅处理该异常。此项定位为本项目适配器/编码接口检查，不能拿来宣称HVSA协议优于GPVO原文或作为新增密码学贡献。后续若修复必须另建源码版本，保留本轮与既有成本归档。','',
        '## 长期密钥恢复的独立诊断','',
        'VLMRS每个维度先执行一轮全部上传，再执行一轮u0上传前缺席；第二轮恢复其长期掩码私钥后，使用云已有的旧密文、已恢复个人种子与后来得到的私钥重构u0先前向量。攻击构造不读取参考向量或验证密钥，参考仅用于事后比较。以下计数独立于六类异常验证，不额外加进1320次。','',
        '| 维度 | 过去轮次 | 后续恢复轮次 | 重构是否逐维一致 |','|---:|---:|---:|---|']
    for item in c['lifecycle_diagnostics']:
        lines.append(f"| {sizes[item['setting']]['d']} | {item['past_round']} | {item['reconstruction_round']} | {'是' if item['exact_match'] else '否'} |")
    lines += ['',
        '该现象限定为本项目长期掩码DH密钥复用且后续允许恢复该私钥的选择；它不是对原文全部生命周期解释的统一判决，也不通过暗改刷新策略消除既有记录。','',
        '## 记录核验与结论边界','',
        '独立核验检查归档源码、此前成本来源链、公开参数/规格、输入NPZ及同编码参考、实际公开载荷、六类构造、诚实验证者数量、逐次验证结果/耗时和汇总的一致性。它不重跑秘密密钥验证，也不把来源一致性当作安全证明。','',
        '新入口试跑独立于本正式矩阵。首次独立调用发现配对依赖路径缺失，已修复核验入口；未改原试跑数据。随后8项测试通过（57.35秒），包括重新计算输出摘要后伪造结果、载荷、参考和、完整验证者数量及汇总仍被拒绝。该次测试使用试跑归档，不声称全部历史测试在同一次调用中重跑。','',
        'HVSA既有18类1980次拒绝与本轮基线六类矩阵不是相同用例全集，不合并成一个总成功率，也不直接以拒绝次数多少作安全排名。完整跨方案多轮、同人数对照和分阶段掉线仍需后续执行。当前结果不支持SOTA或同等安全排名。','',
        '## 逐设置次数及来源','',
        '| 维度 | 注册人数 | 分层M | 每方案当前输入 | 每方案锚点 |','|---:|---:|---:|---:|---:|']
    for sp in m['settings']:lines.append(f"| {sp['d']} | {sp['N']} | {sp['M']} | {sp['count']} | 2 |")
    lines += ['',f"原始运行：`{source}`；manifest SHA-256 `{digest(source/'manifest.json')}`。",'',
        f"分析目录：`{output}`。`analysis.json`保存逐类数量与来源，原始目录保留所有NPZ、公开pickle载荷、raw/messages/correctness/summary及完成前缀。",'',
        '执行规格见[生产参数异常阶段](POINT1_DECLARED_INTEGRITY_STAGE_20261023.md)；既有[成本报告](POINT1_DECIDED_MINIMUM_COST_COMPARISON_20261020.md)保持原样。','']
    report='\n'.join(lines)
    (output/'RESULTS_SUMMARY.md').write_text(report,encoding='utf-8')
    shutil.copy2(__file__,output/'analysis_source.py')
    doc=Path(args.document).resolve();doc.write_text(report,encoding='utf-8')
    dump_json(output/'analysis_manifest.json',{'kind':'analysis_of_existing_evidence',
        'created_at':datetime.datetime.now().astimezone().isoformat(),
        'source_manifest':{'path':str(source/'manifest.json'),'sha256':digest(source/'manifest.json')},
        'document':{'path':str(doc),'sha256':digest(doc)},
        'output_sha256':{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    print('REPORT='+str(doc),flush=True)


if __name__=='__main__':main()
