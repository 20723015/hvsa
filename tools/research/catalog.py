"""Static research/experiment inventory; never import or launch experiments."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[2]
TARGET='research/catalog.json'
EXCLUDED_LOCAL_CONFIGURATION={'examples/exp_agent_overhead.py'}
GROUPS=[
 ('shared','公共四角色/密码/学习',['user.py','edge.py','cloud.py','ta.py','utils.py','models.py','datasets.py','point2_types.py','commit_tx.py']),
 ('p1.e1','点一E1正确性',['experiments/point1_compare/run_compare.py','tests/test_point1_compare.py']),
 ('p1.e2','点一E2成本与完整规模扫描',['experiments/point1_compare/run_performance.py','experiments/point1_compare/run_decided_comparison.py','experiments/point1_compare/run_decided_batch.py','experiments/point1_compare/full_scale_contract.py','experiments/point1_compare/run_full_scale_point.py','experiments/point1_compare/run_full_scale_batch.py','experiments/point1_compare/validate_full_scale.py','experiments/point1_compare/report_full_scale.py','tests/test_point1_full_scale.py']),
 ('p1.e3','点一E3多轮',['experiments/point1_compare/run_large_multirun.py','experiments/point1_compare/run_hvsa_paired40.py','experiments/point1_compare/run_decided_trajectories.py','experiments/point1_compare/continue_decided_trajectories.py','experiments/point1_compare/validate_large_multirun.py','tests/test_point1_large_multirun.py']),
 ('p1.e4','点一E4掉线与定向边界',['experiments/point1_compare/run_directed_dropout.py','experiments/point1_compare/directed_dropout_contract.py','experiments/point1_compare/run_large_dropout.py','experiments/point1_compare/validate_large_dropout.py','tests/test_point1_directed_dropout.py','tests/test_point1_large_dropout.py']),
 ('p1.e5','点一E5篡改/重放',['experiments/point1_compare/run_integrity.py','experiments/point1_compare/run_decided_integrity.py','experiments/point1_compare/validate_decided_integrity.py','tests/test_point1_boundary.py']),
 ('p1.e6','点一E6消融',['experiments/point1_compare/run_performance.py','experiments/point1_compare/report_ablation.py','tests/test_point1_derivation.py']),
 ('p1.e7','点一E7训练与明文对照',['experiments/point1_compare/run_training.py','experiments/point1_compare/run_training_controls.py','experiments/point1_compare/training_control_contract.py','experiments/point1_compare/validate_training_controls.py','tests/test_point1_training_controls.py']),
 ('p1.e8','点一E8可信假设审查',['docs/POINT1_FUNCTION_TRUST_COMPARISON_20261018.md','docs/POINT1_BASELINE_IMPLEMENTATION_REVIEW_20261018.md']),
 ('p1.adapters','点一真实实体与声明实例化',['experiments/point1_compare/adapters/hvsa_point1.py','experiments/point1_compare/adapters/gpvo_decided.py','experiments/point1_compare/adapters/gpvo_full.py','experiments/point1_compare/adapters/vlmrs_fullgraph.py','experiments/point1_compare/adapters/vlmrs_verification.py','experiments/point1_compare/configs/baseline_decisions_20261019.json']),
 ('p2.protocol','点二半异步与提交主链',['main_point2.py','commit_tx.py','point2_types.py','tests/test_receipt_flow.py','tests/test_policy_binding.py']),
 ('p2.learning','点二主矩阵与统计',['run_thesis_core.py','fedavg_point2.py','thesis_stats.py','thesis_tables.py','run_point_matrix.py','tests/test_thesis_pipeline.py']),
 ('p2.baselines','点二明文基线',['plain_bench.py','run_plain_bench.py','plain_cache_bench.py','plain_bench_stats.py','tests/test_plain_bench.py','docs/PLAIN_BENCH_ADAPTATIONS.md']),
 ('p2.integrity','点二策略/结果绑定攻击',['run_attack_evidence.py','run_seed_attack_chunked.py','run_attack_v2.py','regress_attack_fix.py','analyze_security.py']),
 ('p2.ablation','点二消融与敏感性',['run_all_ablations.py','run_tier1_batch.py','thesis_ablation_extra.py','analyze_tier1.py','analyze_tier2.py']),
 ('p2.candidates','点二历史候选机制',['run_vpda_stages.py','run_urvcm_stages.py','run_cma_stages.py','run_asig_verify.py','cache_calibration.py','run_cache_screen.py','analyze_cache_screen.py','tests/test_cache_calibration.py']),
 ('p2.overhead','点二时间/通信/内存',['exp_security_overhead.py','exp_comm_per_role.py','exp_memory_overhead.py','examples/exp_n4_phase_breakdown.py']),
 ('p2.privacy','点二隐私辅助证据',['exp_n2_dlg_inversion.py','exp_n7_masked_distribution.py']),
 ('p2.latest_design','点二checkpoint矩校准：仅设计',['docs/POINT2_CHECKPOINT_MOMENT_DESIGN_20261012.md','docs/point2_checkpoint_moment_v0_20261012.json','docs/point2_checkpoint_moment_mathcheck_20261012.json']),
]

def selected_files():
    paths=set()
    for p in ROOT.glob('*.py'):
        if not p.name.startswith('lit_') and p.name not in ['build_search_log.py','build_refs_bib.py']:paths.add(p)
    for pattern in ['tests/*.py','baselines/*.py','examples/*.py','experiments/point1_compare/*.py',
        'experiments/point1_compare/adapters/*.py','experiments/point1_compare/configs/*.json',
        'experiments/fig6_correctness_20261013/*.py','docs/*.md','docs/*.json','docs/archive/*.md',
        'research/**/*.md','research/resources/*.json','tools/research/*.py']:
        for p in ROOT.glob(pattern):
            if p.is_file() and p.name.isascii():paths.add(p)
    for rel in ['README.md','.gitattributes','requirements-review.txt','pytest.ini',
        'research/review.gitignore','experiments/point1_compare/protocol_mapping.md',
        'experiments/fig6_correctness_20261013/README.md']:
        if (ROOT/rel).is_file():paths.add(ROOT/rel)
    return sorted(p.relative_to(ROOT).as_posix() for p in paths
        if p.relative_to(ROOT).as_posix() not in EXCLUDED_LOCAL_CONFIGURATION)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def category(path):
    groups=[g for g,_,files in GROUPS if path in files]
    if groups:return groups
    if path.startswith('experiments/point1_compare/') or path.startswith('tests/test_point1_'):return ['p1.support']
    if path.startswith('docs/POINT1_'):return ['p1.documents']
    if path.startswith('docs/POINT2_') or path.startswith('docs/point2_'):return ['p2.documents']
    if path.startswith(('research/','tools/research/')) or path in ['README.md','.gitattributes','requirements-review.txt','pytest.ini']:return ['review.navigation']
    if path in ['main.py','agent_async.py','agent_async_secure.py','llm_reasoner.py','orchestrate_agent.py','rule_engine.py','run_reuse_perm.py','run_reuse_temp.py','run_reuse_temp2.py','_diag_reuse_direction.py','_verify_p03.py']:return ['archive.exploration']
    if path.startswith(('baselines/','examples/','experiments/fig6_','docs/archive/')):return ['archive.exploration']
    if path.startswith('tests/'):return ['shared.tests']
    if path.startswith('docs/'):return ['shared.documents']
    return ['p2.legacy_or_auxiliary']

def build():
    return dict(schema='hvsa-research-code-review-v1',snapshot_date='2026-10-28',
        layout='Navigation categories reference original source paths; sources were not moved.',
        scope='Code/config/tests/design review; raw evidence, models, datasets and private state excluded.',
        excluded_local_configuration=sorted(EXCLUDED_LOCAL_CONFIGURATION),
        status=dict(full_scale_comparison='last point in progress at snapshot',checkpoint_moment='design_only_not_integrated_or_training_evaluated'),
        generated_publication_files={'.gitignore':'research/review.gitignore'},
        groups=[dict(id=g,title=title,files=files) for g,title,files in GROUPS],
        file_inventory=[dict(path=p,groups=category(p),bytes=(ROOT/p).stat().st_size,sha256=sha(ROOT/p)) for p in selected_files()],
        self_inventory='catalog.json excluded from its own hash list to avoid circularity')

def check(data):
    errors=[];inventory={x['path']:x for x in data['file_inventory']}
    for g in data['groups']:
        for name in g['files']:
            if name not in inventory:errors.append('Missing classified file: '+name)
    python_files=json_files=links=0
    for rel,row in inventory.items():
        path=ROOT/rel
        if not path.is_file():errors.append('Missing file: '+rel);continue
        if sha(path)!=row['sha256']:errors.append('Content differs: '+rel)
        try:
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'),filename=rel);python_files+=1
            if path.suffix=='.json':json.loads(path.read_text(encoding='utf-8-sig'));json_files+=1
        except (SyntaxError,UnicodeError,ValueError) as e:errors.append('Syntax '+rel+': '+str(e))
        if rel=='README.md' or (rel.startswith('research/') and path.suffix=='.md'):
            for value in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
                value=unquote(value.strip('<>')).split('#',1)[0]
                if not value or re.match(r'\w+://|mailto:',value):continue
                target=(path.parent/value).resolve();links+=1
                if target.is_file() and target!=ROOT/TARGET:
                    if ROOT not in target.parents or target.relative_to(ROOT).as_posix() not in inventory:errors.append('Unpublished navigation link '+rel+': '+value)
                elif not target.exists():errors.append('Missing navigation link '+rel+': '+value)
    return dict(valid=not errors,errors=errors,files=len(inventory),python_syntax_checked=python_files,
        json_syntax_checked=json_files,navigation_links_checked=links,
        category_file_counts=dict(Counter(g for row in inventory.values() for g in row['groups'])),
        protocol_or_training_experiments_executed=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--write',action='store_true');m.add_argument('--check',action='store_true')
    args=p.parse_args();target=ROOT/TARGET
    if args.write:
        data=build();target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    else:data=json.loads(target.read_text(encoding='utf-8'))
    result=check(data);print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(int(not result['valid']))

if __name__=='__main__':main()
