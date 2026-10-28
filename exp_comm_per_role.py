"""per-role 通信量表 (盲审补全执行计划 阶段 4.3).

不改仿真器: main_point2._CommAccount (main_point2.py:82-129) 已按协议消息
类型记账 (comm_bytes_by_type 落在既有 npz/meta), 本脚本经**静态 type→role
映射**直接聚合出四角色字节表:

  消息类型   发起角色   语义
  upload     用户       用户 → ES 掩码更新 (dim×8B, int64 掩码域)
  package    边缘 ES    ES → Cloud 组包 (掩码和 + H_k + 签名 + 元数据)
  manifest   云端       Cloud 广播候选 (raw sum + M_Q + 元数据)
  vote       边缘 ES    ES → TA 签名投票 (256B × |ES|)
  receipt    TA         TA → Cloud 提交凭证 (320B)
  notify     TA         TA/Cloud → ES commit 通知 (128B × (1+|ES|))

口径注记:
  - manifest 的字节属于 Cloud 广播 (发送方口径), 其载荷含 TA 掩码 M_Q ——
    TA 生成的内容经 Cloud 通道下发, 论文表述为"Cloud 广播 (含 TA 掩码)"。
  - notify 含 1 TA + |ES| 个通知条目, 发起方 TA, 全额记 TA (保守)。
  - 与 exp_security_overhead 的定长元数据结论互补: upload/package/manifest
    随 d 线性, vote/receipt/notify 与 d 无关。

用法:
  python exp_comm_per_role.py                       # 扫描默认 root 集
  python exp_comm_per_role.py --roots results_thesis_core results_thesis_smoke
输出: comm_per_role.json + 控制台表格 (per-run 一行 + 汇总)。
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 静态 type→role 映射 (main_point2._CommAccount.KEYS 顺序)
TYPE_ROLE = {
    'upload': '用户',
    'package': '边缘ES',
    'manifest': '云端',
    'vote': '边缘ES',
    'receipt': 'TA',
    'notify': 'TA',
}
ROLES = ['用户', '边缘ES', '云端', 'TA']


def collect(root):
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(root, '*',
                                                   'point2_meta_*.json'))):
        try:
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        cbt = meta.get('comm_bytes_by_type')
        if not isinstance(cbt, dict):
            continue
        run = os.path.basename(meta_path)[len('point2_meta_'):-len('.json')]
        rows.append({'root': os.path.basename(root), 'run': run,
                     'dataset': meta.get('dataset'),
                     'seed': meta.get('seed'),
                     'verify_total': meta.get('verify_total'),
                     'by_type': cbt})
    return rows


def role_rollup(by_type):
    out = {r: 0 for r in ROLES}
    for t, b in by_type.items():
        out[TYPE_ROLE.get(t, '云端')] += int(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+',
                    default=['results_thesis_core', 'results_thesis_smoke'])
    ap.add_argument('--out', default='comm_per_role.json')
    args = ap.parse_args()

    all_rows = []
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"[skip] root 不存在: {root}")
            continue
        rows = collect(root)
        print(f"[collect] {root}: {len(rows)} runs")
        all_rows.extend(rows)

    if not all_rows:
        print("无可用 comm_bytes_by_type 记录")
        return 1

    # per-run 四角色表
    print()
    print(f"{'run':<44} {'用户':>12} {'边缘ES':>12} {'云端':>12} "
          f"{'TA':>10} {'合计(KiB)':>10}")
    totals = {r: 0 for r in ROLES}
    n = 0
    for row in all_rows:
        rr = role_rollup(row['by_type'])
        for r in ROLES:
            totals[r] += rr[r]
        n += 1
        total_b = sum(rr.values())
        if n <= 12 or n % 50 == 0:
            print(f"{row['run'][:44]:<44} {rr['用户']:>12,} "
                  f"{rr['边缘ES']:>12,} {rr['云端']:>12,} {rr['TA']:>10,} "
                  f"{total_b / 1024:>10.1f}")
    print('-' * 100)
    print(f"{'TOTAL (%d runs)':<44} {totals['用户']:>12,} "
          f"{totals['边缘ES']:>12,} {totals['云端']:>12,} {totals['TA']:>10,} "
          f"{sum(totals.values()) / 1024:>10.1f}")

    report = {
        'generated_at': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'type_role_map': TYPE_ROLE,
        'n_runs': len(all_rows),
        'totals_bytes': totals,
        'totals_bytes_by_type': {
            t: sum(int(r['by_type'].get(t, 0)) for r in all_rows)
            for t in TYPE_ROLE},
        'runs': [{'root': r['root'], 'run': r['run'],
                  'roles': role_rollup(r['by_type']),
                  'by_type': r['by_type']} for r in all_rows],
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
