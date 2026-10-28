"""诊断: stale 复用时的方向失配测量 (不改协议, 仅观测).

仓库既有的复用负结果把根因归为"Δ_stale 方向陈旧, 重锚定只修锚点不修方向"
(docs/POINT2_VCM_REUSE_DIAGNOSIS.md §4.5)。但方向失配从未被直接测量。
本脚本用 monkey-patch 包住 User.p2_train_mix_mask_and_upload, 在每次复用发生时
记录:

  E_i            : 陈旧度 (global_version − stale_version)
  cos_stale_new  : cos(Δ_stale, Δ_new)      —— 投影到"新训增量"是否可行
  cos_new_glob   : 基准方向参考 (可选)
  ‖Δ_stale‖/‖Δ_new‖ : 幅度比
  rho_i          : 实际混合权重

用法:
  python _diag_reuse_direction.py --noniid --alpha 0.5 --seed 42 \
      --upload-dropout-rate 0.2 --reuse-rule ours --max-checkpoints 20
  (其余参数与 main_point2 相同; 结果 JSON 写入 scratch_reuse_dir_<tag>.json)
"""

import json
import math
import os
import runpy
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from user import User  # noqa: E402

RECORDS = []
_ORIG = User.p2_train_mix_mask_and_upload


def _patched(self, global_flat, global_version, *a, **kw):
    stale = self._p2_stale_flat
    stale_base = self._p2_stale_base_flat
    stale_ver = self._p2_stale_version
    reuse_rule = kw.get("reuse_rule", "none")
    rho_max = float(kw.get("rho_max", 0.5))
    beta = float(kw.get("staleness_beta", 0.3))

    out = _ORIG(self, global_flat, global_version, *a, **kw)

    try:
        if (reuse_rule != "none" and stale is not None and stale_ver >= 0
                and global_version - stale_ver > 0 and out is not None):
            E = int(global_version - stale_ver)
            new_flat = self._p2_pending_plain_flat
            if new_flat is None:
                return out
            delta_new = np.asarray(new_flat, dtype=np.float64) - np.asarray(global_flat, dtype=np.float64)
            base = stale_base if stale_base is not None else np.asarray(global_flat, dtype=np.float64)
            delta_stale = np.asarray(stale, dtype=np.float64) - np.asarray(base, dtype=np.float64)
            nn, ns = np.linalg.norm(delta_new), np.linalg.norm(delta_stale)
            if nn > 1e-12 and ns > 1e-12:
                cos = float(np.dot(delta_stale, delta_new) / (nn * ns))
            else:
                cos = float("nan")
            # 锚点漂移项: 参数空间混合注入 ρ·(W(r_stale)−W(r_k)), 方向为"倒退"
            anchor = np.asarray(base, dtype=np.float64) - np.asarray(global_flat, dtype=np.float64)
            na = float(np.linalg.norm(anchor))
            cos_anchor_new = (float(np.dot(anchor, delta_new) / (na * nn))
                              if na > 1e-12 and nn > 1e-12 else float("nan"))
            RECORDS.append(dict(
                user=self.id, E=E, reuse_rule=reuse_rule,
                rho=rho_max * math.exp(-beta * E),
                cos_stale_new=cos,
                norm_ratio=float(ns / nn) if nn > 1e-12 else float("nan"),
                anchor_ratio=float(na / nn) if nn > 1e-12 else float("nan"),
                cos_anchor_new=cos_anchor_new,
                stale_ver=int(stale_ver), base_ver=int(global_version),
            ))
    except Exception as e:  # 诊断绝不打断主流程
        RECORDS.append(dict(user=getattr(self, "id", "?"), error=str(e)))
    return out


User.p2_train_mix_mask_and_upload = _patched

try:
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "main_point2.py"), run_name="__main__")
finally:
    tag = ""
    for i, a in enumerate(sys.argv):
        if a == "--tag" and i + 1 < len(sys.argv):
            tag = sys.argv[i + 1]
    out_path = f"scratch_reuse_dir_{tag or 'run'}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(RECORDS, f, ensure_ascii=False, indent=1)
    if RECORDS:
        cos = np.array([r["cos_stale_new"] for r in RECORDS
                        if isinstance(r.get("cos_stale_new"), float)
                        and not math.isnan(r["cos_stale_new"])])
        nr = np.array([r["norm_ratio"] for r in RECORDS
                       if isinstance(r.get("norm_ratio"), float)
                       and not math.isnan(r["norm_ratio"])])
        Es = np.array([r["E"] for r in RECORDS if "E" in r])
        print(f"\n[诊断] 复用事件 n={len(RECORDS)} -> {out_path}")
        if cos.size:
            print(f"  cos(Δ_stale, Δ_new): 均值 {cos.mean():+.3f} "
                  f"中位 {np.median(cos):+.3f} 范围 [{cos.min():+.3f},{cos.max():+.3f}] "
                  f"负比例 {float((cos < 0).mean()):.1%}")
        if nr.size:
            print(f"  ‖Δ_stale‖/‖Δ_new‖: 均值 {nr.mean():.3f} 范围 [{nr.min():.3f},{nr.max():.3f}]")
        if Es.size:
            print(f"  陈旧度 E: 均值 {Es.mean():.2f} 范围 [{Es.min()},{Es.max()}]")
        ar = np.array([r["anchor_ratio"] for r in RECORDS
                       if isinstance(r.get("anchor_ratio"), float)
                       and not math.isnan(r["anchor_ratio"])])
        ca = np.array([r["cos_anchor_new"] for r in RECORDS
                       if isinstance(r.get("cos_anchor_new"), float)
                       and not math.isnan(r["cos_anchor_new"])])
        if ar.size:
            print(f"  锚点漂移/‖Δ_new‖: 均值 {ar.mean():.2f} 中位 {np.median(ar):.2f} "
                  f"范围 [{ar.min():.2f},{ar.max():.2f}]  >1 占比 {float((ar>1).mean()):.0%}")
        if ca.size:
            print(f"  cos(锚点漂移, Δ_new): 均值 {ca.mean():+.3f} 负比例 {float((ca<0).mean()):.0%}")
    else:
        print(f"\n[诊断] 无复用事件 -> {out_path}")
