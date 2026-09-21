"""N7 实验: 掩码上传向量的分布一致性 (论文 3.5.5).

在真实 run 中捕获每个 (user, package_token) 的:
  - 掩码 mask  (PRG.mask_int 的输出)
  - 上传向量 w̄ (User.p2_train_mix_mask_and_upload 的返回值)
明文 w = w̄ − mask。再比较三者分布: 掩码/上传应近似均匀, 明文应高度集中。

判据 (docs/THESIS_FULL_DESIGN.md 3.5.5): 掩码后向量服从均匀分布、与明文分布显著不同。
指标: 各维度直方图 + KL 散度 (bits)。

捕获方式为运行时 monkeypatch, 不改动任何协议代码, 因此不影响已冻结批次的指纹。

用法:
  python exp_n7_masked_distribution.py                     # 默认 MNIST, seed 42
  python exp_n7_masked_distribution.py --dataset CIFAR10 --bins 80
"""

import argparse
import os
import runpy
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MNIST",
                    choices=["MNIST", "FashionMNIST", "CIFAR10"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--max-checkpoints", type=int, default=3)
    ap.add_argument("--max-ticks", type=int, default=40)
    ap.add_argument("--out-dir", default="results_n7_diag")
    args = ap.parse_args()

    os.chdir(ROOT)
    import utils
    import user as user_mod

    masks, masked = {}, {}
    _orig_mask = utils.PRG.mask_int

    def patched_mask(secret_seed, user_id, round_token, dim):
        m = _orig_mask(secret_seed, user_id, round_token, dim)
        masks[(user_id, str(round_token))] = np.asarray(m).copy()
        return m

    utils.PRG.mask_int = patched_mask

    _orig_up = user_mod.User.p2_train_mix_mask_and_upload

    def patched_up(self, global_flat, global_version, package_token, *a, **kw):
        r = _orig_up(self, global_flat, global_version, package_token, *a, **kw)
        if r is not None:
            masked[(self.id, str(package_token))] = np.asarray(r).copy()
        return r

    user_mod.User.p2_train_mix_mask_and_upload = patched_up

    sys.argv = [
        "main_point2.py", "--dataset", args.dataset, "--seed", str(args.seed),
        "--deterministic", "-K", "3", "--s-max", "3", "--reuse-rule", "none",
        "--max-checkpoints", str(args.max_checkpoints),
        "--max-ticks", str(args.max_ticks),
        "--output-dir", args.out_dir, "--run-id", "n7_diag", "--force",
    ]
    try:
        runpy.run_path("main_point2.py", run_name="__main__")
    except SystemExit:
        pass

    def kl_vs_uniform(x, lo, hi, bins):
        """KL(P || Uniform), bits; 0 表示与均匀分布不可区分."""
        h, _ = np.histogram(x, bins=bins, range=(lo, hi))
        p = h.astype(float)
        if p.sum() == 0:
            return float("nan")
        p /= p.sum()
        u = 1.0 / bins
        m = p > 0
        return float(np.sum(p[m] * np.log2(p[m] / u)))

    keys = sorted(set(masks) & set(masked))
    print()
    print("=" * 78)
    print("N7: 掩码上传向量的分布一致性")
    print("=" * 78)
    print("捕获 %d 个掩码 / %d 个上传向量; 可配对 %d"
          % (len(masks), len(masked), len(keys)))
    if not keys:
        print("!! 未捕获到配对样本")
        return 1

    plain, mk, msk = [], [], []
    for k in keys:
        a, b = masks[k], masked[k]
        if a.shape != b.shape:
            continue
        plain.append(b - a)
        mk.append(a)
        msk.append(b)
    if not plain:
        print("!! 形状全部不匹配")
        return 1
    plain = np.concatenate(plain)
    mk = np.concatenate(mk)
    msk = np.concatenate(msk)

    lo, hi = float(utils.PRG.MASK_LOW), float(utils.PRG.MASK_HIGH)
    print("掩码取值域: [%g, %g), 宽度 %g" % (lo, hi, hi - lo))
    print("样本数: %d 维 (d=%d × %d 次上传)"
          % (plain.size, plain.size // len(keys), len(keys)))
    print()

    bins = args.bins
    rows = []
    for nm, x in [("明文 w", plain), ("掩码 mask", mk), ("上传 w̄", msk)]:
        kl = kl_vs_uniform(x, lo, hi, bins)
        h, _ = np.histogram(x, bins=bins, range=(lo, hi))
        p = h / h.sum()
        dev = float(np.abs(p - 1.0 / bins).max())
        p05, p95 = np.percentile(x, [5, 95])
        rows.append((nm, x.size, x.mean(), x.std(), p05, p95, kl, dev))
        print("  %-10s mean=%12.4g std=%12.4g  5%%=%12.4g 95%%=%12.4g"
              % (nm, x.mean(), x.std(), p05, p95))
        print("             KL vs uniform = %8.4f bits   逐 bin 最大偏差 = %.5f"
              % (kl, dev))
    print()
    print("  (均匀分布理论 bin 概率 = %.5f)" % (1.0 / bins))
    print()
    print("尺度比 std(mask)/std(w) = %.1f 倍" % (mk.std() / plain.std()))
    print()
    kl_w = rows[0][6]
    kl_masked = rows[2][6]
    ok = (kl_masked < 0.01) and (kl_w > 1.0)
    print("判据: KL(w̄) ≈ 0 且 KL(w) 显著 > 0  →  %s" % ("成立" if ok else "不成立"))
    print("  KL(w) = %.4f bits   KL(w̄) = %.4f bits" % (kl_w, kl_masked))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
