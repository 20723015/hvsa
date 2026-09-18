"""安全机制开销微基准 (论文开销表数据源).

实测 (纯 Python ecdsa, NIST P-256):
  - LHH.Hash 逐维标量乘成本, d ∈ {1000,2000,4000,8000} 线性外推到全维 582,026
  - Ed25519 签名/验签 (cryptography, C 实现)
  - PRG 掩码生成 (AES-CTR) 每维成本
口径: median(3 次), 单线程 (OMP=1 不影响纯 python)。与 docs/OVERHEAD_EXPERIMENT.md
的分析法口径一致 —— 全维真 LHH 端到端不可行 (单 Hash ~96s), 功能性实验在
Tiny (d=7850) 上全真进行, 开销按实测线性外推。

用法: python exp_security_overhead.py
"""

import statistics
import time

import numpy as np

from utils import LHH, PRG


def bench(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    print("== LHH.Hash (逐维 EC 标量乘, 纯 Python ecdsa NIST256p) ==")
    xs, ys = [], []
    for d in [1000, 2000, 4000, 8000]:
        pp = LHH.Gen(d)
        v = np.random.randint(-10**6, 10**6, d).astype(np.int64)
        t = bench(lambda: LHH.Hash(pp, v))
        xs.append(d)
        ys.append(t)
        print(f"  d={d:6d}: {t:8.3f}s  ({1e6*t/d:6.1f} us/dim)")
    # 线性拟合 (过原点): t = c · d
    c = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    d_full = 582026
    print(f"  拟合: t = {1e6*c:.1f} us/dim (R^2 见残差: "
          f"{max(abs(y - c*x) / y for x, y in zip(xs, ys)):.1%})")
    print(f"  外推 d={d_full}: 单次 Hash ≈ {c*d_full:.1f}s "
          f"({c*d_full/60:.1f} min)")
    # 端到端口径: 每 checkpoint = 3 ES × verify + ~3 package seal
    for n_hash, label in [(6, "每 checkpoint (3 verify + 3 seal)"),
                          (96, "每 run ~96 seals + 60 verify")]:
        print(f"  {label}: ≈ {n_hash} Hash → {n_hash*c*d_full/60:.1f} min")

    print("\n== Ed25519 (cryptography, C 实现) ==")
    from cryptography.hazmat.primitives.asymmetric import ed25519
    k = ed25519.Ed25519PrivateKey.generate()
    pub = k.public_key()
    msg = b"x" * 64
    t_sign = bench(lambda: k.sign(msg), 20)
    sig = k.sign(msg)
    t_ver = bench(lambda: pub.verify(sig, msg), 20)
    print(f"  sign: {1e6*t_sign:.1f} us   verify: {1e6*t_ver:.1f} us"
          f"  (每 run ~96 seals: {96*1e3*t_sign:.1f} ms)")

    print("\n== PRG 掩码 (HKDF + AES-CTR, 每用户每包 O(d)) ==")
    from utils import PRG
    seed = __import__('os').urandom(32)
    for d in [7850, 582026]:
        t = bench(lambda: PRG.mask_int(seed, "U-0", "edge_0:v0:p0", d), 3)
        print(f"  d={d}: {t*1e3:.1f} ms ({1e6*t/d:.2f} us/dim)")


if __name__ == "__main__":
    main()
