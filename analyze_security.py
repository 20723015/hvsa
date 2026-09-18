"""安全线结果分析: 透明性 / 验证开销 / 攻击检出 / 纵深防御.

数据源: 带标签日志 (TINY/VF/TGv/TMv/DRv/RPlv/TGn/TMn/DRn/RPn)。
- 透明性: VF (verify on) 与 TINY (verify off) 的 acc history 应逐值一致
  (验证不改变聚合数学, 不消耗 RNG)。
- 开销: 同 seed 配对 wall(VF) − wall(TINY) → 端到端验证开销 (d=7850, 纯 Python)。
- 检出: verify on 的四类攻击 det%; verify off 的协议层守卫 (commit guard /
  exactly-once) 拒绝计数 → 纵深防御; TGn/DRn 的 acc 损害 = 无验证后果。

用法: python analyze_security.py
"""

import glob
import re
import statistics as st

SEEDS = [12, 42, 43, 44, 45, 46, 47, 48]


def find_log(prefix):
    hits = sorted(glob.glob(f"point2_log_{prefix}_MNIST_*.txt"))
    return hits[-1] if hits else None


def parse_log(path):
    if path is None:
        return {"path": None, "acc": None, "ckps": 0, "ticks": 0, "wall": 0.0,
                "hist": [], "vp": 0, "vt": 0, "det": None,
                "guard_rej": 0, "verify_rej": 0, "replay_rej": 0, "replay_acc": 0}
    try:
        txt = open(path, encoding="utf-8", errors="strict").read()
    except UnicodeDecodeError:
        txt = open(path, encoding="gbk", errors="replace").read()
    out = {"path": path}
    m = re.search(r"最终准确率: ([0-9.]+)%", txt)
    out["acc"] = float(m.group(1)) if m else None
    m = re.search(r"模拟完成: (\d+) checkpoints, (\d+) ticks, 墙钟 ([0-9.]+)s", txt)
    out["ckps"], out["ticks"], out["wall"] = (
        int(m.group(1)), int(m.group(2)), float(m.group(3))) if m else (0, 0, 0.0)
    hist = re.search(r"Accuracy history: (\[.*?\])", txt, re.S)
    out["hist"] = [b for _, b in eval(hist.group(1))] if hist else []
    m = re.search(r"验证通过: (\d+)/(\d+)", txt)
    out["vp"], out["vt"] = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    m = re.search(r"Attack detection rate: ([0-9.]+)%", txt)
    out["det"] = float(m.group(1)) if m else None
    out["guard_rej"] = len(re.findall(r"rejected by commit guard", txt))
    out["verify_rej"] = len(re.findall(r"checkpoint rejected; global version", txt))
    out["replay_rej"] = len(re.findall(r"rejected \(exactly-once\)", txt))
    out["replay_acc"] = len(re.findall(r"ACCEPTED \(BUG\)", txt))
    return out


def main():
    tiny = {s: parse_log(find_log(f"TINY_s{s}")) for s in SEEDS}
    vf = {s: parse_log(find_log(f"VF_s{s}")) for s in SEEDS}

    # --- 透明性 ---
    print("== 1. 验证透明性 (verify on/off 结果应逐值一致) ==")
    ok = all(vf[s]["hist"] == tiny[s]["hist"] and vf[s]["hist"] for s in SEEDS)
    print(f"  8 seeds acc history 逐值一致: {'OK' if ok else 'FAIL!'}")
    print(f"  TINY acc: {st.mean(t for t in (tiny[s]['acc'] for s in SEEDS) if t):.2f}  "
          f"VF acc: {st.mean(v['acc'] for v in (vf[s] for s in SEEDS)):.2f}  "
          f"验证通过: {vf[SEEDS[0]]['vp']}/{vf[SEEDS[0]]['vt']}")

    # --- 开销 ---
    print("\n== 2. 端到端验证开销 (同 seed 配对, d=7850, 纯 Python ecdsa) ==")
    diffs = [vf[s]["wall"] - tiny[s]["wall"] for s in SEEDS]
    for s in SEEDS:
        print(f"  s{s}: {tiny[s]['wall']:6.1f}s -> {vf[s]['wall']:7.1f}s  "
              f"(+{vf[s]['wall']-tiny[s]['wall']:6.1f}s)")
    per_ckp = st.mean(diffs) / 20.0
    base_wall = st.mean([tiny[s]["wall"] for s in SEEDS])
    print(f"  mean overhead: {st.mean(diffs):.1f}s/run  = {per_ckp:.1f}s/checkpoint"
          f"  (占比 {st.mean(diffs)/base_wall*100:.0f}%)")

    # --- 攻击检出 (verify on) ---
    print("\n== 3. 攻击检出 (verify on, LHH+Ed25519, σ/VCM 激活) ==")
    print(f"  {'攻击':6s} {'检出率':>8s} {'verify尝试':>10s} {'提交ckp':>8s} {'ticks':>6s} {'LHH拒绝':>8s}")
    for pre in ["TGv", "TMv", "DRv", "RPv"]:
        dets, vps, ckps, ticks, rej = [], [], [], [], []
        for s in SEEDS:
            r = parse_log(find_log(f"{pre}_s{s}"))
            if r["det"] is not None:
                dets.append(r["det"])
            vps.append(r["vt"])
            ckps.append(r["ckps"])
            ticks.append(r["ticks"])
            rej.append(r["verify_rej"] + r["guard_rej"])
        det_s = f"{st.mean(dets):.0f}%" if dets else "n/a"
        print(f"  {pre:6s} {det_s:>8s} {st.mean(vps):>10.1f} {st.mean(ckps):>8.1f} "
              f"{st.mean(ticks):>6.0f} {st.mean(rej):>8.1f}")

    # --- 纵深防御 (verify off) ---
    print("\n== 4. 纵深防御 (verify off: 协议层守卫) + 无验证损害 ==")
    for pre in ["TGn", "TMn", "DRn", "RPn"]:
        rows = [parse_log(find_log(f"{pre}_s{s}")) for s in (12, 42, 43)]
        accs = [r["acc"] for r in rows if r["acc"] is not None]
        gr = st.mean([r["guard_rej"] for r in rows])
        vr = st.mean([r["verify_rej"] for r in rows])
        rr = st.mean([r["replay_rej"] for r in rows])
        ra = st.mean([r["replay_acc"] for r in rows])
        base = st.mean([tiny[s]["acc"] for s in (12, 42, 43) if tiny[s]["acc"]])
        dmg = f" acc={st.mean(accs):.2f} (vs TINY {base:.2f})" if accs else " (无提交, fail-closed)"
        print(f"  {pre:6s}:{dmg}  guard拒绝={gr:.1f} LHH拒绝={vr:.1f} "
              f"重放拒绝={rr:.1f} 重放接受={ra:.1f}")


if __name__ == "__main__":
    main()
