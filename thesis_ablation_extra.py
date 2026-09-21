"""论文消融补充分析: 三口径主表 + GFB 等计算量 + VCM 模式诊断 + s_max=0 参照 + I1 验证.

只读 results_* 目录, 输出 markdown 到 results_thesis_core/stats/。
不改动任何 CODE_FILES, 不影响已冻结批次的指纹。
"""
import json
import os
import statistics as st

import numpy as np

ROOT = r"E:\hvsa"
T95 = 2.776
OUT = os.path.join(ROOT, "results_thesis_core", "stats", "thesis_ablation_extra.md")
DATASETS = ["MNIST", "FashionMNIST", "CIFAR10"]
PARTS = ["IID", "NI"]
SEEDS = [42, 43, 44, 45, 46]


def load(root):
    p = os.path.join(ROOT, root, "thesis_core_manifest.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def get(recs, ds, part, arm, seed):
    for r in recs:
        if (r.get("arm") == arm and r.get("dataset") == ds
                and r.get("part") == part and r.get("seed") == seed):
            return r
    return None


def npz(root, tag):
    p = os.path.join(ROOT, root, tag, "point2_data_%s.npz" % tag)
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True)


def curve(root, tag):
    d = npz(root, tag)
    if d is None:
        return None, None
    a = np.asarray(d.get("acc_history"))
    a = (a[:, 1] if a.ndim == 2 else a).astype(float)
    t = d.get("checkpoint_ticks")
    t = (np.asarray(t, dtype=np.int64) if t is not None and len(t) == len(a)
         else np.arange(1, len(a) + 1))
    return a, t


def cum_compute(root, tag):
    d = npz(root, tag)
    if d is None or "checkpoint_Ss" not in d.files:
        return None
    s = np.asarray(d["checkpoint_Ss"], dtype=float).ravel()
    return np.cumsum(s) if s.size else None


def stats_of(diffs):
    m = st.mean(diffs)
    sd = st.stdev(diffs)
    ci = T95 * sd / len(diffs) ** 0.5
    pos = sum(1 for x in diffs if x > 0)
    verdict = ("显著提升" if m - ci > 0 else
               "显著下降" if m + ci < 0 else "不显著")
    return m, sd, ci, pos, len(diffs), verdict


def row(cells):
    return "| " + " | ".join(cells) + " |"


MAIN = load("results_thesis_core")
L = ["# 论文消融补充分析 (三口径主表 / GFB / VCM 机制)", ""]

# ---------- 1. 主表三口径 ----------
L += ["## 1. 主表三口径 (G3v = 主方法)", "",
      "计算量以 `checkpoint_Ss` 累计和 ΣS_Q 度量 (逐 checkpoint 样本贡献)。",
      "注意: G2 为 K=5、G3 家族为 K=3，同为 20 个 checkpoint 时 G2 的计算量约为 G3 的 1.66 倍，",
      "因此“等轮次”不等于“等计算量”。", ""]

arms = ["G2", "G3", "G3v"]
by = {}
for ds in DATASETS:
    for part in PARTS:
        for seed in SEEDS:
            cell = {}
            for a in arms:
                r = get(MAIN, ds, part, a, seed)
                if not r:
                    continue
                c, t = curve("results_thesis_core", r["tag"])
                if c is not None:
                    cell[a] = (c, t, r)
            if len(cell) == len(arms):
                by[(ds, part, seed)] = cell

L += ["### 1.1 等轮次 (各臂 20 个 checkpoint, 取末次精度)", "",
      row(["数据集", "分布"] + arms), row(["---"] * (2 + len(arms)))]
for ds in DATASETS:
    for part in PARTS:
        cells = [ds, part]
        for a in arms:
            v = [by[k][a][2]["final_acc"] for k in by
                 if k[0] == ds and k[1] == part]
            cells.append("%.2f ± %.2f" % (st.mean(v), st.stdev(v)) if len(v) > 1 else "—")
        L.append(row(cells))

L += ["", "### 1.2 等计算量 (目标 = 两臂 ΣS_Q 的较小者)", "",
      row(["比较", "数据集", "分布", "Δmean", "95% CI", "正种子", "判定"]),
      row(["---"] * 7)]
for lbl, (A, B) in [("半异步① G3−G2", ("G3", "G2")),
                    ("VCM③ G3v−G3", ("G3v", "G3")),
                    ("主方法 G3v−G2", ("G3v", "G2"))]:
    for ds in DATASETS:
        for part in PARTS:
            d = []
            for k, cell in by.items():
                if k[0] != ds or k[1] != part:
                    continue
                ca = cum_compute("results_thesis_core", cell[A][2]["tag"])
                cb = cum_compute("results_thesis_core", cell[B][2]["tag"])
                if ca is None or cb is None:
                    continue
                tgt = min(ca[-1], cb[-1])
                ma, mb = ca <= tgt, cb <= tgt
                if ma.any() and mb.any():
                    d.append(cell[A][0][ma][-1] - cell[B][0][mb][-1])
            if len(d) > 1:
                m, sd, ci, pos, n, v = stats_of(d)
                L.append(row([lbl, ds, part, "%+.2f" % m,
                              "[%+.2f, %+.2f]" % (m - ci, m + ci),
                              "%d/%d" % (pos, n), v]))

L += ["", "### 1.3 等时间 (预算 T = 同 seed 的 G3 末次 tick, 基准 G2)", "",
      row(["数据集", "分布", "臂", "臂@T", "G2@T", "Δmean", "95% CI", "正种子", "判定"]),
      row(["---"] * 9)]
for ds in DATASETS:
    for part in PARTS:
        for a in ["G3", "G3v"]:
            d, aa, gg = [], [], []
            for k, cell in by.items():
                if k[0] != ds or k[1] != part:
                    continue
                T = cell["G3"][1][-1]
                x = cell[a][0][cell[a][1] <= T]
                y = cell["G2"][0][cell["G2"][1] <= T]
                if x.size and y.size:
                    d.append(x[-1] - y[-1])
                    aa.append(x[-1])
                    gg.append(y[-1])
            if len(d) > 1:
                m, sd, ci, pos, n, v = stats_of(d)
                L.append(row([ds, part, a, "%.2f" % st.mean(aa), "%.2f" % st.mean(gg),
                              "%+.2f" % m, "[%+.2f, %+.2f]" % (m - ci, m + ci),
                              "%d/%d" % (pos, n), v]))

# ---------- 2. GFB 等计算量 ----------
GFB = load("results_gfb_eqcompute")
L += ["", "## 2. GFB (K=1) 阈值消融: 等计算量对照", "",
      "GFB 在 20 个 checkpoint 时仅采纳 73 次用户更新 (G3 为 218)，计算量不足；",
      "补跑 `--max-checkpoints 60` 对齐到 218 次。", "",
      row(["数据集", "分布", "GFB(60ckpt)", "G3", "配对差", "95% CI",
           "GFB tick", "G3 tick", "判定"]),
      row(["---"] * 9)]
for ds in ["FashionMNIST", "CIFAR10"]:
    for part in PARTS:
        d, ga, g3a, gt, g3t = [], [], [], [], []
        for seed in SEEDS:
            x = get(GFB, ds, part, "GFB", seed)
            y = get(MAIN, ds, part, "G3", seed)
            if not x or not y:
                continue
            d.append(x["final_acc"] - y["final_acc"])
            ga.append(x["final_acc"])
            g3a.append(y["final_acc"])
            gt.append(x["final_tick"])
            g3t.append(y["final_tick"])
        if len(d) > 1:
            m, sd, ci, pos, n, v = stats_of(d)
            L.append(row([ds, part,
                          "%.2f ± %.2f" % (st.mean(ga), st.stdev(ga)),
                          "%.2f ± %.2f" % (st.mean(g3a), st.stdev(g3a)),
                          "%+.2f" % m, "[%+.2f, %+.2f]" % (m - ci, m + ci),
                          "%.0f" % st.mean(gt), "%.0f" % st.mean(g3t), v]))

# ---------- 3. VCM 模式诊断 ----------
L += ["", "## 3. VCM 模式诊断 (NI 单元, 相对基线 G3v)", "",
      "`span`/`clip` 为结构性空操作 (陈旧度恒为 1；补偿范数仅占 raw sum 的 0.58%，",
      "远低于 clip 的 10% 阈值)，已由独立诊断脚本证实，故不列表。",
      "`moment*`/`curve` 不可验证 (ES 无法独立复算，属 verify-off 研究模式)，",
      "仅作机制诊断，不作为主方法。", "",
      row(["数据集", "模式", "acc", "std", "Δ vs G3v", "95% CI", "正种子"]),
      row(["---"] * 7)]
for ds in ["MNIST", "FashionMNIST"]:
    b = [get(MAIN, ds, "NI", "G3v", s) for s in SEEDS]
    b = [x for x in b if x]
    L.append(row([ds, "G3v (基线)",
                  "%.2f" % st.mean([x["final_acc"] for x in b]),
                  "%.2f" % st.stdev([x["final_acc"] for x in b]),
                  "—", "—", "—"]))
    for mode in ["moment1", "moment2", "curve"]:
        recs = load("results_vcm_" + mode)
        v = [get(recs, ds, "NI", "G3v", s) for s in SEEDS]
        v = [x for x in v if x]
        if len(v) < 2:
            continue
        d = [x["final_acc"] - y["final_acc"] for x, y in zip(v, b)]
        m, sd, ci, pos, n, _ = stats_of(d)
        L.append(row([ds, mode,
                      "%.2f" % st.mean([x["final_acc"] for x in v]),
                      "%.2f" % st.stdev([x["final_acc"] for x in v]),
                      "%+.2f" % m, "[%+.2f, %+.2f]" % (m - ci, m + ci),
                      "%d/%d" % (pos, n)]))

# ---------- 4. s_max=0 参照 ----------
Z = load("results_smax0")
L += ["", "## 4. 无陈旧参照 (s_max=0) 与 VCM 追回率", "",
      "四个配置计算量完全相同 (224 次采纳用户更新 / 60 组包)，",
      "差别仅在是否容忍陈旧与是否补偿。", "",
      row(["数据集", "分布", "s_max=0 无补偿", "s_max=0 +VCM",
           "s_max=3 无补偿", "s_max=3 +VCM", "tick(0→3)"]),
      row(["---"] * 7)]
rec = {}
for ds in ["MNIST", "FashionMNIST"]:
    for part in PARTS:
        c = {}
        for lbl, src, arm in [("z3", Z, "G3"), ("z3v", Z, "G3v"),
                              ("n3", MAIN, "G3"), ("n3v", MAIN, "G3v")]:
            v = [get(src, ds, part, arm, s) for s in SEEDS]
            c[lbl] = [x for x in v if x]
        if not all(c.values()):
            continue
        m = dict((k, st.mean([x["final_acc"] for x in v])) for k, v in c.items())
        t0 = st.mean([x["final_tick"] for x in c["z3"]])
        t3 = st.mean([x["final_tick"] for x in c["n3"]])
        L.append(row([ds, part, "%.2f" % m["z3"], "%.2f" % m["z3v"],
                      "%.2f" % m["n3"], "%.2f" % m["n3v"],
                      "%.0f → %.0f" % (t0, t3)]))
        rec[(ds, part)] = (m, t0, t3)

L += ["", "### 4.1 VCM 追回率 (相对无陈旧参照)", "",
      row(["数据集", "分布", "容忍陈旧的损失", "VCM 追回", "追回比例", "tick 节省"]),
      row(["---"] * 6)]
for (ds, part), (m, t0, t3) in rec.items():
    loss = m["z3"] - m["n3"]
    gain = m["n3v"] - m["n3"]
    pct = "**%.0f%%**" % (gain / loss * 100) if loss > 0 else "—"
    L.append(row([ds, part, "%+.2f" % loss, "%+.2f" % gain, pct,
                  "%.0f%%" % ((1 - t3 / t0) * 100)]))

# ---------- 5. I1 验证 ----------
L += ["", "## 5. I1 验证 (无陈旧时 VCM 恒为 0)", "",
      "判据: `s_max=0` 时 G3v 与 G3 的 acc 曲线应逐值相同。", "",
      row(["数据集", "分布", "逐值相同", "max|Δacc|"]), row(["---"] * 4)]
for ds in ["MNIST", "FashionMNIST"]:
    for part in PARTS:
        ok, mx = [], []
        for seed in SEEDS:
            a, _ = curve("results_smax0", "G3_%s_%s_s%d" % (ds, part, seed))
            b, _ = curve("results_smax0", "G3v_%s_%s_s%d" % (ds, part, seed))
            if a is None or b is None:
                continue
            ok.append(bool(np.array_equal(a, b)))
            mx.append(float(np.max(np.abs(a - b))))
        if ok:
            L.append(row([ds, part, "%s (%d/%d)" % (all(ok), sum(ok), len(ok)),
                          "%.3e" % max(mx)]))

L += ["", "---", "",
      "生成脚本: `thesis_ablation_extra.py` (只读 results_*，不改动 CODE_FILES)", ""]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("报告已保存:", OUT)
