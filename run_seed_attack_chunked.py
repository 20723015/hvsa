"""按攻击类分块跑 seed 46 并合并报告.

动机: 这台机器会周期性被 Windows Update/睡眠打断, 而 run_attack_evidence.py
无 seed 内续跑能力, 一个长成员攻击 (replay/clean 各约 2500s) 被杀就损失整个
seed (~2h)。分块后单次损失上限降到一个类 (~40min)。

用法: python run_seed_attack_chunked.py <seed>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = r"E:\hvsa"
OUT = os.path.join(ROOT, "results_thesis_core_attack")
CLASSES = ["raw_sum", "manifest_replay", "weight_source", "comp_param", "clean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--units", type=int, default=100)
    args = ap.parse_args()
    seed = args.seed
    os.chdir(ROOT)

    parts = []
    for cls in CLASSES:
        log = os.path.join(ROOT, "attack_s%d_%s.log" % (seed, cls))
        print("[%s] === class %s start ===" % (
            datetime.now().strftime("%H:%M:%S"), cls), flush=True)
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.call(
                [sys.executable, "run_attack_evidence.py", "--units",
                 str(args.units), "--seed", str(seed), "--classes", cls],
                stdout=f, stderr=subprocess.STDOUT)
        src = os.path.join(OUT, "attack_evidence.json")
        if not os.path.exists(src):
            print("  class %s: rc=%s 但无 JSON, 跳过" % (cls, rc), flush=True)
            continue
        with open(src, encoding="utf-8") as f:
            j = json.load(f)
        shutil.copy(src, os.path.join(OUT, "_part_s%d_%s.json" % (seed, cls)))
        md = os.path.join(OUT, "attack_evidence.md")
        if os.path.exists(md):
            shutil.copy(md, os.path.join(OUT, "_part_s%d_%s.md" % (seed, cls)))
        n_cls = len(j.get("classes", []))
        n_att = sum(c.get("attempted", 0) for c in j.get("classes", []))
        print("  class %s: rc=%s classes=%d attempted=%d"
              % (cls, rc, n_cls, n_att), flush=True)
        parts.append(j)

    if not parts:
        print("没有任何可用分块, 退出", flush=True)
        return 1

    merged = {
        "classes": [],
        "clean": None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": parts[0].get("args"),
        "merged_from_classes": CLASSES,
    }
    for p in parts:
        merged["classes"] += p.get("classes", [])
        if p.get("clean"):
            merged["clean"] = p["clean"]

    with open(os.path.join(OUT, "attack_evidence_s%d.json" % seed), "w",
              encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print("合并 JSON: attack_evidence_s%d.json (classes=%d)"
          % (seed, len(merged["classes"])), flush=True)

    # 用 runner 自己的渲染器出 markdown, 保证口径一致
    sys.path.insert(0, ROOT)
    import run_attack_evidence as rae
    ns = argparse.Namespace(**parts[0]["args"])
    md = rae.render_md(merged, ns)
    path = os.path.join(OUT, "attack_evidence_s%d.md" % seed)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print("合并 MD:", path, flush=True)
    verdict = [l for l in md.splitlines() if "达标" in l]
    print("判定:", verdict[0].strip() if verdict else "?", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
