"""研究点2 实验结果注册器.

扫描目录下的 point2 运行产物 (NPZ 数据 + LOG 日志), 提取关键指标,
追加到持久化注册表 point2_results_registry.json (append-only, 按内容去重),
并重渲染人类可读视图 point2_results_registry.md.

用途: 实验自己跑, 每次跑完执行一次即可累积到注册表——会话中断数据不丢.

用法:
  python collect_point2_results.py                          # 扫当前目录
  python collect_point2_results.py --dir results_MNIST_xxx  # 扫指定目录
  python collect_point2_results.py --dir A --dir B          # 扫多个目录
  python collect_point2_results.py --show                   # 只打印当前注册表
  python collect_point2_results.py --reset                  # 清空注册表 (慎用)
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from datetime import datetime

REGISTRY_JSON = "point2_results_registry.json"
REGISTRY_MD = "point2_results_registry.md"

MD_HEADER = """# 研究点2 实验结果注册表 (point2)

> 由 `collect_point2_results.py` 自动维护。每条实验 import 一次, 按 NPZ 内容去重。
> 字段: acc=最终准确率% wall=墙钟秒 await=平均等待(ticks) mwait=最大等待
> stale=Stale reuse 总人次 exp=组包过期移除 vp/vt=验证通过/总次 det=攻击检出率
"""


def npz_id(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:16]


def read_registry() -> list:
    if not os.path.exists(REGISTRY_JSON):
        return []
    with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def write_registry(rows: list):
    with open(REGISTRY_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _parse_namespace_str(s: str) -> dict:
    """解析 main_point2 保存的 config 字符串 ``Namespace(k=v, ...)``."""
    import ast
    import re
    s = s.strip()
    if s.startswith("Namespace(") and s.endswith(")"):
        s = s[len("Namespace("):-1]
    out = {}
    for m in re.finditer(r"(\w+)=([^,]+)(?:,|$)", s):
        k, v = m.group(1), m.group(2).strip()
        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            out[k] = v
    return out


def parse_npz(path: str):
    import numpy as np
    d = np.load(path, allow_pickle=True)

    def g(key, default=None):
        v = d.get(key)
        return default if v is None else v

    config = g("config")
    if config is not None:
        if isinstance(config, np.ndarray) and config.ndim == 0:
            config = config.item()  # np.savez 存 Namespace 时是 0-d object array
        if isinstance(config, str):
            config = _parse_namespace_str(config)
        elif hasattr(config, "__dict__"):
            config = vars(config)
        elif not isinstance(config, dict):
            config = {}

    acc_hist = g("acc_history")
    if acc_hist is not None and hasattr(acc_hist, "shape") and len(acc_hist.shape) > 0:
        acc_hist = acc_hist.tolist()

    wait_times = g("wait_times")
    max_wait = g("max_wait")
    if max_wait is None and wait_times is not None and len(wait_times) > 0:
        max_wait = float(np.max(wait_times))

    c = config or {}

    def _acc(v):
        # npz 中未启用/非对应模式时存 nan, 统一规整为 None
        if v is None:
            return None
        f = float(v)
        return None if math.isnan(f) else round(f, 2)

    return {
        "imported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "npz": os.path.basename(path),
        "final_acc": round(float(g("final_acc", 0) or 0), 2),
        "final_acc_ema": _acc(g("final_acc_ema")),
        "final_acc_cma": _acc(g("final_acc_cma")),
        "wall_clock_s": round(float(g("wall_clock", 0) or 0), 1),
        "checkpoints": len(acc_hist) if acc_hist else 0,
        "avg_wait": round(float(g("avg_wait", 0) or 0), 2),
        "max_wait": round(float(max_wait or 0), 2),
        "comp_gain": round(float(g("comp_gain", 0) or 0), 4),
        "total_stale_reuses": int(g("total_stale_reuses", 0) or 0),
        "expired_buffer": int(g("expired_buffer_count", 0) or 0),
        "fairness_var": round(float(g("fairness_var", 0) or 0), 4),
        "verify_pass": int(g("verify_pass_count", 0) or 0),
        "verify_total": int(g("verify_total_count", 0) or 0),
        "attack_detected": round(float(g("attack_detected", 0) or 0), 3),
        "dataset": c.get("dataset", ""),
        "seed": c.get("seed", ""),
        "config_summary": {
            "edges": c.get("num_edge_servers", ""),
            "users_per_edge": c.get("users_per_edge", ""),
            "K": c.get("K", ""),
            "s_max": c.get("s_max", ""),
            "reuse": c.get("reuse_rule", ""),
            "rho_max": c.get("rho_max", ""),
            "beta_c": c.get("staleness_beta", ""),
            "vcm": c.get("vcm", ""),
            "gamma_vcm": c.get("gamma_vcm", ""),
            "attack": c.get("attack", ""),
            "verify": c.get("verify", ""),
            "sigma": c.get("staleness_sigma", ""),
            "staleness_metric": c.get("staleness_metric", ""),
            "noniid": c.get("noniid", ""),
            "alpha": c.get("alpha", ""),
            "fedprox_mu": c.get("fedprox_mu", ""),
            "ema_decay": c.get("ema_decay", ""),
            "cma_mode": c.get("cma_mode", ""),
            "cma_beta": c.get("cma_beta", ""),
            "cma_beta_min": c.get("cma_beta_min", ""),
            "cma_beta_max": c.get("cma_beta_max", ""),
            "cma_lambda_stale": c.get("cma_lambda_stale", ""),
            "momentum": c.get("momentum", ""),
        },
    }


def render_md(rows: list) -> str:
    lines = [MD_HEADER, ""]
    if not rows:
        lines.append("(空)")
        return "\n".join(lines)
    lines.append(f"共 {len(rows)} 条实验 (最近先显示):")
    lines.append("")
    lines.append("| # | NPZ | 数据集 | acc% | wall秒 | ckp | await | mwait | stale | exp | vp/vt | det% | 配置摘要 |")
    lines.append("|---|-----|--------|------|--------|-----|-------|-------|-------|-----|-------|------|----------|")
    for i, r in enumerate(reversed(rows), 1):
        cs = r["config_summary"]
        cfg = (f"E{cs['edges']}xU{cs['users_per_edge']} K{cs['K']} s{cs['s_max']} "
               f"reuse={cs['reuse']} rho={cs['rho_max']} vcm={cs['vcm']}(g{cs['gamma_vcm']})")
        if cs.get("sigma"):
            cfg += f" σ={cs['sigma']}"
        if cs.get("fedprox_mu"):
            cfg += f" μ={cs['fedprox_mu']}"
        if cs.get("ema_decay"):
            cfg += f" EMA={cs['ema_decay']}"
        if cs.get("cma_mode") and cs["cma_mode"] != "none":
            m = cs["cma_mode"]
            cfg += f" CMA={m}"
            if m in ("stale", "conflict", "risk") and cs.get("cma_beta_min"):
                cfg += f"[β{cs['cma_beta_min']}-{cs['cma_beta_max']}]"
                if m == "risk" and cs.get("cma_lambda_stale"):
                    cfg += f"λs{cs['cma_lambda_stale']}"
            if r["final_acc_cma"] is not None:
                cfg += f"({r['final_acc_cma']:.2f})"
        if cs.get("momentum"):
            cfg += f" mom={cs['momentum']}"
        if cs["verify"]:
            cfg += " VERIFY"
        if cs["attack"] and cs["attack"] != "none":
            cfg += f" ATK-{cs['attack']}"
        det = f"{r['attack_detected']*100:.0f}" if (r["verify_total"] > 0 and cs["attack"] not in ("", "none")) else "-"
        lines.append(
            f"| {i} | {r['npz']} | {r['dataset']} | {r['final_acc']:.2f} | {r['wall_clock_s']:.1f} "
            f"| {r['checkpoints']} | {r['avg_wait']:.1f} | {r['max_wait']:.1f} | {r['total_stale_reuses']} "
            f"| {r['expired_buffer']} | {r['verify_pass']}/{r['verify_total']} | {det} | {cfg} |"
        )
    lines.append("")
    lines.append("## 配置细则")
    lines.append("")
    for i, r in enumerate(reversed(rows), 1):
        lines.append(f"- #{i} `{r['npz']}` (导入 {r['imported_at']}):")
        lines.append(f"  ```json")
        lines.append(f"  {json.dumps(r['config_summary'], ensure_ascii=False)}")
        lines.append(f"  ```")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="研究点2 实验结果注册器")
    parser.add_argument("--dir", action="append", default=None,
                        help="扫描目录 (可多次指定; 默认当前目录)")
    parser.add_argument("--show", action="store_true", help="只打印当前注册表")
    parser.add_argument("--reset", action="store_true", help="清空注册表 (慎用)")
    args = parser.parse_args()

    if args.reset:
        write_registry([])
        with open(REGISTRY_MD, "w", encoding="utf-8") as f:
            f.write(render_md([]))
        print(f"registry reset: {REGISTRY_JSON}")
        return

    rows = read_registry()
    if args.show:
        print(f"共 {len(rows)} 条:")
        for r in rows:
            print(f"  {r['npz']} acc={r['final_acc']} wall={r['wall_clock_s']}s "
                  f"verify={r['verify_pass']}/{r['verify_total']} "
                  f"det={r['attack_detected']}")
        return

    if args.dir is None:
        args.dir = ["."]
    existing = {r["npz_id"]: r for r in rows}
    newly = 0
    for d in args.dir:
        for path in sorted(glob.glob(os.path.join(d, "point2_data_*.npz"))):
            nid = npz_id(path)
            if nid in existing:
                continue
            try:
                r = parse_npz(path)
            except Exception as e:
                print(f"  WARN skip {path}: {e}")
                continue
            r["npz_id"] = nid
            existing[nid] = r
            newly += 1
            print(f"  + {os.path.basename(path)} "
                  f"acc={r['final_acc']} wall={r['wall_clock_s']}s "
                  f"verify={r['verify_pass']}/{r['verify_total']}")

    rows = sorted(existing.values(), key=lambda r: (r["imported_at"], r["npz"]))
    write_registry(rows)
    with open(REGISTRY_MD, "w", encoding="utf-8") as f:
        f.write(render_md(rows))
    print(f"done: {newly} new, {len(rows)} total -> {REGISTRY_JSON} / {REGISTRY_MD}")


if __name__ == "__main__":
    main()