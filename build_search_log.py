"""检索日志生成: 汇总 arXiv + Crossref 两库检索, 产出正式检索日志 markdown.

输入: arxiv_hits.json / crossref_hits.json / refs_report.json
输出: 检索日志.md

日志必须包含: 检索式、数据库、时间窗、命中清单、覆盖度检验结果、局限性声明。
覆盖度检验与局限性声明是必需的 —— 没有它们, 命中数会被误读为新颖性证据。

注: 本文件内的中文引号统一用「」, 避免与 Python 字符串定界符混淆。
"""

import argparse
import json
import os
from datetime import datetime

DIRECTION_TITLES = {
    "D1_async_semiasync": "方向 1：异步/半异步联邦学习主干",
    "D2_verifiable_weighted_aggregation": "方向 2：可验证加权聚合与策略正确性",
    "D3_hierarchical_verifiable": "方向 3：分层（云边端）可验证安全聚合",
    "KNOWN_BASELINES": "对照基线（已知条目反查，用于覆盖度检验）",
}


def load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arx = load(os.path.join(args.dir, "arxiv_hits.json"))
    cr = load(os.path.join(args.dir, "crossref_hits.json"))
    rep = load(os.path.join(args.dir, "refs_report.json"))

    L = ["# 新颖性检索日志（研究点一 / 研究点二）", ""]
    L += ["本日志记录为确认本文「首次/据本文检索所知」类表述而执行的系统检索。",
          "**检索结果只能说明「检索范围内未见同类工作」，不构成「不存在」的证明**——",
          "见文末局限性声明。", ""]

    L += ["## 1. 检索设置", "", "| 项 | 值 |", "|---|---|",
          "| 数据库 | arXiv API（`export.arxiv.org`）；Crossref REST API（`api.crossref.org`） |"]
    if arx:
        L += ["| 时间窗（arXiv） | `%s` |" % arx.get("date_window", ""),
              "| 排序 | %s |" % arx.get("sort", "")]
    if cr:
        L += ["| 时间窗（Crossref） | from-pub-date `%s` |" % cr.get("from_pub_date", "")]
    L += ["| 检索日期 | %s |" % datetime.now().strftime("%Y-%m-%d"),
          "| 覆盖缺口 | arXiv 不收录 IEEE/ACM 期刊；Crossref 对 NeurIPS 与 arXiv 预印本覆盖不全 |", ""]

    for src, data in (("arXiv", arx), ("Crossref", cr)):
        if not data:
            continue
        L += ["## 2. %s 检索式与命中" % src, ""]
        for key, d in data["directions"].items():
            L += ["### %s（%s）" % (DIRECTION_TITLES.get(key, key), src), "",
                  "| 检索式 | 命中 | 去重新增 |", "|---|---|---|"]
            for q in d["queries"]:
                if q.get("status") != "ok":
                    L += ["| `%s` | 失败 | — |" % q["query"]]
                else:
                    L += ["| `%s` | %d | %d |" % (q["query"], q.get("n", 0), q.get("n_new", 0))]
            L += ["", "**去重命中合计：%d**" % d["n_unique"], "",
                  "命中清单（按发表时间倒序，取前 15 条）：", "",
                  "| 日期 | 标题 | 标识 |", "|---|---|---|"]
            for h in d["hits"][:15]:
                ident = h.get("arxiv_id") or h.get("doi") or ""
                L += ["| %s | %s | `%s` |" % (h.get("published") or h.get("year", ""),
                                              (h.get("title") or "")[:88], ident)]
            if d["n_unique"] > 15:
                L += ["", "（其余 %d 条见 `%s_hits.json`）" % (d["n_unique"] - 15, src.lower())]
            L += [""]

    L += ["## 3. 覆盖度检验（**必读**）", "",
          "对**已知存在**的对照条目做反查，检验上述检索式能否命中：", "",
          "| 已知条目 | 查询式检索是否命中 |", "|---|---|",
          "| GPVO-FL | 命中（DOI 精确） |",
          "| VeriFL | 仅命中衍生工作 LightVeriFL，原篇未命中 |",
          "| VerifyNet | 未命中（查询式检索漏检） |",
          "| VLMRS | 未命中 |",
          "| MAFSA / DVAFL | 未命中 |",
          "| FedBuff | 仅命中引用它的后续工作 |", "",
          "**结论：查询式检索对已知条目存在明显漏检**（6 条对照中仅 1 条被精确命中）。",
          "因此**不得**以「某方向命中数低」直接论证新颖性——低命中更可能反映检索式与数据库覆盖的局限。",
          "已知条目改用**精确标题反查 DOI** 的方式单独核验（见第 4 节）。", ""]

    if rep:
        bad = [r for r in rep if not r["status"].startswith("verified")]
        L += ["## 4. 参考文献条目核验（`ref_point1.md` → BibTeX）", "",
              "对 %d 条来源著录逐条做 DOI/arXiv 反查，**年份一致性校验已启用**" % len(rep),
              "（模糊标题匹配会抓到同名的其它年份条目，实测 Bonawitz CCS 2017 曾被误配到 2023 年 proceedings）：",
              "", "| 状态 | 条数 |", "|---|---|",
              "| Crossref 验证（有 DOI） | %d |" % sum(1 for r in rep if r["status"] == "verified"),
              "| arXiv 验证（有 arXiv ID） | %d |" % sum(1 for r in rep if r["status"] == "verified_arxiv"),
              "| 未验证（仅来源著录，**无 DOI**） | %d |" % len(bad), ""]
        if bad:
            L += ["未验证条目（引用前须人工核对，bib 中已用 `% UNVERIFIED` 标注）：", ""]
            for r in bad:
                L += ["- `%s` — %s" % (r["key"], r["title"][:80])]
            L += [""]

    L += ["## 5. 与本文最接近的工作及区分点", "",
          "以下两条由检索发现，**机制与本文不同但主题最接近**，正文必须引用并明确区分：", "",
          "### 5.1 End-to-End Verifiable and Robust Federated Learning（arXiv 2609.15521，2026-09）", "",
          "- **做了什么**：用密码学承诺 + 非交互零知识证明，使**鲁棒聚合流水线**（余弦相似度离群点剔除）可公开审计。",
          "- **与本文区分**：①验证对象不同——它验证「离群点是否被正确剔除」，本文验证「聚合结果与权重来源」；",
          "  ②原语不同——它用 NIZK，本文用 LHH 同态哈希 + Ed25519；",
          "  ③信任模型不同——它面向公开可审计，本文面向单 TA 架构下 ES 的独立复算。", "",
          "### 5.2 Nautilus: A Verifiable Hierarchical FL Framework for Vehicular-Edge-Cloud Systems（arXiv 2606.23017，2026-06）", "",
          "- **做了什么**：分层（车—边—云）FL 中，用 ZKP 验证**调度公平性与压缩指令执行合规性**。",
          "- **与本文区分**：①它验证调度决策与压缩合规，本文验证聚合正确性与策略绑定；",
          "  ②同样基于 ZKP 而非同态哈希；③它面向车联网动态调度，本文面向组级半异步的 checkpoint 提交链路。", "",
          "## 6. 局限性声明（正文引用检索结论时必须一并交代）", "",
          "1. **检索范围有限**：仅 arXiv 与 Crossref，未覆盖 IEEE Xplore / ACM DL / SpringerLink 全文检索，",
          "   也未做引文追踪（backward/forward citation chasing）。",
          "2. **查询式检索存在漏检**：第 3 节的覆盖度检验已证明确知条目亦会被漏检，故低命中数不能作为新颖性证据。",
          "3. **时间窗**：arXiv 限 2022-01-01 起；Crossref 限 from-pub-date 2022-01-01。窗口之前的经典工作",
          "   （如 Bonawitz 2017、McMahan 2016）不在命中清单内，但已通过 `ref_point1.md` 单独著录。",
          "4. **未做全文精读**：命中清单基于题录与摘要，逐条区分点需精读原文后补入，本日志不替代精读。",
          "5. **因此**：正文如需保留「首次」类表述，措辞须限定为「据本文检索所知（检索式与范围见附录）」，",
          "   且**不得**用于对 IEEE/ACM 期刊文献的排他性断言。", ""]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("检索日志已生成:", args.out, "(%d 行)" % len(L))


if __name__ == "__main__":
    main()
