"""检索日志底稿: 按三个方向查 arXiv (公开 API, 无需密钥).

产出可复现的检索式 + 命中清单, 供 docs/search_log/检索日志.md 使用。
查询式与时间窗全部显式记录, 便于他人复现同一检索。

注意: arXiv API 建议请求间隔 >= 3s, 本脚本已内置节流。
用法: python lit_search.py --out <输出目录>
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}
ARXIV = "http://export.arxiv.org/api/query"
DATE_WINDOW = "submittedDate:[202201010000 TO 202612312359]"

# 三个方向的检索式 (显式记录进日志)
QUERIES = {
    "D1_async_semiasync": [
        'all:"federated learning" AND all:"asynchronous"',
        'all:"federated learning" AND all:"staleness"',
        'all:"federated learning" AND all:"semi-asynchronous"',
        'all:"federated learning" AND all:"buffered aggregation"',
        'ti:"FedBuff" OR ti:"FedAsync" OR ti:"Papaya" OR ti:"FedAT"',
    ],
    "D2_verifiable_weighted_aggregation": [
        'all:"federated learning" AND all:"verifiable aggregation"',
        'all:"secure aggregation" AND all:"verifiable"',
        'all:"federated learning" AND all:"malicious server"',
        'all:"federated learning" AND all:"aggregation weight" AND all:"verification"',
        'all:"federated learning" AND all:"Byzantine" AND all:"robust aggregation"',
    ],
    "D3_hierarchical_verifiable": [
        'all:"hierarchical federated learning" AND all:"secure aggregation"',
        'all:"federated learning" AND all:"edge" AND all:"verifiable"',
        'all:"federated learning" AND all:"homomorphic hash"',
        'all:"federated learning" AND all:"cloud-edge" AND all:"privacy"',
    ],
}


def fetch(query, max_results=40, retries=3):
    q = "%s AND %s" % (query, DATE_WINDOW)
    url = "%s?search_query=%s&start=0&max_results=%d&sortBy=submittedDate&sortOrder=descending" % (
        ARXIV, urllib.parse.quote(q), max_results)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "thesis-litsearch/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print("    retry %d (%s)" % (attempt + 1, type(e).__name__))
            time.sleep(5 * (attempt + 1))
    return None


def parse(xml_bytes):
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall("a:entry", NS):
        def txt(tag):
            n = e.find(tag, NS)
            return n.text.strip() if n is not None and n.text else ""
        aid = txt("a:id")
        out.append({
            "arxiv_id": aid.rsplit("/", 1)[-1] if aid else "",
            "title": " ".join(txt("a:title").split()),
            "published": txt("a:published")[:10],
            "updated": txt("a:updated")[:10],
            "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)][:8],
            "summary": " ".join(txt("a:summary").split())[:1200],
            "url": aid,
            "primary_category": (e.find("a:primary_category", NS).get("term")
                                 if e.find("a:primary_category", NS) is not None else ""),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-results", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    log = {"database": "arXiv API (export.arxiv.org)",
           "date_window": DATE_WINDOW,
           "sort": "submittedDate descending",
           "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "directions": {}}

    for direction, queries in QUERIES.items():
        print("=" * 90)
        print(direction)
        print("=" * 90)
        hits = {}
        qlog = []
        for q in queries:
            print("  查询:", q)
            raw = fetch(q, args.max_results)
            if raw is None:
                print("    -> 失败")
                qlog.append({"query": q, "status": "failed", "n": 0})
                time.sleep(3)
                continue
            entries = parse(raw)
            new = 0
            for e in entries:
                if e["arxiv_id"] and e["arxiv_id"] not in hits:
                    hits[e["arxiv_id"]] = e
                    new += 1
            print("    -> %d 条 (新增 %d)" % (len(entries), new))
            qlog.append({"query": q, "status": "ok", "n": len(entries), "n_new": new})
            time.sleep(3.5)
        log["directions"][direction] = {
            "queries": qlog,
            "n_unique": len(hits),
            "hits": sorted(hits.values(), key=lambda x: x["published"], reverse=True),
        }
        print("  去重后命中: %d" % len(hits))

    path = os.path.join(args.out, "arxiv_hits.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print()
    print("已保存:", path)
    total = sum(d["n_unique"] for d in log["directions"].values())
    print("三方向去重命中合计:", total)


if __name__ == "__main__":
    main()
