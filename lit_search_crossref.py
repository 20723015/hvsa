"""检索日志底稿 (期刊侧): Crossref API 检索.

与 lit_search.py (arXiv) 互补 —— arXiv 不覆盖 IEEE/ACM 期刊, 而本文的对照
基线 (VerifyNet/VeriFL/VLMRS/GPVO-FL/MAFSA/DVAFL) 均在期刊上。

用法: python lit_search_crossref.py --out <输出目录>
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

API = "https://api.crossref.org/works"
FROM_DATE = "2022-01-01"
ROWS = 30

QUERIES = {
    "D1_async_semiasync": [
        "asynchronous federated learning staleness",
        "semi-asynchronous federated learning buffered aggregation",
        "hierarchical federated learning asynchronous aggregation",
    ],
    "D2_verifiable_weighted_aggregation": [
        "verifiable aggregation federated learning",
        "secure aggregation verification malicious server federated learning",
        "weighted aggregation verifiable federated learning",
        "robust aggregation Byzantine federated learning verification",
    ],
    "D3_hierarchical_verifiable": [
        "hierarchical federated learning secure aggregation cloud edge",
        "verifiable secure aggregation edge computing federated learning",
        "homomorphic hash verification federated learning",
    ],
    "KNOWN_BASELINES": [
        "VerifyNet secure verifiable federated learning",
        "VeriFL communication-efficient verifiable aggregation federated learning",
        "VLMRS verifiable lightweight multi-dimensional federated learning",
        "GPVO-FL grouped privacy-preserving verification-outsourced federated learning",
        "FedBuff buffered asynchronous federated learning",
    ],
}


def fetch(query, rows=ROWS, retries=3):
    params = urllib.parse.urlencode({
        "query.bibliographic": query,
        "rows": rows,
        "filter": "from-pub-date:%s" % FROM_DATE,
        "select": "DOI,title,author,container-title,issued,abstract,type,URL",
        "mailto": "thesis-litsearch@example.org",
    })
    url = "%s?%s" % (API, params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "thesis-litsearch/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001
            print("    retry %d (%s)" % (attempt + 1, type(e).__name__))
            time.sleep(4 * (attempt + 1))
    return None


def norm(item):
    def first(x):
        return x[0] if isinstance(x, list) and x else (x if isinstance(x, str) else "")
    au = item.get("author") or []
    names = [" ".join(filter(None, [a.get("given", ""), a.get("family", "")])).strip()
             for a in au[:8]]
    yr = ""
    iss = item.get("issued", {}).get("date-parts") or [[""]]
    if iss and iss[0]:
        yr = str(iss[0][0] or "")
    ab = item.get("abstract", "") or ""
    for tag in ("<jats:p>", "</jats:p>", "<jats:title>", "</jats:title>"):
        ab = ab.replace(tag, " ")
    return {
        "doi": item.get("DOI", ""),
        "title": " ".join(first(item.get("title", "")).split()),
        "authors": names,
        "venue": first(item.get("container-title", "")),
        "year": yr,
        "type": item.get("type", ""),
        "url": item.get("URL", ""),
        "abstract": " ".join(ab.split())[:1000],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    log = {"database": "Crossref REST API (api.crossref.org)",
           "from_pub_date": FROM_DATE,
           "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "directions": {}}

    for direction, queries in QUERIES.items():
        print("=" * 90)
        print(direction)
        print("=" * 90)
        hits, qlog = {}, []
        for q in queries:
            print("  查询:", q)
            data = fetch(q)
            if data is None:
                qlog.append({"query": q, "status": "failed", "n": 0})
                time.sleep(2)
                continue
            items = data.get("message", {}).get("items", [])
            new = 0
            for it in items:
                n = norm(it)
                if n["doi"] and n["doi"] not in hits:
                    hits[n["doi"]] = n
                    new += 1
            print("    -> %d 条 (新增 %d)" % (len(items), new))
            qlog.append({"query": q, "status": "ok", "n": len(items), "n_new": new})
            time.sleep(2)
        log["directions"][direction] = {
            "queries": qlog, "n_unique": len(hits),
            "hits": sorted(hits.values(), key=lambda x: x["year"], reverse=True),
        }
        print("  去重后命中: %d" % len(hits))

    path = os.path.join(args.out, "crossref_hits.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print()
    print("已保存:", path)
    print("合计去重命中:", sum(d["n_unique"] for d in log["directions"].values()))


if __name__ == "__main__":
    main()
