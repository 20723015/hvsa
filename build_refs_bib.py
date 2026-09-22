"""refs.bib 基线构建: 解析 docs/ref_point1.md -> Crossref 反查 DOI -> BibTeX.

产出一份**经过 DOI 验证**的 refs.bib, 以及一份匹配报告 (哪些条目在 Crossref
上确认、哪些未确认 —— 未确认的必须在正文引用前人工核对, 不得直接使用)。

用法: python build_refs_bib.py --md docs/ref_point1.md --out <输出目录>
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

API = "https://api.crossref.org/works"


def parse_md(path):
    """从 ref_point1.md 抽取 (key, authors, title, venue, year)."""
    text = open(path, encoding="utf-8").read()
    entries = []
    cur_key = None
    for line in text.splitlines():
        m = re.match(r"^\*\*(.+?)\*\*\s*(?:\(= cite key `(.+?)`\))?\s*$", line.strip())
        if m:
            cur_key = m.group(2) or m.group(1)
            continue
        m = re.match(r"^\*\*(.+?)\*\*\s*$", line.strip())
        if m:
            cur_key = m.group(1)
            continue
        # 著录行: Authors: Title. Venue **vol**(issue), pages (year)
        m = re.match(r"^([A-Z][^:]{2,400}?):\s*(.+)$", line.strip())
        if m and cur_key and "http" not in line:
            authors = m.group(1)
            rest = m.group(2)
            ym = re.search(r"\((\d{4})\)\s*$", rest)
            year = ym.group(1) if ym else ""
            # 标题 = 到第一个句号 (后接空格+大写或 Venue 前)
            tm = re.match(r"^(.+?\.)\s+(?=[A-Z])", rest)
            title = tm.group(1) if tm else rest.split(".")[0] + "."
            entries.append({
                "key": cur_key, "authors": authors,
                "title": title.rstrip("."), "raw": rest, "year": year,
            })
            cur_key = None
    return entries


def norm_title(t):
    t = re.sub(r"[^a-z0-9 ]", " ", t.lower())
    return " ".join(t.split())


def lookup(title, retries=3):
    params = urllib.parse.urlencode({
        "query.bibliographic": title, "rows": 3,
        "select": "DOI,title,author,container-title,issued,type,volume,issue,page,publisher,event",
        "mailto": "thesis-litsearch@example.org",
    })
    url = "%s?%s" % (API, params)
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "thesis-litsearch/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            time.sleep(3 * (a + 1))
    return None


def pick(items, title, year=""):
    """标题模糊匹配 + **年份一致性校验**.

    年份校验是必须的: 模糊标题匹配会抓到同名的其它年份/会议条目 (实测
    Bonawitz CCS 2017 被误配到 2023 年的 proceedings), 错误 DOI 进参考文献
    是实质缺陷, 故年份不符一律判为未确认。
    """
    nt = norm_title(title)
    best = None
    for it in items:
        cand = (it.get("title") or [""])[0]
        nc = norm_title(cand)
        if not nc:
            continue
        toks = nt.split()[:6]
        score = sum(1 for t in toks if t in nc)
        if score < max(3, len(toks) - 1):
            continue
        iy = ""
        dp = it.get("issued", {}).get("date-parts") or [[""]]
        if dp and dp[0]:
            iy = str(dp[0][0] or "")
        # 年份不符 -> 拒绝 (允许 1 年误差: 在线首发 vs 正式出版)
        if year and iy and abs(int(iy) - int(year)) > 1:
            print("        跳过年份不符: 期望 %s 得到 %s (%s)"
                  % (year, iy, cand[:50]))
            continue
        if best is None or score > best[0]:
            best = (score, it)
    return best[1] if best else None


def arxiv_lookup(title, year="", retries=3):
    """Crossref 覆盖差的 arXiv/NeurIPS 条目走 arXiv API 兜底."""
    import xml.etree.ElementTree as ET
    NS = {"a": "http://www.w3.org/2005/Atom"}
    words = " ".join(norm_title(title).split()[:8])
    q = urllib.parse.quote('ti:"%s"' % words)
    url = ("http://export.arxiv.org/api/query?search_query=%s&max_results=3"
           % q)
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "thesis-litsearch/1.0"})
            with urllib.request.urlopen(req, timeout=35) as r:
                root = ET.fromstring(r.read())
            nt = norm_title(title)
            for ent in root.findall("a:entry", NS):
                t = ent.find("a:title", NS)
                t = " ".join(t.text.split()) if t is not None and t.text else ""
                if not t:
                    continue
                nc = norm_title(t)
                toks = nt.split()[:6]
                if sum(1 for x in toks if x in nc) < max(3, len(toks) - 1):
                    continue
                aid = ent.find("a:id", NS)
                pub = ent.find("a:published", NS)
                return {"arxiv_id": aid.text.rsplit("/", 1)[-1] if aid is not None else "",
                        "title": t,
                        "published": pub.text[:4] if pub is not None else ""}
            return None
        except Exception:  # noqa: BLE001
            time.sleep(3 * (a + 1))
    return None


def arxiv_bibtex(e, arx):
    def esc(x):
        return x.replace("&", chr(92)+"&").replace("%", chr(92)+"%").replace("_", chr(92)+"_")
    lines = [
        "@misc{%s," % e["key"],
        "  author = {%s}," % e["authors"],
        "  title = {%s}," % esc(arx["title"]),
        "  year = {%s}," % (e["year"] or arx["published"]),
        "  eprint = {%s}," % arx["arxiv_id"],
        "  archivePrefix = {arXiv},",
        "  doi = {10.48550/arXiv.%s}" % arx["arxiv_id"],
        "}",
    ]
    return chr(10).join(lines)


def source_bibtex(e):
    """未确认条目: 用 ref_point1.md 的来源著录补齐, **不填 DOI**.

    宁可缺 DOI 也不编造 —— 错误 DOI 进参考文献是实质缺陷.
    """
    raw = " ".join(e["raw"].split())
    lines = [
        "% UNVERIFIED: DOI 未由 Crossref/arXiv 确认, 引用前须人工核对",
        "% 来源著录: " + raw[:300],
        "@misc{%s," % e["key"],
        "  author = {%s}," % e["authors"],
        "  title = {%s}," % e["title"],
        "  year = {%s}" % e["year"],
        "}",
    ]
    return chr(10).join(lines)


def bibtex(e, it):
    def esc(s):
        return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    au = it.get("author") or []
    names = []
    for a in au:
        fam = a.get("family", "")
        giv = a.get("given", "")
        names.append(("%s, %s" % (fam, giv)).strip(", ") if fam else giv)
    authors = " and ".join(names) if names else e["authors"]
    venue = (it.get("container-title") or [""])[0]
    yr = ""
    dp = it.get("issued", {}).get("date-parts") or [[""]]
    if dp and dp[0]:
        yr = str(dp[0][0] or "")
    typ = it.get("type", "")
    fields = [("author", authors), ("title", esc((it.get("title") or [e["title"]])[0]))]
    if typ == "proceedings-article":
        ev = (it.get("event") or {}).get("name", "") or venue
        fields.append(("booktitle", esc(ev)))
    elif venue:
        fields.append(("journal", esc(venue)))
    if it.get("volume"):
        fields.append(("volume", it["volume"]))
    if it.get("issue"):
        fields.append(("number", it["issue"]))
    if it.get("page"):
        fields.append(("pages", it["page"].replace("-", "--")))
    if yr:
        fields.append(("year", yr))
    if it.get("publisher"):
        fields.append(("publisher", esc(it["publisher"])))
    if it.get("DOI"):
        fields.append(("doi", it["DOI"]))
    kind = "inproceedings" if typ == "proceedings-article" else "article"
    body = ",\n  ".join("%s = {%s}" % (k, v) for k, v in fields)
    return "@%s{%s,\n  %s\n}\n" % (kind, e["key"], body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    entries = parse_md(args.md)
    print("解析出条目: %d" % len(entries))
    bib, report = [], []
    for i, e in enumerate(entries, 1):
        print("  [%2d/%d] %s" % (i, len(entries), e["title"][:60]))
        data = lookup(e["title"])
        it = pick(data["message"]["items"], e["title"], e["year"]) if data else None
        if it:
            bib.append(bibtex(e, it))
            report.append({"key": e["key"], "title": e["title"], "status": "verified",
                           "doi": it.get("DOI", ""), "year": e["year"]})
            print("        ✅ %s" % it.get("DOI", ""))
        else:
            arx = arxiv_lookup(e["title"], e["year"])
            if arx:
                bib.append(arxiv_bibtex(e, arx))
                report.append({"key": e["key"], "title": e["title"],
                               "status": "verified_arxiv",
                               "doi": arx["arxiv_id"], "year": e["year"]})
                print("        ✅ (arXiv) %s" % arx["arxiv_id"])
            else:
                bib.append(source_bibtex(e))
                report.append({"key": e["key"], "title": e["title"],
                               "status": "UNVERIFIED", "doi": "", "year": e["year"]})
                print("        ❌ 未确认 (用来源著录补齐, 无 DOI, 须人工核对)")
        time.sleep(1.5)

    with open(os.path.join(args.out, "refs_verified.bib"), "w", encoding="utf-8") as f:
        f.write("\n".join(bib))
    with open(os.path.join(args.out, "refs_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in report if r["status"] == "verified")
    print()
    print("已确认: %d / %d" % (ok, len(report)))
    print("未确认 (须人工核对, 不得直接引用):")
    for r in report:
        if r["status"] != "verified":
            print("  - %s: %s" % (r["key"], r["title"][:70]))
    print()
    print("输出:", os.path.join(args.out, "refs_verified.bib"))


if __name__ == "__main__":
    main()
