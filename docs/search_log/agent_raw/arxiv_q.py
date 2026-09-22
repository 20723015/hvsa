import urllib.request, urllib.parse, json, time, sys, re

QUERIES = [
    'all:"hierarchical federated learning" AND all:"secure aggregation"',
    'all:"client-edge-cloud" AND all:"federated learning"',
    'all:"verifiable aggregation" AND all:"federated learning"',
    'all:"homomorphic hash" AND all:aggregation',
    'all:"secure aggregation" AND all:"edge server"',
    'all:"verifiable secure aggregation"',
    'all:"hierarchical federated learning" AND all:"privacy-preserving"',
    'all:"threshold" AND all:"secure aggregation" AND all:"federated learning"',
]

def fetch(q, max_results=40):
    url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode({
        'search_query': q, 'start': 0, 'max_results': max_results,
        'sortBy': 'relevance', 'sortOrder': 'descending'})
    req = urllib.request.Request(url, headers={'User-Agent': 'litreview/1.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode('utf-8', 'replace')

def parse(xml):
    entries = re.findall(r'<entry>(.*?)</entry>', xml, re.S)
    out = []
    for e in entries:
        def g(tag):
            m = re.search(r'<%s[^>]*>(.*?)</%s>' % (tag, tag), e, re.S)
            return re.sub(r'\s+', ' ', m.group(1)).strip() if m else ''
        idm = g('id')
        aid = idm.rsplit('/', 1)[-1] if idm else ''
        title = g('title')
        summ = g('summary')
        pub = g('published')[:10]
        upd = g('updated')[:10]
        authors = re.findall(r'<name>(.*?)</name>', e)
        cats = re.findall(r'<category term="([^"]+)"', e)
        out.append({'arxiv': aid, 'title': title, 'pub': pub, 'upd': upd,
                    'authors': authors[:6], 'cats': cats[:4], 'abs': summ})
    return out

res = {}
for q in QUERIES:
    try:
        xml = fetch(q)
        items = parse(xml)
    except Exception as ex:
        items = [{'error': str(ex)}]
    res[q] = items
    print('=' * 100)
    print('QUERY:', q, ' -> ', len(items), 'results')
    for it in items:
        if 'error' in it:
            print('  ERROR', it['error']); continue
        print('---')
        print('  arXiv:%s | pub %s | upd %s' % (it['arxiv'], it['pub'], it['upd']))
        print('  T:', it['title'])
        print('  A:', ', '.join(it['authors']))
        print('  C:', ','.join(it['cats']))
        print('  ABS:', it['abs'][:900])
    sys.stdout.flush()
    time.sleep(3.5)

with open(r'E:\hvsa\.litsearch\arxiv_raw.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('DONE')
