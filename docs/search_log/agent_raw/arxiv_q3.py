import urllib.request, urllib.parse, json, time, sys, re

QUERIES = [
    'abs:"homomorphic hash" AND abs:"federated"',
    'abs:"linear homomorphic hash"',
    'abs:"verifiable" AND abs:"aggregation" AND abs:"edge servers"',
    'all:"secure aggregation" AND all:"hierarchical" AND all:"verification"',
    'all:"aggregate signature" AND all:"federated learning"',
    'abs:"publicly verifiable" AND abs:"federated learning" AND abs:"aggregation"',
    'abs:"verification" AND abs:"model dimension" AND abs:"secure aggregation"',
    'all:"edge-assisted federated learning" AND all:"verifiable"',
]

def fetch(q, max_results=40):
    url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode({
        'search_query': q, 'start': 0, 'max_results': max_results,
        'sortBy': 'relevance', 'sortOrder': 'descending'})
    req = urllib.request.Request(url, headers={'User-Agent': 'litreview/1.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode('utf-8', 'replace')

def parse(xml):
    out = []
    for e in re.findall(r'<entry>(.*?)</entry>', xml, re.S):
        def g(tag):
            m = re.search(r'<%s[^>]*>(.*?)</%s>' % (tag, tag), e, re.S)
            return re.sub(r'\s+', ' ', m.group(1)).strip() if m else ''
        idm = g('id')
        out.append({'arxiv': idm.rsplit('/', 1)[-1] if idm else '', 'title': g('title'),
                    'pub': g('published')[:10], 'upd': g('updated')[:10],
                    'authors': re.findall(r'<name>(.*?)</name>', e)[:6],
                    'cats': re.findall(r'<category term="([^"]+)"', e)[:4], 'abs': g('summary')})
    return out

res = {}
for q in QUERIES:
    try:
        items = parse(fetch(q))
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
        print('  ABS:', it['abs'][:700])
    sys.stdout.flush()
    time.sleep(3.5)

with open(r'E:\hvsa\.litsearch\arxiv_raw3.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('DONE')
