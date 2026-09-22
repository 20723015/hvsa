import urllib.request, urllib.parse, json, time, sys

QUERIES = [
    'verifiable secure aggregation',
    'hierarchical secure aggregation',
    'hierarchical federated learning aggregation verification',
    'homomorphic hash federated',
    'verifiable federated learning edge',
    'secure aggregation cloud edge',
    'aggregation verification outsourcing',
    'threshold secure aggregation aggregator',
]

def fetch(q, h=60):
    url = 'https://dblp.org/search/publ/api?' + urllib.parse.urlencode(
        {'q': q, 'format': 'json', 'h': h})
    req = urllib.request.Request(url, headers={'User-Agent': 'litreview/1.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode('utf-8', 'replace'))

res = {}
for q in QUERIES:
    try:
        hits = fetch(q)['result']['hits']
        items = hits.get('hit', [])
    except Exception as ex:
        items = [{'error': str(ex)}]
    res[q] = items
    print('=' * 100)
    print('DBLP:', q, '->', len(items), 'of', (hits.get('@total') if isinstance(items, list) and items and 'error' not in items[0] else '?'))
    for it in items:
        if 'error' in it:
            print('  ERROR', it['error']); continue
        info = it.get('info', {})
        au = info.get('authors', {}).get('author', [])
        if isinstance(au, dict):
            au = [au]
        names = ', '.join(a.get('text', '') for a in au[:4])
        print('  - %s | %s | %s | %s' % (info.get('year'), info.get('venue'), info.get('type'), info.get('title')))
        print('      A:', names, '| DOI:', info.get('doi'), '| key:', info.get('key'))
    sys.stdout.flush()
    time.sleep(1.5)

with open(r'E:\hvsa\.litsearch\dblp_raw.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('DONE')
