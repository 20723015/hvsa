import urllib.request, urllib.parse, json, time, sys, re

QUERIES = [
    'hierarchical federated learning secure aggregation',
    'verifiable aggregation edge servers federated learning',
    'homomorphic hash federated learning verification',
    'client-edge-cloud federated learning privacy preserving aggregation',
    'verifiable secure aggregation constant size proof',
    'threshold secure aggregation multiple aggregators federated learning',
    'aggregation result verification cloud edge collaborative',
    'linear homomorphic hash aggregation integrity verification',
]

def fetch(q, rows=30):
    url = 'https://api.crossref.org/works?' + urllib.parse.urlencode({
        'query.bibliographic': q, 'rows': rows,
        'filter': 'from-pub-date:2022-01-01',
        'select': 'DOI,title,author,issued,container-title,type,abstract,is-referenced-by-count',
        'mailto': 'litreview@example.org'})
    req = urllib.request.Request(url, headers={'User-Agent': 'litreview/1.0 (mailto:litreview@example.org)'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode('utf-8', 'replace'))

res = {}
for q in QUERIES:
    try:
        data = fetch(q)
        items = data['message']['items']
    except Exception as ex:
        items = [{'error': str(ex)}]
    res[q] = items
    print('=' * 100)
    print('CROSSREF QUERY:', q, ' -> ', len(items))
    for it in items:
        if 'error' in it:
            print('  ERROR', it['error']); continue
        t = (it.get('title') or [''])[0]
        ct = (it.get('container-title') or [''])
        ct = ct[0] if ct else ''
        yr = ''
        iss = it.get('issued', {}).get('date-parts', [['']])
        if iss and iss[0]:
            yr = iss[0][0]
        au = ', '.join((a.get('family', '') or '') for a in (it.get('author') or [])[:4])
        ab = re.sub(r'<[^>]+>', ' ', it.get('abstract', '') or '')
        ab = re.sub(r'\s+', ' ', ab).strip()
        print('---')
        print('  DOI:%s | %s | %s | cites=%s' % (it.get('DOI'), yr, it.get('type'), it.get('is-referenced-by-count')))
        print('  T:', re.sub(r'\s+', ' ', t))
        print('  A:', au)
        print('  V:', ct)
        if ab:
            print('  ABS:', ab[:700])
    sys.stdout.flush()
    time.sleep(2.0)

with open(r'E:\hvsa\.litsearch\crossref_raw.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('DONE')
