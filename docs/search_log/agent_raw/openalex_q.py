import urllib.request, urllib.parse, json, time, sys, re

QUERIES = [
    'hierarchical federated learning secure aggregation',
    'verifiable aggregation edge server federated learning',
    'homomorphic hash verification federated learning aggregation',
    'linear homomorphic hash aggregation integrity',
    'verifiable secure aggregation constant verification overhead',
    'hierarchical secure aggregation edge servers verification',
    'cloud edge federated learning privacy preserving aggregation verification',
    'threshold secure aggregation federated learning multiple aggregators',
    'secure aggregation verification model dimension independent',
]

def inv_to_abs(inv):
    if not inv:
        return ''
    pos = {}
    for w, ps in inv.items():
        for p in ps:
            pos[p] = w
    return ' '.join(pos[k] for k in sorted(pos))

def fetch(q, per=25):
    url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({
        'search': q, 'per-page': per,
        'filter': 'from_publication_date:2022-01-01',
        'select': 'id,doi,title,publication_year,primary_location,authorships,abstract_inverted_index,cited_by_count,type',
        'mailto': 'litreview@example.org'})
    req = urllib.request.Request(url, headers={'User-Agent': 'litreview/1.0 (mailto:litreview@example.org)'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode('utf-8', 'replace'))

out = {}
for q in QUERIES:
    try:
        items = fetch(q)['results']
    except Exception as ex:
        items = [{'error': str(ex)}]
    out[q] = items
    print('=' * 100)
    print('OPENALEX:', q, '->', len(items))
    for it in items:
        if 'error' in it:
            print(' ERROR', it['error']); continue
        src = ((it.get('primary_location') or {}).get('source') or {}).get('display_name') or ''
        au = ', '.join((a.get('author') or {}).get('display_name', '') for a in (it.get('authorships') or [])[:4])
        ab = inv_to_abs(it.get('abstract_inverted_index'))
        print('---')
        print('  %s | %s | %s | cites=%s' % (it.get('doi'), it.get('publication_year'), src, it.get('cited_by_count')))
        print('  T:', it.get('title'))
        print('  A:', au)
        if ab:
            print('  ABS:', ab[:700])
    sys.stdout.flush()
    time.sleep(1.5)

with open(r'E:\hvsa\.litsearch\openalex_raw.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print('DONE')
