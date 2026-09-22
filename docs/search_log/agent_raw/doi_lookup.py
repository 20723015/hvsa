import urllib.request, urllib.parse, json, time, sys, re

DOIS = [
    '10.1109/tmc.2026.3679423',
    '10.1016/j.comcom.2025.108271',
    '10.1109/icc59461.2026.11587871',
    '10.1109/tdsc.2026.3707153',
    '10.1109/tdsc.2024.3472869',
    '10.1109/tdsc.2025.3589160',
    '10.1109/tdsc.2021.3126323',
    '10.1109/tdsc.2022.3226508',
    '10.1109/tdsc.2025.3565640',
    '10.1109/tdsc.2024.3380669',
    '10.1109/tdsc.2026.3717223',
    '10.1016/j.comnet.2023.110079',
    '10.1016/j.neucom.2025.130712',
    '10.1109/flta63145.2024.10839609',
    '10.1109/jiot.2023.3277553',
    '10.1109/tdsc.2024.3350206',
    '10.1109/tcc.2026.3723542',
    '10.1016/j.comnet.2026.112421',
    '10.1016/j.comnet.2026.112498',
    '10.3390/math13223687',
    '10.3390/app14135361',
    '10.3390/e25081125',
]

def inv_to_abs(inv):
    if not inv:
        return ''
    pos = {}
    for w, ps in inv.items():
        for p in ps:
            pos[p] = w
    return ' '.join(pos[k] for k in sorted(pos))

def openalex(doi):
    url = 'https://api.openalex.org/works/https://doi.org/' + doi + '?mailto=litreview@example.org'
    req = urllib.request.Request(url, headers={'User-Agent': 'litreview/1.0 (mailto:litreview@example.org)'})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode('utf-8', 'replace'))
    src = ((d.get('primary_location') or {}).get('source') or {}).get('display_name') or ''
    return {'src': src, 'year': d.get('publication_year'), 'title': d.get('title'),
            'abs': inv_to_abs(d.get('abstract_inverted_index')), 'cites': d.get('cited_by_count'),
            'type': d.get('type')}

def crossref(doi):
    url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={'User-Agent': 'litreview/1.0 (mailto:litreview@example.org)'})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode('utf-8', 'replace'))['message']
    ab = re.sub(r'<[^>]+>', ' ', d.get('abstract', '') or '')
    return {'title': (d.get('title') or [''])[0], 'abs': re.sub(r'\s+', ' ', ab).strip(),
            'src': ((d.get('container-title') or ['']) or [''])[0]}

out = {}
for doi in DOIS:
    rec = {'doi': doi}
    try:
        rec['openalex'] = openalex(doi)
    except Exception as ex:
        rec['openalex_error'] = str(ex)
    time.sleep(1.0)
    try:
        rec['crossref'] = crossref(doi)
    except Exception as ex:
        rec['crossref_error'] = str(ex)
    out[doi] = rec
    print('=' * 100)
    print('DOI:', doi)
    oa = rec.get('openalex')
    if oa:
        print('  OA src=%s year=%s cites=%s type=%s' % (oa['src'], oa['year'], oa['cites'], oa['type']))
        print('  OA T:', oa['title'])
        if oa['abs']:
            print('  OA ABS:', oa['abs'][:1500])
    else:
        print('  OA ERR:', rec.get('openalex_error'))
    cr = rec.get('crossref')
    if cr:
        print('  CR V:', cr['src'])
        if cr['abs']:
            print('  CR ABS:', cr['abs'][:1500])
    sys.stdout.flush()
    time.sleep(1.5)

with open(r'E:\hvsa\.litsearch\dois_raw.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print('DONE')
