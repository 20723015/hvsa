import urllib.request, urllib.parse, json, time, sys, re

TITLES = [
    'VerifyNet Secure and Verifiable Federated Learning',
    'VeriFL Communication-Efficient and Fast Verifiable Aggregation for Federated Learning',
    'Practical Secure Aggregation for Privacy-Preserving Machine Learning',
    'On-the-fly verification of rateless erasure codes for efficient content distribution',
    'Secure Single-Server Aggregation with (Poly)Logarithmic Overhead',
    'ELSA Secure Aggregation for Federated Learning with Malicious Actors',
    'EIFFeL Ensuring Integrity for Federated Learning',
    'RoFL Robustness of Secure Federated Learning',
    'Verifiable Privacy-Preserving Aggregation in Federated Learning with Edge Computing',
    'Secure and Verifiable Federated Learning against poisoning attacks hierarchical',
]

def fetch(q, rows=5):
    url = 'https://api.crossref.org/works?' + urllib.parse.urlencode({
        'query.bibliographic': q, 'rows': rows,
        'select': 'DOI,title,author,issued,container-title,type,is-referenced-by-count',
        'mailto': 'litreview@example.org'})
    req = urllib.request.Request(url, headers={'User-Agent': 'litreview/1.0 (mailto:litreview@example.org)'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode('utf-8', 'replace'))['message']['items']

for q in TITLES:
    print('=' * 100)
    print('Q:', q)
    try:
        items = fetch(q)
    except Exception as ex:
        print('  ERR', ex); continue
    for it in items:
        t = re.sub(r'\s+', ' ', (it.get('title') or [''])[0])
        ct = (it.get('container-title') or [''])
        ct = ct[0] if ct else ''
        yr = (it.get('issued', {}).get('date-parts', [['']]) or [['']])[0]
        yr = yr[0] if yr else ''
        au = ', '.join((a.get('family', '') or '') for a in (it.get('author') or [])[:3])
        print('  - %s | %s | %s | cites=%s' % (it.get('DOI'), yr, ct, it.get('is-referenced-by-count')))
        print('      T:', t)
        print('      A:', au, '| type:', it.get('type'))
    sys.stdout.flush()
    time.sleep(1.5)
print('DONE')
