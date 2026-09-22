import urllib.request, urllib.parse, json, time, sys, re

TITLES = [
    'VSAFL Verifiable and Secure Aggregation With Poly Logarithmic Overhead in Federated Learning',
    'PriVeriFL Privacy-Preserving and Aggregation-Verifiable Federated Learning',
    'ClusterGuard Secure Clustered Aggregation for Federated Learning With Robustness',
    'VCFL Verifiable and communication-efficient federated learning against collusion attack for secure aggregation',
    'SecHFed Achieving secure hierarchical federated learning against poisoning attacks without data assumptions',
    'HFedM2ND secure hierarchical federated learning framework lightweight multi-layer malicious-node detection',
    'RFLPA Robust Federated Learning Framework against Poisoning Attacks with Secure Aggregation',
    'Secure group aggregation for privacy-protection federated learning',
    'Dual-Key Verifiable Secure Aggregation for Federated Learning via Blockchain',
    'Lightweight and Verifiable Secure Aggregation for Multi-dimensional Data in Edge-enhanced IoT',
]

def openaire(title):
    url = 'https://api.openaire.eu/search/publications?' + urllib.parse.urlencode(
        {'title': title, 'format': 'json', 'size': 3})
    req = urllib.request.Request(url, headers={'User-Agent': 'litreview/1.0'})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode('utf-8', 'replace'))

def dig(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ('description', 'title') and isinstance(v, dict) and '$' in v:
                out.append((k, str(v['$'])[:1500]))
            else:
                dig(v, out)
    elif isinstance(obj, list):
        for v in obj:
            dig(v, out)

for t in TITLES:
    print('=' * 100)
    print('TITLE:', t)
    try:
        d = openaire(t)
        got = []
        dig(d.get('response', {}).get('results', {}), got)
        seen = set()
        for k, v in got:
            v2 = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', v)).strip()
            if v2 in seen or len(v2) < 120:
                continue
            seen.add(v2)
            print('  %s: %s' % (k, v2[:1100]))
        if not got:
            print('  (no results)')
    except Exception as ex:
        print('  ERR', ex)
    sys.stdout.flush()
    time.sleep(2)
print('DONE')
