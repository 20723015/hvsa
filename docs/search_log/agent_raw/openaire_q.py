import urllib.request, urllib.parse, json, time, sys, re

TITLES = [
    'Practical Multi-Key Secure Aggregation With Homomorphic Hash Verification for Cross-Silo Federated Learning',
    'A Homomorphic MAC-based verifiable secure aggregation for federated learning in cloud-edge AIoT',
    'Privacy-Preserving Verification of Secure Aggregation for Hierarchical Peer-to-Peer Federated Learning',
    'SVTAgg Secure Verifiable and Traceable Model Aggregation for Edge-Assisted Federated Learning',
    'SHIELD Secure Aggregation Against Poisoning in Hierarchical Federated Learning',
    'Privacy-preserving and Verifiable Federated Learning with weighted average aggregation in edge computing',
    'VDBFL Verifiable privacy-preserving federated learning with decentralized Byzantine-robust aggregation in edge computing',
    'LVSA Lightweight and verifiable secure aggregation for federated learning',
    'VeSAFL Verifiable Secure Aggregation for Privacy-Preserving Federated Learning',
    'Malicious-Resistant Non-Interactive Verifiable Aggregation for Federated Learning',
    'FedAggVerify Publicly Verifiable and Privacy-Preserving Aggregation With Ownership Protection for Federated Learning',
    'OPSA Efficient and Verifiable One-Pass Secure Aggregation With TEE for Federated Learning',
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
                out.append((k, str(v['$'])[:1200]))
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
            if v2 in seen:
                continue
            seen.add(v2)
            print('  %s: %s' % (k, v2[:900]))
        if not got:
            print('  (no results)')
    except Exception as ex:
        print('  ERR', ex)
    sys.stdout.flush()
    time.sleep(2)
print('DONE')
