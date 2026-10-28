"""Check or install the byte-exact public freeze inventory, never raw results."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RESOURCE='research/resources/original_cost_inventory_20261020.json'
DESTINATION='results/point1_compare/20261020-212351-556444_decided_minimum_continuation/batch.json'
SHA='cce46db69060fb4656d6e50e58349addb9254dd0ad8b49a56b2efddb6dc2d696'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--prepare',action='store_true')
    args=p.parse_args();source=ROOT/RESOURCE;target=ROOT/DESTINATION
    assert sha(source)==SHA,'Packaged historical freeze inventory differs'
    data=json.loads(source.read_text(encoding='utf-8'))
    missing=[name for name,h in data['critical_source_sha256'].items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
    if missing:raise ValueError('Frozen source bytes differ: '+', '.join(missing))
    if target.exists():
        assert sha(target)==SHA,'Existing target differs; refusing to replace it'
        action='existing inventory verified; no write'
    elif args.prepare:
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(source.read_bytes())
        assert sha(target)==SHA
        action='freeze inventory prepared; no raw historical evidence provided'
    else:action='resource verified; target absent; --prepare would create it'
    print(json.dumps(dict(valid=True,critical_sources_checked=len(data['critical_source_sha256']),
        resource_sha256=SHA,target=DESTINATION,action=action,full_historical_evidence_included=False),ensure_ascii=False))

if __name__=='__main__':main()
