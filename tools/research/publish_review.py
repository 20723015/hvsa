"""Publish only catalogued bytes on a review branch, preserving HEAD and index.

Creates a separate Git index and an archive for static verification. Never
checks out a branch, stages into the user's index, or starts an experiment.
The --push switch is required for remote publication.
"""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_BRANCH='codex/research-code-review-20261028'

def git(*args,env=None,data=None):
    return subprocess.check_output(['git',*args],cwd=str(ROOT),env=env,input=data,stderr=subprocess.PIPE)

def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',default=DEFAULT_BRANCH)
    parser.add_argument('--push',action='store_true')
    args=parser.parse_args()
    assert args.branch.startswith('codex/') and re.fullmatch(r'[A-Za-z0-9_./-]+',args.branch)
    ref='refs/heads/'+args.branch
    assert not git('for-each-ref','--format=%(refname)',ref).strip(),'Local review branch already exists'
    assert not git('ls-remote','--heads','origin',ref).strip(),'Remote review branch already exists'
    original_head=git('rev-parse','HEAD').decode().strip()
    original_ref=git('symbolic-ref','HEAD').decode().strip()
    index=Path(git('rev-parse','--git-path','index').decode().strip())
    if not index.is_absolute():index=ROOT/index
    index_bytes=index.read_bytes()
    catalog=json.loads((ROOT/'research/catalog.json').read_text(encoding='utf-8'))
    files={row['path']:row for row in catalog['file_inventory']}
    files['research/catalog.json']=dict(sha256=digest((ROOT/'research/catalog.json').read_bytes()))
    patterns=[rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----',
        rb'gh[pousr]_[A-Za-z0-9_]{20,}',rb'github_pat_[A-Za-z0-9_]{30,}',
        rb'sk-(?:proj-|ant-)[A-Za-z0-9_-]{20,}',rb'AKIA[0-9A-Z]{16}',
        rb'(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*[\x22\x27]([A-Za-z0-9_-]{24,})[\x22\x27]']
    suspect=[]
    for name,row in files.items():
        path=(ROOT/name).resolve();assert ROOT in path.parents and path.is_file()
        content=path.read_bytes();assert digest(content)==row['sha256'],'Catalog drift: '+name
        assert len(content)<5*2**20,'Unexpected large code-review file: '+name
        for p in patterns:
            for match in re.finditer(p,content):
                suspect.append(dict(path=name,line=content[:match.start()].count(b'\n')+1))
    if suspect:raise ValueError('Potential secret (values withheld): '+json.dumps(suspect))
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/'tmp'/('research_review_publish_'+stamp);out.mkdir(exist_ok=False)
    env=os.environ.copy();env['GIT_INDEX_FILE']=str(out/'review.index')
    git('read-tree','--empty',env=env)
    pathspec=b''.join(name.encode('utf-8')+b'\0' for name in sorted(files))
    spec=out/'paths.nul';spec.write_bytes(pathspec)
    git('--literal-pathspecs','add','--force','--pathspec-from-file='+str(spec),'--pathspec-file-nul',env=env)
    generated=catalog['generated_publication_files']
    for name,source in generated.items():
        blob=git('hash-object','-w','--stdin',data=(ROOT/source).read_bytes()).decode().strip()
        git('update-index','--add','--cacheinfo','100644',blob,name,env=env)
    tree=git('write-tree',env=env).decode().strip()
    entries=git('ls-tree','-rz',tree).split(b'\0')
    seen={}
    batch=[]
    for line in entries:
        if not line:continue
        meta,name=line.split(b'\t',1);mode,kind,oid=meta.split()
        assert kind==b'blob';name=name.decode('utf-8');seen[name]=oid.decode();batch.append(oid+b'\n')
    assert set(seen)==set(files)|set(generated),'Unexpected snapshot contents'
    stream=io.BytesIO(git('cat-file','--batch',data=b''.join(batch)))
    blob_sha={}
    for name,oid in seen.items():
        header=stream.readline().split();assert header[0].decode()==oid and header[1]==b'blob'
        content=stream.read(int(header[2]));assert stream.read(1)==b'\n';blob_sha[name]=digest(content)
        wanted=files[name]['sha256'] if name in files else digest((ROOT/generated[name]).read_bytes())
        assert blob_sha[name]==wanted,'Published byte drift: '+name
    # Test the actual tree as a fresh filesystem, without importing protocol code.
    archive=git('archive','--format=tar',tree);checkout=out/'checkout';checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            dest=(checkout/member.name).resolve()
            assert checkout.resolve() in dest.parents and not member.issym() and not member.islnk()
        tar.extractall(checkout)
    checks=[]
    for rel,flag in [('tools/research/catalog.py','--check'),('tools/research/prepare_inventory.py','--check')]:
        p=subprocess.run([sys.executable,'-B',str(checkout/rel),flag],cwd=str(checkout),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        checks.append(dict(command=[rel,flag],exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
        assert p.returncode==0,(rel,p.stdout,p.stderr)
    # Independent snapshot history prevents inheriting old examples containing credentials.
    base=None
    body=('Organize HVSA research code for experiment review\n\n'
        'Add research-point and experiment navigation with code/config/test mappings. '
        'Include current protocol and experiment sources with byte-exact provenance checks. '
        'Keep raw data, models, credentials, generated slides and local runtime outside the review snapshot.\n\n'
        'Validation: static Python/JSON checks, navigation links, frozen-source inventory, '
        'and exported-tree checks; no protocol benchmarks or training rerun.\n')
    bodyfile=out/'commit_message.txt';bodyfile.write_text(body,encoding='utf-8')
    commit=git('commit-tree',tree,'-F',str(bodyfile)).decode().strip()
    git('update-ref',ref,commit,'0'*40)
    assert index.read_bytes()==index_bytes,'User index changed during publication'
    assert git('rev-parse','HEAD').decode().strip()==original_head
    assert git('symbolic-ref','HEAD').decode().strip()==original_ref
    pushed=False
    if args.push:
        push=git('push','origin',ref+':'+ref);pushed=True
        remote=git('ls-remote','--heads','origin',ref).decode().split()[0]
        assert remote==commit,'Remote branch verification failed'
    assert index.read_bytes()==index_bytes
    assert git('rev-parse','HEAD').decode().strip()==original_head
    result=dict(branch=args.branch,commit=commit,parent_commit=base,independent_snapshot_history=True,tree=tree,pushed=pushed,
        files=len(seen),bytes=sum((ROOT/name).stat().st_size for name in files),
        secret_scan='passed; no matching credential literals or private-key headers',
        original_head=original_head,original_branch_ref=original_ref,
        original_index_sha256=digest(index_bytes),head_and_index_preserved=True,
        exported_tree_checks=checks,git_blob_sha256=blob_sha,
        raw_results_or_models_included=False,protocol_or_training_experiments_executed=False)
    (out/'publication_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['exported_tree_checks','git_blob_sha256']},ensure_ascii=False,indent=2))
    print('RECEIPT='+str(out/'publication_receipt.json'))

if __name__=='__main__':main()
