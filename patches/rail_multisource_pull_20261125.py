"""Run a small campus transfer probe before fetching the frozen development folds."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit


EXPECTED={f'{fold}_{role}.npz' for fold in ('r3_2023a','r3_2023b') for role in ('train','val')}
EXPECTED|={'schema.json','route_probe.bin'}


def sha(path):
    digest=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(base,name,target,interface,timeout,continuation=False):
    cmd=['curl','--fail','--silent','--show-error','--noproxy','*','--interface',interface,
         '--connect-timeout','8','--max-time',str(timeout),'--output',str(target),'--write-out',
         '\npeer=%{remote_ip} local=%{local_ip} bytes=%{size_download}\n']
    if continuation and target.exists():
        cmd.extend(['--continue-at','-'])
    cmd.append(base+'/'+name)
    subprocess.run(cmd,check=True,timeout=timeout+3)


def main(args):
    url=urlsplit(args.endpoint)
    if not url.hostname or url.scheme!='http' or not url.path:
        raise ValueError('Expected the campus endpoint in the task record')
    args.out.mkdir(exist_ok=True,parents=True)
    probe=args.out/'route_probe.bin'
    manifest=args.out/'manifest.json'
    fetch(args.endpoint,'route_probe.bin',probe,args.interface,16)
    assert probe.stat().st_size==4096 and sha(probe)==args.probe
    fetch(args.endpoint,'manifest.json',manifest,args.interface,16)
    assert sha(manifest)==args.manifest
    rows=json.loads(manifest.read_text(encoding='utf-8'))['assets']
    assert set(rows)==EXPECTED
    for name,record in rows.items():
        target=args.out/name
        size=int(record['bytes'])
        assert 0<size<=50_000_000
        if target.is_file() and target.stat().st_size==size and sha(target)==record['sha256']:
            continue
        part=target.with_suffix(target.suffix+'.part')
        if part.exists() and part.stat().st_size==size and sha(part)!=record['sha256']:
            part.rename(part.with_suffix('.mismatch'))
        if not(part.exists() and part.stat().st_size==size):
            fetch(args.endpoint,name,part,args.interface,300,continuation=True)
        assert part.stat().st_size==size and sha(part)==record['sha256'],name
        part.replace(target)
        print('verified',name,size,flush=True)
    print('Railway development dataset ready',args.out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--endpoint',required=True)
    p.add_argument('--interface',required=True)
    p.add_argument('--probe',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--out',type=Path,default=Path('rail-development'))
    main(p.parse_args())
