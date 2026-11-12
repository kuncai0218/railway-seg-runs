"""Export pinned git object bytes, avoiding Windows checkout newline conversion."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser()
    for key in ['repo', 'manifest', 'out', 'commit']:
        ap.add_argument('--' + key, required=True)
    a = ap.parse_args()
    repo, out = Path(a.repo).resolve(), Path(a.out).resolve()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == a.commit
    manifest = json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    for name, expected in manifest['tracked_file_hashes'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        data = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=repo)
        assert hashlib.sha256(data).hexdigest() == expected
        target = out / name
        assert target.resolve().is_relative_to(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == data, f'existing different source retained: {target}'
        else:
            target.write_bytes(data)
    print(json.dumps(dict(passed=True, files=len(manifest['tracked_file_hashes']),
                          commit=a.commit, destination=str(out))), flush=True)


if __name__ == '__main__':
    main()
