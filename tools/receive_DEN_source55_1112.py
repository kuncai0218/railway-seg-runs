"""Receive only the SHA-pinned public DEN archives from the named campus sender."""
import argparse
import datetime as dt
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(2**20):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for key in ['bind', 'sender', 'token', 'manifest', 'manifest-sha', 'out']:
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--port', type=int, default=41987)
    a = ap.parse_args()
    manifest_path = Path(a.manifest)
    assert sha(manifest_path) == a.manifest_sha
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['complete'] and manifest['archive_count'] == 55
    records = {r['file']: r for r in manifest['archives']}
    assert len(records) == 55
    assert all(re.fullmatch(r'DEN_source_\d+_\d+_13\.zip', n) for n in records)
    assert all(0 < r['bytes'] <= 512 * 2**20 and re.fullmatch('[0-9a-f]{64}', r['SHA256'])
               for r in records.values())
    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def log(event, **values):
        row = dict(time=dt.datetime.now().astimezone().isoformat(), event=event, **values)
        with (root / 'receive_events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, values):
            body = json.dumps(values).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)

        def path_parts(self):
            parts = self.path.strip('/').split('/')
            if self.client_address[0] != a.sender or not parts or parts[0] != a.token:
                self.send_error(403)
                return None
            return parts[1:]

        def do_GET(self):
            parts = self.path_parts()
            if parts == ['manifest']:
                self.reply(200, dict(manifest_SHA256=a.manifest_sha, total_bytes=manifest['total_bytes'],
                                     peer=list(self.client_address), server=list(self.connection.getsockname())))
            elif parts and len(parts) == 2 and parts[0] == 'status' and parts[1] in records:
                name = parts[1]
                p = root / name
                accepted = p.is_file() and not p.is_symlink() and p.stat().st_size == records[name]['bytes']
                accepted = accepted and sha(p) == records[name]['SHA256']
                self.reply(200, dict(file=name, accepted=accepted, exists=p.exists()))
            elif parts is not None:
                self.send_error(404)

        def do_POST(self):
            parts = self.path_parts()
            if parts is None:
                return
            self.connection.settimeout(60)
            if parts == ['probe'] and self.headers.get('Content-Length') == '4096':
                data = self.rfile.read(4096)
                if len(data) != 4096:
                    self.send_error(400)
                    return
                result = dict(bytes=4096, SHA256=hashlib.sha256(data).hexdigest(),
                              peer=list(self.client_address), server=list(self.connection.getsockname()))
                log('physical_data_port_probe', **result)
                self.reply(200, result)
                return
            if len(parts) != 2 or parts[0] != 'archive' or parts[1] not in records:
                self.send_error(404)
                return
            name = parts[1]
            expected = records[name]
            if (self.headers.get('Content-Length') != str(expected['bytes'])
                    or self.headers.get('X-Archive-SHA256') != expected['SHA256']):
                self.send_error(400)
                return
            target = root / name
            if target.exists() or target.is_symlink():
                self.reply(409, dict(error='existing file retained; check its status before retransmission'))
                return
            partial = root / (name + f'.partial_{time.time_ns()}')
            h = hashlib.sha256()
            count = 0
            start = time.monotonic()
            try:
                with partial.open('xb') as f:
                    while count < expected['bytes']:
                        data = self.rfile.read(min(2**20, expected['bytes'] - count))
                        if not data:
                            raise EOFError('short archive upload; partial retained')
                        f.write(data)
                        h.update(data)
                        count += len(data)
                assert h.hexdigest() == expected['SHA256'], 'archive hash mismatch; partial retained'
                assert not target.exists()
                os.rename(partial, target)
                result = dict(file=name, accepted=True, bytes=count, SHA256=h.hexdigest(),
                              seconds=time.monotonic() - start, peer=list(self.client_address),
                              server=list(self.connection.getsockname()))
                log('archive_accepted', **result)
                (root / (name + '.receipt.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
                self.reply(200, result)
            except Exception as error:
                log('archive_failed', file=name, bytes_received=count, error=repr(error), partial=str(partial))
                self.reply(400, dict(error=repr(error), bytes_received=count, partial_retained=True))

    log('receiver_started', PID=os.getpid(), bind=a.bind, port=a.port, sender=a.sender,
        manifest_SHA256=a.manifest_sha)
    print(json.dumps(dict(PID=os.getpid(), bind=a.bind, port=a.port,
                          manifest_SHA256=a.manifest_sha, destination=str(root))), flush=True)
    http.server.HTTPServer((a.bind, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
