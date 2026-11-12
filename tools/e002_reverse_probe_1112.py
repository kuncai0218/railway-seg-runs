"""Four-KiB, source-restricted campus reverse-path diagnostic; no file upload API."""
import argparse
import datetime
import hashlib
import http.server
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bind', required=True)
    ap.add_argument('--sender', required=True)
    ap.add_argument('--token', required=True)
    ap.add_argument('--port', type=int, default=41986)
    ap.add_argument('--receipt', required=True)
    a = ap.parse_args()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            if self.client_address[0] != a.sender or self.path != '/' + a.token + '/probe':
                self.send_error(403)
                return
            if self.headers.get('Content-Length') != '4096':
                self.send_error(413)
                return
            self.connection.settimeout(10)
            data = self.rfile.read(4096)
            if len(data) != 4096:
                self.send_error(400)
                return
            result = dict(time=datetime.datetime.now().astimezone().isoformat(),
                          peer=list(self.client_address), server=list(self.connection.getsockname()),
                          bytes=len(data), SHA256=hashlib.sha256(data).hexdigest(),
                          bulk_bytes_received=0, scope='4KiB path probe only')
            Path(a.receipt).write_text(json.dumps(result, indent=2), encoding='utf-8')
            body = json.dumps(result).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(json.dumps(dict(bind=a.bind, port=a.port, sender=a.sender,
                          purpose='campus reverse-path 4KiB probe; no bulk endpoint')), flush=True)
    http.server.HTTPServer((a.bind, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
