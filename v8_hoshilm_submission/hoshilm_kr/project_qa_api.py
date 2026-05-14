from __future__ import annotations
import argparse, json, mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from project_qa_engine import ProjectQABot
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR.parent / 'web_project_qa'
BOT = None
class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    def _send_file(self, path: Path):
        if not path.exists() or not path.is_file(): self.send_error(404); return
        data = path.read_bytes(); ctype = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
        if path.suffix == '.html': ctype = 'text/html; charset=utf-8'
        self.send_response(200); self.send_header('Content-Type',ctype); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {'/','/index.html'}: return self._send_file(WEB_DIR/'index.html')
        if parsed.path == '/api/health': return self._send_json({'ok': True, 'mode': 'project_qa', 'ckpt': str(BOT.ckpt) if BOT and BOT.ckpt else None})
        target = (WEB_DIR / parsed.path.lstrip('/')).resolve()
        if str(target).startswith(str(WEB_DIR.resolve())): return self._send_file(target)
        self.send_error(404)
    def do_POST(self):
        if urlparse(self.path).path != '/api/chat': self.send_error(404); return
        raw = self.rfile.read(int(self.headers.get('Content-Length','0'))).decode('utf-8', errors='replace')
        try: body = json.loads(raw) if raw else {}
        except json.JSONDecodeError: return self._send_json({'error':'Invalid JSON'}, 400)
        q = str(body.get('message','')).strip(); use_lm = bool(body.get('use_lm', True))
        if not q: return self._send_json({'error':'Empty message'}, 400)
        try: return self._send_json(BOT.ask(q, use_lm=use_lm))
        except Exception as e: return self._send_json({'error': str(e)}, 500)
    def log_message(self, fmt, *args): print('[project_qa_api]', self.address_string(), fmt % args)
def main():
    global BOT
    ap = argparse.ArgumentParser(); ap.add_argument('--host', default='0.0.0.0'); ap.add_argument('--port', type=int, default=8000); ap.add_argument('--ckpt', default=None); args = ap.parse_args()
    BOT = ProjectQABot(ckpt=args.ckpt)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Project QA web running: http://{args.host}:{args.port}'); print('checkpoint:', BOT.ckpt); print('web dir:', WEB_DIR)
    server.serve_forever()
if __name__ == '__main__': main()
