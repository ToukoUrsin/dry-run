#!/usr/bin/env python3
"""Dry Run dashboard: zero dependency local server."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import json, random, time
ROOT=Path(__file__).resolve().parents[1]

def score():
    p=ROOT/'data'/'benchmark.json'
    if p.exists():
        try:return json.loads(p.read_text())
        except Exception: pass
    return {'accuracy':0.0,'bricks':0,'total':0,'cycle_s':0,'drops':0,'false_accepts':0,'false_rejects':0,'checkpoint':'demo'}
class Handler(BaseHTTPRequestHandler):
    def _json(self,obj,status=200):
        b=json.dumps(obj).encode(); self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path=='/api/score': return self._json(score())
        if self.path=='/api/rollouts':
            rows=[]
            p=ROOT/'data'/'rollouts.jsonl'
            if p.exists():
                for line in p.read_text().splitlines()[-20:]:
                    try: rows.append(json.loads(line))
                    except: pass
            return self._json(rows or [{'id':'demo-001','seed':42,'status':'success','bricks':6,'cycle_s':18.4,'reason':'clean separation'}])
        if self.path in ('/','/index.html'):
            b=(ROOT/'web'/'index.html').read_bytes(); self.send_response(200); self.send_header('Content-Type','text/html'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b); return
        self.send_error(404)
    def do_POST(self):
        if self.path!='/api/defect': return self._json({'error':'not found'},404)
        n=int(self.headers.get('Content-Length','0')); raw=self.rfile.read(n)
        try: obj=json.loads(raw); obj['created_at']=time.time()
        except: return self._json({'error':'invalid json'},400)
        p=ROOT/'data'/'defects.jsonl'; p.parent.mkdir(exist_ok=True); p.open('a').write(json.dumps(obj)+'\n')
        return self._json({'ok':True,'defect_id':f"custom-{int(time.time())}"},201)
    def log_message(self,*args): pass
if __name__=='__main__':
    port=int(__import__('os').environ.get('PORT','8787')); print(f'Dry Run dashboard http://localhost:{port}'); ThreadingHTTPServer(('0.0.0.0',port),Handler).serve_forever()
