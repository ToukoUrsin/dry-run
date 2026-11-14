#!/usr/bin/env python3
"""Dry Run dashboard: zero dependency local server."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import json, random, time
import os
ROOT=Path(__file__).resolve().parents[1]

def aggregates():
    url=os.environ.get('DRYRUN_DB_URL')
    if not url:
        return {'status':'unavailable','reason':'DRYRUN_DB_URL is not set','rows':[]}
    try:
        import psycopg
        with psycopg.connect(url) as c:
            rows=c.execute('SELECT bucket, rollout_id, seed, q0, q1, samples, status FROM rollout_1s ORDER BY bucket DESC LIMIT 100').fetchall()
        keys=('bucket','rollout_id','seed','q0','q1','samples','status')
        clean=[]
        for r in rows:
            row=dict(zip(keys,r))
            if hasattr(row.get('bucket'),'isoformat'): row['bucket']=row['bucket'].isoformat()
            clean.append(row)
        return {'status':'ok','rows':clean}
    except Exception as e:
        return {'status':'error','reason':str(e),'rows':[]}

def score():
    agg=aggregates()
    if agg['status']=='ok':
        return {'status':'ok','source':'timescale:rollout_1s','aggregate_rows':len(agg['rows']),'telemetry_samples':sum(r['samples'] for r in agg['rows'])}
    p=ROOT/'data'/'benchmark.json'
    if p.exists():
        try:return json.loads(p.read_text())
        except Exception: pass
    return {'status':'unavailable','source':'timescale:rollout_1s','reason':agg.get('reason','no aggregate data'),'aggregate_rows':0,'telemetry_samples':0}
class Handler(BaseHTTPRequestHandler):
    def _json(self,obj,status=200):
        b=json.dumps(obj).encode(); self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path=='/api/score': return self._json(score())
        if self.path=='/api/telemetry': return self._json(aggregates())
        if self.path=='/api/rollouts':
            rows=[]
            p=ROOT/'data'/'rollouts.jsonl'
            if p.exists():
                for line in p.read_text().splitlines()[-20:]:
                    try: rows.append(json.loads(line))
                    except: pass
            return self._json(rows)
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
