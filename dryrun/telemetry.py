"""TimescaleDB telemetry setup and 30 Hz rollout writer."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
try:
 import psycopg
except ImportError: psycopg=None

SCHEMA=Path(__file__).resolve().parents[1]/'data/schema.sql'
def db_url(): return os.environ.get('DRYRUN_DB_URL','postgresql://postgres:dryrun@127.0.0.1:55439/dryrun')
def setup(url=None):
 if psycopg is None: raise RuntimeError('install psycopg[binary]')
 with psycopg.connect(url or db_url(), autocommit=True) as c:
  statements=[]; buf=[]; in_do=False
  for line in SCHEMA.read_text().splitlines(True):
   buf.append(line)
   if line.lstrip().startswith('DO $$'): in_do=True
   if in_do and line.strip() == 'END $$;':
    statements.append(''.join(buf)); buf=[]; in_do=False
   elif not in_do and line.rstrip().endswith(';'):
    statements.append(''.join(buf)); buf=[]
  if buf: statements.append(''.join(buf))
  for statement in statements:
   if statement.strip(): c.execute(statement)
def write_rollout(seed=7, seconds=2.0, url=None):
 if psycopg is None: raise RuntimeError('install psycopg[binary]')
 rid=f'seed-{seed}-{int(time.time())}'; n=int(seconds*30); t=datetime.now(timezone.utc)
 with psycopg.connect(url or db_url()) as c:
  with c.cursor() as cur:
   for i in range(n):
    ts=t+timedelta(seconds=i/30); q={'q0':0.01*i,'q1':-0.02*i,'q2':0.03*i,'q3':0.0,'q4':0.0,'q5':0.2}; ev='step' if i else 'start'; outcome={'status':'success'} if i==n-1 else None
    cur.execute('INSERT INTO rollout_telemetry(ts,rollout_id,seed,joint,event,outcome) VALUES(%s,%s,%s,%s,%s,%s)',(ts,rid,seed,json.dumps(q),ev,json.dumps(outcome) if outcome else None))
  c.commit()
 return rid,n
def counts(url=None):
 with psycopg.connect(url or db_url()) as c:
  return c.execute("SELECT count(*) FROM rollout_telemetry").fetchone()[0], c.execute("SELECT count(*) FROM rollout_1s").fetchone()[0]
def main():
 p=argparse.ArgumentParser(); p.add_argument('cmd',choices=['setup','write','counts']);p.add_argument('--seed',type=int,default=7);p.add_argument('--seconds',type=float,default=2.0);a=p.parse_args();
 if a.cmd=='setup': setup(); print(json.dumps({'schema':'ok','db':db_url()}))
 elif a.cmd=='write':
  rid,n=write_rollout(a.seed,a.seconds); print(json.dumps({'rollout_id':rid,'rows':n}))
 else: print(json.dumps({'telemetry_rows':counts()[0],'aggregate_rows':counts()[1]}))
if __name__=='__main__': main()
