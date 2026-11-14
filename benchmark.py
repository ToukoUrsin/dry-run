#!/usr/bin/env python3
from dryrun.sim import run_benchmark
import argparse,json
p=argparse.ArgumentParser(); p.add_argument('--seeds',type=int,default=200); p.add_argument('--output',default='runs/benchmark.json'); a=p.parse_args()
print(json.dumps({k:v for k,v in run_benchmark(range(a.seeds),a.output).items() if k!='rows'},indent=2))
