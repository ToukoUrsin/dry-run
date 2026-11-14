"""Run the reference expert through the repo's own StationEnv. usage: reftest.py REPO SEED0 SEED1 [tier]"""
import sys, time, json
repo = sys.argv[1]; sys.path.insert(0, repo); sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np, mujoco
import dryrun.env as envmod
from expert_reference import Expert
envmod.MeasuredExpert = Expert
from collections import Counter
tier = sys.argv[4] if len(sys.argv) > 4 else 'loose'
tot = nn = 0; C = Counter(); t00 = time.time()
for seed in range(int(sys.argv[2]), int(sys.argv[3])):
    t0 = time.time(); env = envmod.StationEnv(seed, tier=tier, render_size=64); env.reset(); ev = []
    for n in range(1, 4000):
        obs, _, done, info = env.step(env.expert_action())
        if done or env._planner.phase == 'done': break
    for _ in range(30): env.step(env._planner.action())
    m, d = env.model, env.data; ok = 0; bad = []
    bowl = {k: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, k + '_bowl')][:2] for k in ('good', 'defective')}
    for s in env.specs:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'brick_{s.index}'); p = d.xpos[b]
        want = 'good' if s.defect == 'none' else 'defective'
        inb = bool(np.linalg.norm(p[:2] - bowl[want]) < 0.095 and 0 < p[2] < 0.11); ok += inb
        if not inb: bad.append((f'brick_{s.index}', s.size, np.round(p, 3).tolist()))
    evs = Counter(e['event'] for e in env._events); C.update(evs)
    fails = [(e['event'], e.get('brick'), e.get('reason'), e.get('pad_n')) for e in env._events if e['event'] in ('miss', 'slip', 'ik_fail', 'unstable')]
    tot += ok; nn += len(env.specs)
    print(seed, f'{ok}/{len(env.specs)}', 'ticks', n, 'wall', round(time.time() - t0, 1), dict(evs), bad, fails, flush=True)
    env.close()
print(f'TOTAL {tier} {tot}/{nn} = {tot/max(nn,1):.3f} wall {time.time()-t00:.0f}s events {dict(C)}')
