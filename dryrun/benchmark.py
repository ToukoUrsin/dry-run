"""Held-out benchmark that runs generated stations in MuJoCo physics."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import mujoco
from .mjcf_station import station_xml, ROOT, layout_bricks
from .render import rollout

def run(seeds, workdir=Path('/bulk2/runs/dryrun/benchmark')):
    workdir.mkdir(parents=True, exist_ok=True); rows=[]; started=time.perf_counter(); sentence='good and defective bowls on the sides, inspection camera in front'
    for seed in seeds:
        xml=workdir/f'station_{int(seed):04d}.xml'; xml.write_text(station_xml(int(seed),sentence,robot_include=str((ROOT/'assets/SO101/Simulation/SO101/so101_dryrun.xml').resolve())))
        model=mujoco.MjModel.from_xml_path(str(xml)); data=mujoco.MjData(model); mujoco.mj_forward(model,data)
        r=rollout(model,data,int(seed),writer=None,specs=layout_bricks(int(seed),sentence)); rows.append({'seed':int(seed),'bricks':r['bricks'],'successes':r['successes'],'score':r['score'],'picked_by_contact':sum(int(e['picked_by_contact']) for e in r['events'])})
    bricks=sum(x['bricks'] for x in rows); successes=sum(x['successes'] for x in rows)
    return {'benchmark':'mujoco-physics-v2','seeds':len(rows),'bricks':bricks,'successes':successes,'score':successes/bricks if bricks else 0,'picked_by_contact':sum(x['picked_by_contact'] for x in rows),'wall_seconds':round(time.perf_counter()-started,3),'rows':rows}

def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=200);p.add_argument('--output',type=Path,default=Path('/bulk2/runs/dryrun/benchmark/physics.json'));a=p.parse_args();x=run(range(a.seeds));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(x,indent=2));print(json.dumps({k:v for k,v in x.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
