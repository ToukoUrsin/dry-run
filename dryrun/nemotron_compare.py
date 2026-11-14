"""Measure a Nemotron-suggested batch against an equal random batch."""
import json, random
from pathlib import Path
import mujoco
from .render import rollout
from .mjcf_station import station_xml, layout_bricks, ROOT

def score(seeds):
    rows=[]
    for seed in seeds:
        xml=Path('/tmp') / f'dryrun_compare_{seed}.xml'
        specs=layout_bricks(seed,'good and defective bowls on the sides, inspection camera in front')
        xml.write_text(station_xml(seed,'good and defective bowls on the sides, inspection camera in front',robot_include=str((ROOT/'assets/SO101/Simulation/SO101/so101_dryrun.xml').resolve())))
        model=mujoco.MjModel.from_xml_path(str(xml)); data=mujoco.MjData(model); mujoco.mj_forward(model,data)
        x=rollout(model,data,seed,specs=specs); rows.append({"seed":seed,"score":x["score"],"bricks":x["bricks"],"successes":x["successes"]})
    b=sum(x["bricks"] for x in rows); return {"seeds":list(seeds),"bricks":b,"successes":sum(x["successes"] for x in rows),"score":sum(x["successes"] for x in rows)/b}
def main():
    suggested=[101,102,103]; rng=random.Random(7); random_batch=[rng.randint(0,9999) for _ in suggested]
    out={"suggested":score(suggested),"random":score(random_batch),"suggestion_source":"/bulk2/runs/dryrun/m5/response.json"}
    open("/bulk2/runs/dryrun/m5/comparison.json","w").write(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
