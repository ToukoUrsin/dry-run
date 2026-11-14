"""Deterministic MuJoCo-inspired Dry Run station simulator.
Uses a compact kinematic scene model so the benchmark runs on CPU without external assets.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from dataclasses import asdict, dataclass
from pathlib import Path

COLORS=("red","blue","yellow","green","orange","white")
DEFECTS=("none","scribble","dot","sticker","chipped_stud","missing_stud","discolouring")
@dataclass
class Brick:
    id:int; size:str; color:str; x:float; y:float; z:float; defect:str; studs:int
@dataclass
class Scene:
    seed:int; lighting:float; camera_jitter:float; surface:str; bricks:list[Brick]

def scene(seed:int, heldout_defects=True)->Scene:
    r=random.Random(seed)
    n=r.randint(3,12)
    surface=r.choice(["matte_black","warm_white","wood","blue_grid"])
    lighting=round(r.uniform(.65,1.35),3); jitter=round(r.uniform(0,.035),4)
    out=[]
    for i in range(n):
        size=r.choice(["2x2","2x4"])
        # piles may overlap; deterministic geometry is visible to camera renderer
        x=round(r.uniform(-.28,.28),3); y=round(r.uniform(-.2,.2),3); z=round(.028+r.uniform(0,.07),3)
        d="none" if r.random()<.58 else r.choice(DEFECTS[1:])
        studs=4 if size=="2x2" else 8
        if d in ("missing_stud","chipped_stud"): studs-=1
        out.append(Brick(i,size,r.choice(COLORS),x,y,z,d,studs))
    return Scene(seed,lighting,jitter,surface,out)

def camera_frame(sc:Scene, view="top"):
    # image-like metadata; keeps vision interface explicit and avoids privileged labels
    return {"view":view,"width":640,"height":480,"lighting":sc.lighting,"jitter":sc.camera_jitter,
            "pixels_sha256":hashlib.sha256(json.dumps([(b.x,b.y,b.z,b.color,b.size) for b in sc.bricks]).encode()).hexdigest()}

def expert(sc:Scene):
    events=[]; good=bad=0; start=time.perf_counter()
    for b in sc.bricks:
        # scripted expert is allowed ground truth; inspector later replaces this branch
        accepted=b.defect=="none"
        bowl="good" if accepted else "defective"
        events += [{"event":"pick","brick_id":b.id},{"event":"inspect","brick_id":b.id,"label":b.defect},
                   {"event":"place","brick_id":b.id,"bowl":bowl}]
        good += accepted; bad += not accepted
    cycle=round(.35*len(sc.bricks)+.12,3)
    return {"seed":sc.seed,"bricks":len(sc.bricks),"correct":len(sc.bricks),"good":good,"defective":bad,
            "drops":0,"cycle_time_s":cycle,"events":events,"frames":{"top":camera_frame(sc),"inspection":camera_frame(sc,"inspection")},
            "ground_truth_used":True,"wall_s":round(time.perf_counter()-start,6)}

def run_benchmark(seeds=range(200), output=None):
    rows=[]
    for s in seeds:
        sc=scene(int(s)); r=expert(sc); rows.append({k:r[k] for k in ("seed","bricks","correct","drops","cycle_time_s")})
    total=sum(x["bricks"] for x in rows); correct=sum(x["correct"] for x in rows)
    result={"benchmark":"dryrun-v1","seeds":{"count":len(rows),"first":rows[0]["seed"],"last":rows[-1]["seed"]},"bricks":total,
            "correct":correct,"score":round(correct/total,6),"drops":sum(x["drops"] for x in rows),
            "mean_cycle_time_s":round(sum(x["cycle_time_s"] for x in rows)/len(rows),4),"rows":rows}
    if output: Path(output).write_text(json.dumps(result,indent=2))
    return result

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
 s=sub.add_parser("scene"); s.add_argument("seed",type=int); s.add_argument("--json",action="store_true")
 b=sub.add_parser("benchmark"); b.add_argument("--seeds",type=int,default=200); b.add_argument("--output",default="runs/benchmark.json")
 e=sub.add_parser("rollout"); e.add_argument("seed",type=int); e.add_argument("--output")
 a=p.parse_args()
 if a.cmd=="scene":
  x=asdict(scene(a.seed)); print(json.dumps(x,indent=2) if a.json else f"seed={x['seed']} bricks={len(x['bricks'])} surface={x['surface']}")
 elif a.cmd=="rollout":
  x=expert(scene(a.seed)); print(json.dumps(x,indent=2));
  if a.output: Path(a.output).write_text(json.dumps(x,indent=2))
 else:
  x=run_benchmark(range(a.seeds),a.output); print(json.dumps({k:v for k,v in x.items() if k!="rows"},indent=2))
if __name__=='__main__': main()
