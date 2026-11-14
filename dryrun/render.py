"""Run and render a real MuJoCo station rollout.

The station is generated as MJCF from a seed; bricks are free rigid bodies and
are only moved by MuJoCo contacts and actuator dynamics.
"""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from .mjcf_station import station_xml, ROOT, layout_bricks


def ik(model, data, target, q):
    sid = model.site("gripperframe").id; q = np.array(q, dtype=float).copy()
    for _ in range(180):
        data.qpos[:6] = q; mujoco.mj_forward(model, data)
        err = np.asarray(target) - data.site_xpos[sid]
        if np.linalg.norm(err) < .002: break
        J = np.zeros((3, model.nv)); mujoco.mj_jacSite(model, data, J, None, sid); J = J[:, :6]
        q += .45 * J.T @ np.linalg.solve(J @ J.T + .002*np.eye(3), err)
        q = np.clip(q, model.jnt_range[:6,0], model.jnt_range[:6,1])
    return q


def contact_with_arm(model, data, body_id):
    for i in range(data.ncon):
        c=data.contact[i]; a=model.geom_bodyid[c.geom1]; b=model.geom_bodyid[c.geom2]
        if body_id in (a,b) and (a != 0 and b != 0) and a != b:
            other=b if a==body_id else a
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,other) or ""
            if not name.startswith("brick_") and name not in ("table","good_bowl","defective_bowl"):
                return True
    return False


def _drive(model, data, q, grip, steps, writer):
    for step in range(steps):
        data.ctrl[:6]=q; data.ctrl[5]=grip; mujoco.mj_step(model,data)
        if writer and step % 60 == 0:
            for cam,r in writer.items(): r.update_scene(data,cam); _VIDEO.append_data(r.render())

def rollout(model, data, seed, writer=None, specs=None):
    brick_ids=[i for i in range(model.nbody) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '').startswith('brick_')]
    bowls={n:model.body(n).id for n in ('good_bowl','defective_bowl')}; q=np.zeros(6); events=[]; successes=0
    for body in brick_ids:
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body); adr=model.jnt_qposadr[model.body_jntadr[body]]
        start=data.qpos[adr:adr+3].copy(); q=ik(model,data,[float(start[0])+.06,float(start[1]),.055],q)
        _drive(model,data,q,1.7,180,writer)
        _drive(model,data,q,-.17,180,writer)
        picked=bool(data.xpos[body,2]>.035 and contact_with_arm(model,data,body))
        # The demonstrator knows the seeded label; this is the only privileged branch.
        spec = specs[int(name.split('_')[1])] if specs is not None else None
        target=np.array([.35,-.30 if spec is None or spec.defect == 'none' else .30,.22]); q=ik(model,data,target,q)
        _drive(model,data,q,-.17,240,writer)
        _drive(model,data,q,1.7,120,writer)
        final=data.xpos[body].copy(); in_bowl=bool(np.linalg.norm(final[:2]-target[:2])<.14 and final[2]<.10); ok=picked and in_bowl; successes+=int(ok)
        events.append({'brick':name,'defect':(spec.defect if spec is not None else 'unknown'),'picked_by_contact':picked,'final_xyz':[round(float(v),4) for v in final],'in_bowl':in_bowl,'success':ok})
        if writer:
            for cam,r in writer.items(): r.update_scene(data,cam); _VIDEO.append_data(r.render())
    return {'seed':seed,'bricks':len(brick_ids),'successes':successes,'score':successes/len(brick_ids) if brick_ids else 0,'events':events}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=7); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--json',type=Path)
    a=ap.parse_args(); os.environ.setdefault('MUJOCO_GL','egl'); a.out.parent.mkdir(parents=True,exist_ok=True)
    xml=a.out.with_suffix('.xml'); xml.write_text(station_xml(a.seed,'good and defective bowls on the sides, inspection camera in front',robot_include=str((ROOT/'assets/SO101/Simulation/SO101/so101_dryrun.xml').resolve())))
    model=mujoco.MjModel.from_xml_path(str(xml)); data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    writer=imageio.get_writer(a.out,fps=30,codec='libx264',quality=7); renderers={c:mujoco.Renderer(model,360,640) for c in ('free_view','top_view','inspection_view')}
    global _VIDEO; _VIDEO=writer
    for c,r in renderers.items(): r.update_scene(data,c); writer.append_data(r.render())
    result=rollout(model,data,a.seed,renderers,layout_bricks(a.seed,'good and defective bowls on the sides, inspection camera in front')); writer.close(); [r.close() for r in renderers.values()]
    result.update({'model':str(xml),'mujoco':mujoco.__version__,'video':str(a.out),'views':list(renderers)})
    if a.json: a.json.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
if __name__=='__main__': main()
