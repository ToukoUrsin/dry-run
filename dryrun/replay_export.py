"""Export a compact, replayable trace of a real MuJoCo rollout.

The browser receives poses, never simulator state or a reimplemented IK model.
Large camera recordings stay beside the compressed trace in the run directory.
"""
from __future__ import annotations
import argparse, gzip, json, math, os, time
from datetime import datetime, timezone
from pathlib import Path
import imageio.v2 as imageio
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from .grasp_test import ARM_NAMES, ids, pad_force, solve_ik, controls
from .mjcf_station import ROOT, layout_bricks, station_xml

DT = 0.002
SAMPLE_STEPS = 17  # ~29.4 Hz, with the exact simulation time retained

def _quat(q):
    return [int(max(-32767, min(32767, round(float(v) * 32767)))) for v in q]

def _scene(model):
    bodies=[]
    for bid in range(model.nbody):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,bid) or f"body_{bid}"
        parent=int(model.body_parentid[bid])
        geoms=[]
        for gid in range(model.body_geomadr[bid],model.body_geomadr[bid]+model.body_geomnum[bid]):
            typ=int(model.geom_type[gid]); geom={"name":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid) or f"geom_{gid}","type":typ,"pos":[round(float(x),6) for x in model.geom_pos[gid]],"quat":_quat(model.geom_quat[gid]),"size":[round(float(x),6) for x in model.geom_size[gid]],"rgba":[round(float(x),4) for x in model.mat_rgba[model.geom_matid[gid]]] if model.geom_matid[gid]>=0 else [0.5,0.5,0.5,1]}
            if typ==int(mujoco.mjtGeom.mjGEOM_MESH): geom["mesh"]=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_MESH,int(model.geom_dataid[gid]))
            geoms.append(geom)
        bodies.append({"id":bid,"name":name,"parent":parent,"geoms":geoms})
    cameras=[]
    for cid in range(model.ncam):
        cameras.append({
            "name":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_CAMERA,cid),
            "id":cid,
            "pos":[round(float(x),6) for x in model.cam_pos[cid]],
            "quat":_quat(model.cam_quat[cid]),
            "fovy":round(float(model.cam_fovy[cid]),4),
        })
    meshes=[]
    for mid in range(model.nmesh):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_MESH,mid)
        meshes.append({"name":name,"file":name + ".stl" if name else None})
    return {"bodies":bodies,"cameras":cameras,"meshes":meshes,"mesh_root":"assets/SO101/Simulation/SO101/assets","units":"metres","pose_quantization":{"position_m":1e-4,"quaternion":1/32767}}

class Recorder:
    def __init__(self, model, data, writers):
        self.model,self.data,self.writers=model,data,writers; self.frames=[]; self.events=[]; self.last_sample=-1; self.render_count=0
        self.joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in (*ARM_NAMES,"gripper")]
    def sample(self, force=False):
        tick=int(round(self.data.time / (1/30)))
        if not force and tick==self.last_sample: return
        self.last_sample=tick
        poses=[]
        for bid in range(self.model.nbody): poses.append([int(round(float(v)*10000)) for v in self.data.xpos[bid]]+_quat(self.data.xquat[bid]))
        q=[float(self.data.qpos[self.model.jnt_qposadr[j]]) for j in self.joints]
        self.frames.append({"t":round(float(self.data.time),4),"poses":poses,"qpos":[round(v,5) for v in q[:5]],"gripper":round(q[5],5)})
        # MuJoCo still records every 30 Hz pose. Rendering every third pose
        # keeps export usable on a CPU box; the player treats the camera file
        # as a companion view and uses the recorded pose clock for sync.
        self.render_count += 1
        if self.render_count % 3 == 0 or force:
            for cam,writer in self.writers.items():
                writer.update_scene(self.data,cam); writer._writer.append_data(writer.render())
    def event(self,name,**extra):
        self.events.append({"t":round(float(self.data.time),4),"name":name,"frame":max(0,len(self.frames)-1),**extra})

def _set_ctrl(model,data,q,grip): controls(model,data,{n:float(v) for n,v in zip(ARM_NAMES,q)}|{"gripper":float(grip)})

def export(seed:int, out:Path, checkpoint="expert", sentence="good and defective bowls on the sides, inspection camera in front", count: int | None = None):
    out.mkdir(parents=True,exist_ok=True); xml=out/f"station-{seed}.xml"
    xml.write_text(station_xml(seed,sentence,count=count,robot_include=str((ROOT/'assets/SO101/Simulation/SO101/so101_dryrun.xml').resolve())))
    model=mujoco.MjModel.from_xml_path(str(xml)); data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    writers={}
    for key,filename in (("top_view","top-view.mp4"),("inspection_view","inspection-camera.mp4")):
        renderer=mujoco.Renderer(model,240,320)
        renderer._writer=imageio.get_writer(out/filename,fps=10,codec="libx264",quality=6,macro_block_size=1)
        writers[key]=renderer
    recorder=Recorder(model,data,writers); recorder.sample(force=True)
    brick_ids=[i for i in range(model.nbody) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or "").startswith("brick_")]
    specs=layout_bricks(seed,sentence,count=count); arm=ids(model); q=np.zeros(5); opened=1.7; successes=0; drops=0; failures=[]; start_time=time.perf_counter()
    def step(n,qtarget,grip):
        nonlocal q
        for _ in range(n):
            _set_ctrl(model,data,qtarget,grip); mujoco.mj_step(model,data); recorder.sample()
        q=np.array(qtarget,dtype=float)
    for body in brick_ids:
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body); idx=int(name.split("_")[1]); spec=specs[idx]
        recorder.event("approach",brick=idx,defect=spec.defect)
        start=data.xpos[body].copy(); q0,e0,ok0=solve_ik(model,data,[float(start[0]),float(start[1]),.075],arm,opened)
        if not ok0:
            recorder.event("failed",brick=idx,reason="approach_ik",ik_error_m=round(e0,5)); failures.append("approach_ik"); continue
        step(500,q0,opened); current=data.xpos[body].copy(); q1,e1,ok1=solve_ik(model,data,[float(current[0]),float(current[1]),.035],arm,opened)
        if not ok1:
            recorder.event("failed",brick=idx,reason="pick_ik",ik_error_m=round(e1,5)); failures.append("pick_ik"); continue
        step(650,q1,opened); current=data.xpos[body].copy(); q1,e1,ok1=solve_ik(model,data,[float(current[0]),float(current[1]),.035],arm,opened); step(300,q1,opened)
        touched=set(); peak=0.; close=opened
        for i in range(700):
            grip=opened+(1.0-opened)*min(1.,i/500); _set_ctrl(model,data,q1,grip); mujoco.mj_step(model,data); recorder.sample(); force,pads=pad_force(model,data); peak=max(peak,force); touched |= pads
            if len(touched)==2 and force>.5: close=grip; break
        picked=bool(len(touched)==2 and data.xpos[body,2]>.012)
        recorder.event("grasp" if picked else "failed",brick=idx,reason=None if picked else "no_two_pad_contact",pad_contacts=len(touched),peak_force_N=round(peak,3))
        if not picked:
            failures.append("grasp"); continue
        recorder.event("lift_start",brick=idx)
        inspect_y=-.24; qi,ei,oki=solve_ik(model,data,[.0,inspect_y,.19],arm,close)
        if not oki: recorder.event("failed",brick=idx,reason="inspection_ik"); failures.append("inspection_ik"); continue
        for i in range(900): stepq=q1+(qi-q1)*min(1.,i/650.); _set_ctrl(model,data,stepq,close); mujoco.mj_step(model,data); recorder.sample()
        q=qi; recorder.event("lift",brick=idx,z=round(float(data.xpos[body,2]),4)); recorder.event("at_inspection",brick=idx); step(90,qi,close); recorder.event("present",brick=idx); verdict="defective" if spec.defect!="none" else "good"; recorder.event("verdict",brick=idx,verdict=verdict)
        by=.30 if spec.defect!="none" else -.30; qb,eb,okb=solve_ik(model,data,[.35,by,.13],arm,close)
        if not okb: recorder.event("failed",brick=idx,reason="bowl_ik",ik_error_m=round(eb,5)); failures.append("bowl_ik"); continue
        for i in range(1000): stepq=qi+(qb-qi)*min(1.,i/700.); _set_ctrl(model,data,stepq,close); mujoco.mj_step(model,data); recorder.sample()
        q=qb; recorder.event("release",brick=idx); recorder.event("drop",brick=idx); step(180,qb,opened); final=data.xpos[body].copy(); in_bowl=bool(np.linalg.norm(final[:2]-np.array([.35,by]))<.14 and final[2]<.10)
        if in_bowl: successes+=1; recorder.event("in_bowl",brick=idx,xyz=[round(float(v),4) for v in final])
        else: drops+=1; failures.append("dropped"); recorder.event("dropped",brick=idx,xyz=[round(float(v),4) for v in final])
    recorder.sample(force=True)
    for r in writers.values(): r._writer.close(); r.close()
    bricks=len(brick_ids); outcome={"status":"success" if not failures else "failed","bricks":bricks,"correct":successes,"successes":successes,"drops":drops,"score":successes/bricks if bricks else 0,"failure_reason":failures[0] if failures else None,"cycle_time_s":round((time.perf_counter()-start_time)/max(1,bricks),3),"seed":seed}
    payload={"format":"dryrun-replay-v1","seed":seed,"checkpoint":checkpoint,"created_at":datetime.now(timezone.utc).isoformat(),"model":"SO-101 official MJCF","scene":_scene(model),"frames":recorder.frames,"events":recorder.events,"outcome":outcome,"videos":{"top":"top-view.mp4","inspection":"inspection-camera.mp4","fps":10,"source_pose_hz":30,"frame_stride":3}}
    replay=out/"replay.json.gz"
    with gzip.open(replay,"wt",encoding="utf-8",compresslevel=9) as fh: json.dump(payload,fh,separators=(",",":"))
    # Optional Tiger/Timescale write is deliberately best-effort: the replay remains useful offline.
    telemetry_status="not configured"
    if os.environ.get("DRYRUN_DB_URL"):
        try:
            from .telemetry import record_rollout, refresh
            base=datetime.now(timezone.utc); samples=[]
            for frame in recorder.frames: samples.append({"ts":base.timestamp()+frame["t"],"qpos":frame["qpos"]+[frame["gripper"]],"gripper":frame["gripper"]})
            record_rollout(samples,recorder.events,outcome,rollout_id=f"seed-{seed}-{int(time.time())}",seed=seed,checkpoint=checkpoint); refresh(); telemetry_status="written"
        except Exception as exc: telemetry_status=f"error: {exc}"
    manifest={"replay":str(replay),"replay_bytes":replay.stat().st_size,"frames":len(recorder.frames),"events":len(recorder.events),"videos":payload["videos"],"outcome":outcome,"telemetry":telemetry_status}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); return manifest

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=7); p.add_argument("--out",type=Path,required=True); p.add_argument("--checkpoint",default="expert"); a=p.parse_args(); os.environ.setdefault("MUJOCO_GL","egl"); print(json.dumps(export(a.seed,a.out,a.checkpoint),indent=2))

if __name__=="__main__": main()
