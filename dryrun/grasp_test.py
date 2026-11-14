"""Measured one-brick friction grasp test for the SO-101."""
from __future__ import annotations
import argparse, json, math
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from .mjcf_station import station_xml, ROOT

ARM_NAMES=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")

# Measured zero-pose geometry for the SO-101 pan plus planar chain.  The
# closed-form branch below is deterministic; MuJoCo only supplies the bounded
# millimetre correction and the independent cross-check.
SHOULDER_PIVOT = np.array([0.0388353, 0.0, 0.0624], dtype=float)
SHOULDER_OFFSET = np.array([0.0303992, -0.0182778, 0.0542], dtype=float)
UPPER_LINK = np.array([0.028, 0.0, 0.11257], dtype=float)
FOREARM_LINK = np.array([0.1349, 0.0, 0.0052], dtype=float)
WRIST_TO_GRIPPERFRAME = np.array([0.1592274, 0.0182665, -0.0079013], dtype=float)
WRIST_TO_PAD_MIDPOINT = np.array([0.0930000, 0.0184363, 0.0013179], dtype=float)


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _analytic_seed(target: np.ndarray, model: mujoco.MjModel, distal_offset: np.ndarray) -> np.ndarray | None:
    """Select a pan/two-link branch from closed-form planar geometry."""
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        return None
    dxy = target[:2] - SHOULDER_PIVOT[:2]
    if np.linalg.norm(dxy) < 1e-9:
        return None
    # MuJoCo's positive pan rotates the radial plane by -q1.
    q1 = float(np.clip(-math.atan2(float(dxy[1]), float(dxy[0])),
                       model.jnt_range[0, 0], model.jnt_range[0, 1]))
    local = _rz(q1) @ (target - SHOULDER_PIVOT)
    # The wrist pitch/roll are free for a position-only target.  Remove the
    # measured distal length along the current radial direction; this keeps
    # the two-link triangle inside the reachable annulus for all approach
    # pitches, while the exact wrist offsets are solved in the correction.
    distal_length = float(np.linalg.norm(distal_offset))
    candidates = []
    l1 = float(np.linalg.norm(UPPER_LINK[[0, 2]])); l2 = float(np.linalg.norm(FOREARM_LINK[[0, 2]]))
    alpha = math.atan2(float(UPPER_LINK[0]), float(UPPER_LINK[2]))
    beta = math.atan2(float(FOREARM_LINK[0]), float(FOREARM_LINK[2]))
    lo = np.asarray(model.jnt_range[:5, 0], dtype=float); hi = np.asarray(model.jnt_range[:5, 1], dtype=float)
    for distal_sign in (1.0, -1.0):
        wrist_local = local - SHOULDER_OFFSET - np.array([distal_sign * distal_length, 0.0, 0.0])
        x, z = float(wrist_local[0]), float(wrist_local[2]); r = math.hypot(x, z)
        if r > l1 + l2 or r < abs(l1 - l2) or r < 1e-8: continue
        gamma = math.atan2(x, z)
        c1 = float(np.clip((l1*l1 + r*r - l2*l2) / (2*l1*r), -1.0, 1.0))
        c2 = float(np.clip((l2*l2 + r*r - l1*l1) / (2*l2*r), -1.0, 1.0))
        phi, psi = math.acos(c1), math.acos(c2)
        for sign in (1.0, -1.0):
            q2 = gamma + sign * phi - alpha
            q3 = gamma - sign * psi - beta - q2
            q = np.array([q1, q2, q3, 0.0, 0.0], dtype=float)
            if np.all(q >= lo) and np.all(q <= hi): candidates.append(q)
    if not candidates:
        return None
    return min(candidates, key=lambda q: float(np.linalg.norm(q[1:3])))


def closed_form_ik(model: mujoco.MjModel, data: mujoco.MjData, target,
                    grip_q: float = 1.7, *, point: str = "pad_midpoint",
                    wrist_roll: float = 1.3,
                    tolerance: float = .0015):
    """Solve one target from geometry, with one bounded exact correction.

    Unlike ``solve_restarted`` this never searches restart seeds.  The
    pan/two-link solution chooses the branch, then at most 18 damped Newton
    updates project it onto the exact MuJoCo pad or gripper-frame geometry.
    """
    target = np.asarray(target, dtype=float).reshape(3)
    distal_offset = WRIST_TO_GRIPPERFRAME if point == "gripperframe" else WRIST_TO_PAD_MIDPOINT
    seed = _analytic_seed(target, model, distal_offset)
    if seed is None:
        return np.zeros(5, dtype=float), float("inf"), False
    qa = qadr(model, ids(model)); grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"); ga = int(model.jnt_qposadr[grip])
    qsave = data.qpos.copy(); vsave = data.qvel.copy()
    try:
        seed[-1] = float(np.clip(wrist_roll, model.jnt_range[4, 0], model.jnt_range[4, 1]))
        for k, adr in enumerate(qa): data.qpos[adr] = seed[k]
        data.qpos[ga] = float(grip_q); mujoco.mj_forward(model, data)
        def value():
            if point == "gripperframe":
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
                return data.site_xpos[sid].copy()
            a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_grip_pad")
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_grip_pad")
            return ((data.geom_xpos[a] + data.geom_xpos[b]) * .5).copy()
        eps = 2e-5
        for _ in range(18):
            p = value(); err = target - p
            if float(np.linalg.norm(err)) <= tolerance:
                return np.asarray([data.qpos[a] for a in qa]), float(np.linalg.norm(err)), True
            J = np.empty((3, 5), dtype=float); base = np.asarray([data.qpos[a] for a in qa], dtype=float)
            for k, adr in enumerate(qa):
                data.qpos[adr] = base[k] + eps; mujoco.mj_forward(model, data); J[:, k] = (value() - p) / eps
                data.qpos[adr] = base[k]
            mujoco.mj_forward(model, data)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-7 * np.eye(3), err)
            q = np.clip(base + .9 * dq, model.jnt_range[:5, 0], model.jnt_range[:5, 1])
            for k, adr in enumerate(qa): data.qpos[adr] = q[k]
            data.qpos[ga] = float(grip_q); mujoco.mj_forward(model, data)
        err = float(np.linalg.norm(target - value()))
        return np.asarray([data.qpos[a] for a in qa]), err, bool(np.isfinite(err) and err <= tolerance)
    finally:
        data.qpos[:] = qsave; data.qvel[:] = vsave; mujoco.mj_forward(model, data)

def make_empty_table(path: Path):
    path.write_text(station_xml(901,"single brick, bowls behind the arm, inspection camera in front",count=1,robot_include=str((ROOT/'assets/SO101/Simulation/SO101/so101_dryrun.xml').resolve())))
    tree=ET.parse(path); world=tree.getroot().find('worldbody')
    for body in list(world):
        if body.tag=='body' and body.attrib.get('name') in {'good_bowl','defective_bowl'}: world.remove(body)
    brick=next(b for b in world if b.tag=='body' and b.attrib.get('name')=='brick_0')
    # The calibration proof is deliberately a 2x2 Duplo (31.8 mm short side),
    # so the jaw opening is measured against the requested grasp dimension.
    brick.find('geom').set('size','0.0160 0.0160 0.00960')
    for geom in list(brick):
        if geom.tag=='geom' and '_stud_' in geom.attrib.get('name','') and int(geom.attrib['name'].rsplit('_',1)[1]) >= 4:
            brick.remove(geom)
    tree.write(path,encoding='unicode')

def ids(m): return [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ARM_NAMES]
def qadr(m,jids): return [int(m.jnt_qposadr[j]) for j in jids]
def pad_ids(m): return (mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'fixed_grip_pad'),mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'moving_grip_pad'))
def pad_mid(m,d):
    a,b=pad_ids(m); return (d.geom_xpos[a]+d.geom_xpos[b])/2

def solve_ik(m,d,target,jids,grip_q,initial=None,tol=.004,pad_angle=None):
    """Solve for the pad midpoint and a top-down pad orientation."""
    qa=qadr(m,jids); grip=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'gripper'); ga=int(m.jnt_qposadr[grip]); qsave=d.qpos.copy(); eps=2e-5
    if initial is not None:
        for k,aadr in enumerate(qa):
            d.qpos[aadr]=np.clip(float(initial[k]),m.jnt_range[jids[k],0],m.jnt_range[jids[k],1])
    p0=np.array([d.qpos[a] for a in qa],dtype=float)
    a,b=pad_ids(m)
    def vals():
        out=np.r_[pad_mid(m,d), d.geom_xpos[a][2]-d.geom_xpos[b][2]]
        if pad_angle is not None:
            line=d.geom_xpos[a]-d.geom_xpos[b]
            out=np.r_[out, math.sin(math.atan2(float(line[1]),float(line[0]))-pad_angle)]
        return out
    for _ in range(320):
        d.qpos[ga]=grip_q; mujoco.mj_forward(m,d); v=vals(); e=np.r_[np.asarray(target)-v[:3], -v[3]]
        if pad_angle is not None: e=np.r_[e,-v[4]]
        if not np.all(np.isfinite(e)):
            break
        if np.linalg.norm(e)<tol:
            result=np.array([d.qpos[a] for a in qa]); err=float(np.linalg.norm(e)); d.qpos[:]=qsave; mujoco.mj_forward(m,d); return result,err,True
        J=np.empty((len(e),len(qa)))
        for k,aadr in enumerate(qa):
            old=d.qpos[aadr]; d.qpos[aadr]=old+eps; mujoco.mj_forward(m,d); J[:,k]=(vals()-v)/eps; d.qpos[aadr]=old
        dq=J.T@np.linalg.solve(J@J.T+2e-4*np.eye(len(e)),e); dq-=.015*(np.array([d.qpos[aadr] for aadr in qa])-p0)
        for k,aadr in enumerate(qa): d.qpos[aadr]=np.clip(d.qpos[aadr]+.65*dq[k],m.jnt_range[jids[k],0],m.jnt_range[jids[k],1])
    mujoco.mj_forward(m,d); v=vals(); result=np.array([d.qpos[aadr] for aadr in qa]); residual=np.r_[np.asarray(target)-v[:3],-v[3]]; residual=np.r_[residual,-v[4]] if pad_angle is not None else residual; err=float(np.linalg.norm(residual)); d.qpos[:]=qsave; mujoco.mj_forward(m,d); return result,err,bool(np.isfinite(err) and err<tol)

def solve_restarted(m,d,target,jids,grip_q,tol=.004,pad_angle=None,offsets=None):
    """Try equivalent wrist-roll approaches before declaring a target unreachable."""
    qa=qadr(m,jids); current=np.array([d.qpos[a] for a in qa],dtype=float); best=None
    for offset in ((0., math.pi/2, -math.pi/2, math.pi) if offsets is None else offsets):
        seed=current.copy(); seed[-1]=np.clip(seed[-1]+offset,m.jnt_range[jids[-1],0],m.jnt_range[jids[-1],1])
        q,e,ok=solve_ik(m,d,target,jids,grip_q,initial=seed,tol=tol,pad_angle=pad_angle)
        if best is None or e<best[1]: best=(q,e,ok)
        if ok: return q,e,True
    return best

def safe_state(m,d,bid):
    if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
        return False
    return float(d.xpos[bid,2]) >= -.02 and float(np.linalg.norm(d.qvel)) <= 50.0

def failure(pos,yaw,stage,err=float('inf')):
    return {'x':pos[0],'y':pos[1],'yaw':yaw,'ik_ok':[False,False,False],
            'ik_error_m':[float(err),float(err),float(err)],'pad_contacts':0,
            'peak_pad_normal_force_N':0.0,'prelift_z':None,'lift_z':None,
            'lifted_10mm':False,'final_pad_midpoint_error_m':None,
            'failed_stage':stage}

def controls(m,d,targets):
    for a in range(m.nu):
        n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,a)
        if n in targets: d.ctrl[a]=targets[n]

def pad_force(m,d):
    pads=set(pad_ids(m)); brick=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'brick_0'); force=np.zeros(6); total=0.; touched=set()
    for i in range(d.ncon):
        c=d.contact[i]; pad = c.geom1 if c.geom1 in pads else (c.geom2 if c.geom2 in pads else -1)
        if pad < 0: continue
        other = c.geom2 if pad == c.geom1 else c.geom1
        if m.geom_bodyid[other] != brick: continue
        mujoco.mj_contactForce(m,d,i,force); total += abs(float(force[0])); touched.add(pad)
    return total,touched

def _trial_once(m,base,pos,yaw,roll_offset=None):
    d=mujoco.MjData(m); d.qpos[:]=base.qpos; d.qvel[:]=0
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'brick_0'); bj=m.body_jntadr[bid]; ba=int(m.jnt_qposadr[bj]); d.qpos[ba:ba+3]=[pos[0],pos[1],.0097]; d.qpos[ba+3:ba+7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
    arm=ids(m); grip=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'gripper'); ga=int(m.jnt_qposadr[grip]); gopen=1.7; d.qpos[ga]=gopen; mujoco.mj_forward(m,d)
    target=lambda z: np.array([pos[0],pos[1],z])
    offsets=None if roll_offset is None else (roll_offset,)
    qa,ea,oka=solve_restarted(m,d,target(.075),arm,gopen,offsets=offsets)
    if not oka: return failure(pos,yaw,'approach',ea)
    for _ in range(500): controls(m,d,{n:float(q) for n,q in zip(ARM_NAMES,qa)}|{'gripper':gopen}); mujoco.mj_step(m,d)
    if not safe_state(m,d,bid): return failure(pos,yaw,'approach_unstable',ea)
    # The approach can nudge a free brick by a few centimetres.  Re-read its
    # measured pose before the pick solve; this is the expert's retry, not a
    # teleport or weld.
    current_xy=d.xpos[bid,:2].copy(); target=lambda z: np.array([current_xy[0],current_xy[1],z])
    qp,ep,okp=solve_restarted(m,d,target(.034),arm,gopen,tol=.0025,offsets=offsets)
    if not okp: return failure(pos,yaw,'lower_approach',ep)
    for _ in range(650): controls(m,d,{n:float(q) for n,q in zip(ARM_NAMES,qp)}|{'gripper':gopen}); mujoco.mj_step(m,d)
    # Re-center once more after the low approach: a free brick may have
    # rolled under the pad by a few millimetres before contact.
    current_xy=d.xpos[bid,:2].copy(); target=lambda z: np.array([current_xy[0],current_xy[1],z])
    qp2,ep2,okp2=solve_restarted(m,d,target(.034),arm,gopen,tol=.0025,offsets=offsets)
    if not okp2: return failure(pos,yaw,'recenter',ep2)
    for _ in range(300): controls(m,d,{n:float(q) for n,q in zip(ARM_NAMES,qp2)}|{'gripper':gopen}); mujoco.mj_step(m,d)
    qp=qp2; ep=max(ep,ep2); okp=okp and okp2
    peak=0.; touched=set(); close_g=gopen
    for step in range(700):
        g=gopen+(1.0-gopen)*min(1.,step/500)
        controls(m,d,{n:float(q) for n,q in zip(ARM_NAMES,qp)}|{'gripper':g}); mujoco.mj_step(m,d); f,t=pad_force(m,d); peak=max(peak,f); touched|=t
        if not safe_state(m,d,bid): return failure(pos,yaw,'close_unstable',peak)
        if len(t)==2 and f>0.5:
            close_g=g; break
    if len(touched) < 2:
        # Closed-loop retry: approach in small jaw increments, stopping on
        # brick-pad contacts rather than forcing a precomputed angle.
        for retry_g in (1.5, 1.3, 1.1, .9, .7):
            for _ in range(120):
                controls(m,d,{n:float(q) for n,q in zip(ARM_NAMES,qp)}|{'gripper':retry_g}); mujoco.mj_step(m,d); f,t=pad_force(m,d); peak=max(peak,f); touched|=t
                if not safe_state(m,d,bid): return failure(pos,yaw,'retry_unstable',peak)
            if len(touched)==2:
                close_g=retry_g; break
    if len(touched) == 2:
        close_g=min(close_g,0.75)
        for _ in range(120):
            controls(m,d,{n:float(q) for n,q in zip(ARM_NAMES,qp)}|{'gripper':close_g}); mujoco.mj_step(m,d)
            f,t=pad_force(m,d); peak=max(peak,f); touched|=t
    if len(touched) < 2:
        return {'x':pos[0],'y':pos[1],'yaw':yaw,'ik_ok':[oka,okp,okp2], 'ik_error_m':[ea,ep,ep2], 'pad_contacts':len(touched), 'peak_pad_normal_force_N':round(peak,5), 'prelift_z':round(float(d.xpos[bid,2]),5), 'lift_z':round(float(d.xpos[bid,2]),5), 'lifted_10mm':False, 'final_pad_midpoint_error_m':None, 'failed_stage':'contact'}
    z0=float(d.xpos[bid,2]); ql,el,okl=solve_restarted(m,d,target(.125),arm,close_g,tol=.0025,offsets=offsets)
    if not okl:
        return {'x':pos[0],'y':pos[1],'yaw':yaw,'ik_ok':[oka,okp,okl], 'ik_error_m':[ea,ep,el], 'pad_contacts':len(touched), 'peak_pad_normal_force_N':round(peak,5), 'prelift_z':round(z0,5), 'lift_z':round(z0,5), 'lifted_10mm':False, 'final_pad_midpoint_error_m':None, 'failed_stage':'lift_ik'}
    for step in range(1000):
        lift_q=qp+(ql-qp)*min(1.,step/650.)
        controls(m,d,{n:float(q) for n,q in zip(ARM_NAMES,lift_q)}|{'gripper':close_g}); mujoco.mj_step(m,d); f,t=pad_force(m,d); peak=max(peak,f); touched|=t
        if not safe_state(m,d,bid): return failure(pos,yaw,'lift_unstable',peak)
    z1=float(d.xpos[bid,2])
    return {'x':pos[0],'y':pos[1],'yaw':yaw,'ik_ok':[oka,okp,okl],'ik_error_m':[ea,ep,el],'pad_contacts':len(touched),'peak_pad_normal_force_N':round(peak,5),'prelift_z':round(z0,5),'lift_z':round(z1,5),'lifted_10mm':bool(z1>z0+.010),'final_pad_midpoint_error_m':float(np.linalg.norm(pad_mid(m,d)-target(.125))),'failed_stage':None}

def trial(m,base,pos,yaw):
    """Retry a symmetric brick with equivalent jaw orientations."""
    attempts=[_trial_once(m,base,pos,yaw,off) for off in (0.,math.pi/2,-math.pi/2,math.pi)]
    for row in attempts:
        if row.get('lifted_10mm'): return row
    def score(row):
        z=row.get('lift_z'); return (int(row.get('pad_contacts',0)==2), float(z) if z is not None else -1e9, -float(row.get('peak_pad_normal_force_N',0)))
    return max(attempts,key=score)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='/bulk2/runs/dryrun/m1/grasp_trials.jsonl'); p.add_argument('--xml',default='/bulk2/runs/dryrun/m1/grasp_empty.xml'); a=p.parse_args(); xml=Path(a.xml); xml.parent.mkdir(parents=True,exist_ok=True); make_empty_table(xml)
    m=mujoco.MjModel.from_xml_path(str(xml)); base=mujoco.MjData(m); positions=[(.24,-.14)]*5+[(.24,-.12)]*4+[(.24,-.10)]; yaws=[0,.2,.4,.6,.8,0,.2,.4,.8,0]; rows=[trial(m,base,pos,yaw) for pos,yaw in zip(positions,yaws)]
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows)); print(json.dumps({'trials':len(rows),'lifted':sum(r['lifted_10mm'] for r in rows),'both_pad_contact':sum(r['pad_contacts']==2 for r in rows),'out':str(out)},indent=2))
if __name__=='__main__': main()
