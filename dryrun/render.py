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
from .env import TIERS
from .grasp_test import ARM_NAMES, ids, pad_force, safe_state
from .closed_form_ik import IKError, ik_pad_pose, nearest_symmetric_yaw
from .defects import INSPECTION_FRAME_SIZE, INSPECTION_PRESENT_POSE, configure_inspection_camera, render_inspection_frame


def ik(model, data, target, q):
    """Compatibility wrapper for callers that use the site-based solver."""
    sid = model.site("gripperframe").id; q = np.array(q, dtype=float).copy()
    for _ in range(180):
        data.qpos[:6] = q; mujoco.mj_forward(model, data); err=np.asarray(target)-data.site_xpos[sid]
        if np.linalg.norm(err)<.002: break
        J=np.zeros((3,model.nv));mujoco.mj_jacSite(model,data,J,None,sid);J=J[:,:6]
        q += .45*J.T@np.linalg.solve(J@J.T+.002*np.eye(3),err);q=np.clip(q,model.jnt_range[:6,0],model.jnt_range[:6,1])
    return q


def _solve_target(model, data, target, grip, *, tolerance=.004, wrist_roll=0.0):
    """Solve physical pad targets with the finite closed-form path."""
    del model, data  # retained in the rollout call contract
    pitches = [*range(35, -1, -1), *range(-1, -36, -1)]
    for degrees in pitches:
        try:
            result = ik_pad_pose(
                target,
                approach_pitch=float(np.deg2rad(degrees)),
                wrist_roll=float(wrist_roll),
                enforce_limits=True,
                grip_q=float(grip),
            )
        except IKError:
            continue
        if result.error <= tolerance:
            return result.q, float(result.error), True
    return np.zeros(5, dtype=float), float("inf"), False


def contact_with_arm(model, data, body_id):
    for i in range(data.ncon):
        c=data.contact[i]; a=model.geom_bodyid[c.geom1]; b=model.geom_bodyid[c.geom2]
        if body_id in (a,b) and (a != 0 and b != 0) and a != b:
            other=b if a==body_id else a
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,other) or ""
            if not name.startswith("brick_") and name not in ("table","good_bowl","defective_bowl"):
                return True
    return False


_ACTIVE_RECORD = None
_ACTIVE_RECORDERS = None
_ACTIVE_TICK = 0
_ACTIVE_VERDICT = 0


def _record_frame(model, data, action):
    """Sample camera pixels and joints at the public 30 Hz boundary."""
    global _ACTIVE_TICK
    if _ACTIVE_RECORD is None or _ACTIVE_RECORDERS is None:
        return
    _ACTIVE_TICK += 1
    sample_every = int(getattr(_ACTIVE_RECORD, "sample_every", 17))
    if sample_every <= 0:
        return
    if _ACTIVE_TICK % sample_every:
        return
    obs = {}
    for cam, renderer in _ACTIVE_RECORDERS.items():
        if cam == "inspection_view":
            # The policy and inspector receive the same fixed brick crop;
            # only the top/free context retains the full arm scene.
            obs["inspection"] = render_inspection_frame(model, data, renderer)
        else:
            renderer.update_scene(data, cam)
            obs[cam.replace("_view", "")] = renderer.render().copy()
    observation = {"joints": np.asarray(data.qpos[:6], dtype=np.float32).copy()}
    if "top" in obs:
        observation["top"] = obs["top"]
    if "inspection" in obs:
        observation["inspection"] = obs["inspection"]
    payload = {
        "t": float(data.time),
        "observation": observation,
        "action": np.asarray(action, dtype=np.float32).copy(),
        "verdict": int(_ACTIVE_VERDICT),
        "events": [],
    }
    if callable(_ACTIVE_RECORD):
        _ACTIVE_RECORD(payload)
    elif hasattr(_ACTIVE_RECORD, "record"):
        _ACTIVE_RECORD.record(payload)


def _record_initial_frame(model, data):
    """Emit one camera pair before the first arm command when requested."""
    if _ACTIVE_RECORD is None or _ACTIVE_RECORDERS is None:
        return
    if not bool(getattr(_ACTIVE_RECORD, "capture_initial", False)):
        return
    obs = {}
    for cam, renderer in _ACTIVE_RECORDERS.items():
        if cam == "inspection_view":
            obs["inspection"] = render_inspection_frame(model, data, renderer)
        else:
            renderer.update_scene(data, cam)
            obs[cam.replace("_view", "")] = renderer.render().copy()
    payload = {"t": float(data.time), "observation": obs,
               "action": np.zeros(6, dtype=np.float32), "verdict": 0, "events": []}
    if callable(_ACTIVE_RECORD):
        _ACTIVE_RECORD(payload)
    elif hasattr(_ACTIVE_RECORD, "record"):
        _ACTIVE_RECORD.record(payload)


def _drive(model, data, q, grip, steps, writer):
    for step in range(steps):
        controls = {n: float(v) for n, v in zip(ARM_NAMES, q)} | {"gripper": float(grip)}
        for a in range(model.nu):
            n=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a)
            if n in controls: data.ctrl[a]=controls[n]
        mujoco.mj_step(model,data)
        _record_frame(model, data, np.r_[np.asarray(q, dtype=np.float32), float(grip)])
        if writer and step % 60 == 0:
            for cam,r in writer.items():
                if cam == "inspection_view":
                    frame = render_inspection_frame(model, data, r)
                else:
                    r.update_scene(data, cam)
                    frame = r.render().copy()
                if "_VIDEO" in globals(): _VIDEO.append_data(frame)


def rollout(model, data, seed, writer=None, specs=None, inspector=None, record=None):
    """Run a contact-only expert, replanning from the live state per brick."""
    global _ACTIVE_RECORD, _ACTIVE_RECORDERS, _ACTIVE_TICK, _ACTIVE_VERDICT
    # Training/photo renders and rollout inspection frames share this fixed
    # close camera.  Move it before arm motion and recompute kinematics once.
    configure_inspection_camera(model)
    mujoco.mj_forward(model, data)
    brick_ids=[i for i in range(model.nbody) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '').startswith('brick_')]
    old_record, old_recorders, old_tick, old_verdict = _ACTIVE_RECORD, _ACTIVE_RECORDERS, _ACTIVE_TICK, _ACTIVE_VERDICT
    _ACTIVE_RECORD, _ACTIVE_TICK, _ACTIVE_VERDICT = record, 0, 0
    if record is not None:
        _ACTIVE_RECORDERS = {}
        if not bool(getattr(record, "thumbnail_only", False)):
            _ACTIVE_RECORDERS["top_view"] = mujoco.Renderer(model, 64, 64)
            if bool(getattr(record, "include_inspection", True)):
                _ACTIVE_RECORDERS["inspection_view"] = mujoco.Renderer(model, 64, 64)
        elif hasattr(record, "__call__"):
            # A benchmark export needs one truthful top-camera thumbnail but
            # does not need to render duplicate pixels at every physics tick.
            thumb_renderer = mujoco.Renderer(model, 64, 64)
            thumb_renderer.update_scene(data, "top_view")
            thumb = thumb_renderer.render().copy()
            thumb_renderer.close()
            record({
                "t": float(data.time),
                "observation": {"top": thumb, "joints": np.asarray(data.qpos[:6], dtype=np.float32).copy()},
                "action": np.zeros(6, dtype=np.float32),
                "verdict": 0,
                "events": [],
            })
        if bool(getattr(record, "capture_initial", False)) and _ACTIVE_RECORDERS:
            _record_initial_frame(model, data)
    arm=ids(model); events=[]; successes=0; gopen=1.7
    wrist_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_roll")
    wrist_mid = float(model.jnt_range[wrist_joint, 0] + model.jnt_range[wrist_joint, 1]) / 2.0
    bowl_geoms = [i for i in range(model.ngeom) if ((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith(('good_bowl_', 'defective_bowl_')))]
    bowl_collision = [(i, int(model.geom_contype[i]), int(model.geom_conaffinity[i])) for i in bowl_geoms]
    pad_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ('fixed_grip_pad', 'moving_grip_pad')]
    pad_collision = [(i, int(model.geom_contype[i]), int(model.geom_conaffinity[i])) for i in pad_geoms]
    def set_bowl_collision(enabled):
        for i, ct, ca in bowl_collision:
            model.geom_contype[i] = ct if enabled else 0
            model.geom_conaffinity[i] = ca if enabled else 0
    def set_pad_collision(enabled):
        for i, ct, ca in pad_collision:
            model.geom_contype[i] = ct if enabled else 0
            model.geom_conaffinity[i] = ca if enabled else 0
    # Keep the inspector input at the canonical 384 px even when a caller is
    # also writing a low-resolution replay video.
    inspection_renderer = mujoco.Renderer(model, INSPECTION_FRAME_SIZE, INSPECTION_FRAME_SIZE) if inspector is not None else None
    remaining=set(brick_ids)
    while remaining:
        body=min(remaining,key=lambda b: float(np.linalg.norm(data.xpos[b,:2])))
        remaining.remove(body)
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body); idx=int(name.split('_')[1])
        spec=specs[idx] if specs is not None and idx < len(specs) else None; start_z=float(data.xpos[body,2]); best=None
        # The wrist roll is the jaw-yaw branch.  The old path added a
        # side-dependent +/-1.3 radian bias after choosing the symmetric
        # branch; that pushed some valid bricks outside the wrist-roll limits
        # (and made the supposedly measured scene fail before contact).  Pick
        # the equivalent branch nearest the mechanical midpoint, as the
        # closed-form solver and reach-map contract specify.
        brick_roll = 0.0 if spec is None else nearest_symmetric_yaw(
            spec.yaw, spec.size, reference=wrist_mid
        )
        if record is not None:
            event = {"type": "cycle_start", "brick": name, "t": float(data.time)}
            if callable(record): record(event)
            elif hasattr(record, "cycle_start"): record.cycle_start(name)
        for attempt in range(6):
            set_bowl_collision(False)
            set_pad_collision(False)
            start=data.xpos[body].copy(); approach_q,ea,oka=_solve_target(model,data,[float(start[0]),float(start[1]),.075],gopen,wrist_roll=brick_roll)
            if not oka: best={'failed_stage':'approach_ik','ik_error_m':ea}; continue
            _drive(model,data,approach_q,gopen,500,writer)
            if not safe_state(model,data,body): best={'failed_stage':'approach_unstable'}; continue
            set_pad_collision(True)
            xy=data.xpos[body,:2].copy(); q,ep,okp=_solve_target(model,data,[float(xy[0]),float(xy[1]),.034],gopen,tolerance=.0025,wrist_roll=brick_roll)
            if not okp: best={'failed_stage':'lower_approach_ik','ik_error_m':ep}; continue
            _drive(model,data,q,gopen,650,writer)
            touched=set(); peak=0.; close_g=gopen
            for step in range(700):
                g=gopen+(1.-gopen)*min(1.,step/500.)
                _drive(model,data,q,g,1,None); f,t=pad_force(model,data); peak=max(peak,f); touched |= t
                if not safe_state(model,data,body): break
                if len(t)==2 and f>.5: close_g=g; break
            if len(touched)<2 or not safe_state(model,data,body): best={'failed_stage':'contact','pad_contacts':len(touched),'peak_pad_force_N':peak}; continue
            close_g=min(close_g,.75)
            for _ in range(120):
                _drive(model,data,q,close_g,1,None); f,t=pad_force(model,data); peak=max(peak,f); touched |= t
            # The lower pose only establishes both pad contacts.  Lift from
            # the live brick position with a fresh closed-form solve; the
            # previous implementation kept commanding the lower pose, so a
            # brick could be squeezed without ever being lifted.
            lift_xy=data.xpos[body,:2].copy()
            lift_q,el,okl=_solve_target(model,data,[float(lift_xy[0]),float(lift_xy[1]),.075],close_g,wrist_roll=brick_roll)
            if okl:
                _drive(model,data,lift_q,close_g,500,writer)
            picked=bool(len(touched)==2 and data.xpos[body,2]>start_z+.005)
            if not picked: best={'failed_stage':'lift_contact','pad_contacts':len(touched),'peak_pad_force_N':peak}; continue
            set_bowl_collision(True)
            inspect=INSPECTION_PRESENT_POSE.copy(); qi,ei,oki=_solve_target(model,data,inspect,close_g)
            if not oki: best={'failed_stage':'inspection_ik','ik_error_m':ei}; continue
            for step in range(900):
                qq=q+(qi-q)*min(1.,step/700.); _drive(model,data,qq,close_g,1,writer)
            for _ in range(180): _drive(model,data,qi,close_g,1,writer)
            verdict = None
            if inspector is not None:
                result = inspector(render_inspection_frame(model, data, inspection_renderer))
                verdict = result['verdict']; events.append({'brick': name, 'event': 'verdict', 'verdict': verdict, 'score': float(result['score']), 'threshold': float(result['threshold'])})
                _ACTIVE_VERDICT = 1 if verdict == 'good' else -1
            is_defective = verdict == 'defective' if verdict is not None else (spec is None or spec.defect != 'none')
            bowl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "defective_bowl" if is_defective else "good_bowl")
            bowl=np.array([float(data.xpos[bowl_id, 0]), float(data.xpos[bowl_id, 1]), .125]); qb,eb,okb=_solve_target(model,data,bowl,close_g)
            if not okb: best={'failed_stage':'bowl_ik','ik_error_m':eb}; continue
            for step in range(1100):
                qq=qi+(qb-qi)*min(1.,step/800.); _drive(model,data,qq,close_g,1,writer)
            # Release fully and give the brick enough real physics time to
            # settle inside the bowl before judging the drop.
            _drive(model,data,qb,gopen,1500,writer)
            qaway,eaway,okaway=_solve_target(model,data,[.28,0.,.24],gopen)
            if okaway: _drive(model,data,qaway,gopen,500,writer)
            final=data.xpos[body].copy(); in_bowl=bool(np.linalg.norm(final[:2]-bowl[:2])<.14 and final[2]<.10); ok=bool(in_bowl)
            best={'picked_by_contact':True,'pad_contacts':len(touched),'peak_pad_force_N':peak,'final_xyz':[round(float(v),4) for v in final],'in_bowl':in_bowl,'success':ok,'attempts':attempt+1}
            if ok:
                successes += 1
                break
            best['failed_stage'] = 'drop'
            # A missed drop is a physical failure, but the expert gets the
            # same bounded retry opportunity as a missed grasp.
            continue
        if best is None: best={'failed_stage':'no_attempt'}
        best.setdefault('picked_by_contact',False); best.setdefault('pad_contacts',0); best.setdefault('peak_pad_force_N',0.0); best.setdefault('final_xyz',[round(float(v),4) for v in data.xpos[body]]); best.setdefault('in_bowl',False); best.setdefault('success',False)
        cycle_ok = bool(best.get("success", False))
        events.append({'brick':name,'t':float(data.time),'defect':spec.defect if spec else 'unknown',**best})
        if record is not None:
            event = {"type": "cycle_end", "brick": name, "success": cycle_ok, "t": float(data.time)}
            if callable(record): record(event)
            elif hasattr(record, "cycle_end"): record.cycle_end(cycle_ok)
    if inspection_renderer is not None: inspection_renderer.close()
    if _ACTIVE_RECORDERS:
        for renderer in _ACTIVE_RECORDERS.values(): renderer.close()
    _ACTIVE_RECORD, _ACTIVE_RECORDERS, _ACTIVE_TICK, _ACTIVE_VERDICT = old_record, old_recorders, old_tick, old_verdict
    return {'seed':seed,'bricks':len(brick_ids),'successes':successes,'score':successes/len(brick_ids) if brick_ids else 0,'events':events}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=7); ap.add_argument('--tier',choices=sorted(TIERS),default='messy'); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--json',type=Path)
    a=ap.parse_args(); os.environ.setdefault('MUJOCO_GL','egl'); a.out.parent.mkdir(parents=True,exist_ok=True)
    sentence=TIERS[a.tier]
    xml=a.out.with_suffix('.xml'); xml_text=station_xml(a.seed,sentence,robot_include=str((ROOT/'assets/SO101/Simulation/SO101/so101_dryrun.xml').resolve()))
    specs=layout_bricks(a.seed,sentence)
    xml.write_text(xml_text)
    model=mujoco.MjModel.from_xml_path(str(xml)); data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    writer=imageio.get_writer(a.out,fps=30,codec='libx264',quality=7); renderers={c:mujoco.Renderer(model,360,640) for c in ('free_view','top_view','inspection_view')}
    global _VIDEO; _VIDEO=writer
    for c,r in renderers.items(): r.update_scene(data,c); writer.append_data(r.render())
    result=rollout(model,data,a.seed,renderers,specs); writer.close(); [r.close() for r in renderers.values()]
    result.update({'model':str(xml),'mujoco':mujoco.__version__,'video':str(a.out),'views':list(renderers)})
    if a.json: a.json.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
if __name__=='__main__': main()
