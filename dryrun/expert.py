"""Reference scripted expert for the Dry Run station.

Written by the supervisor outside Codex as a measured prototype and handed to Codex in
prompts/021. It is a drop-in for ``dryrun.expert.MeasuredExpert``: ``Expert(env)``, ``action()``
(six joint targets for the next 30 Hz tick), ``events()``, ``.phase`` and ``.current``.

The expert is the demonstrator, so it reads ground-truth brick poses from the simulator. It never
sets a pose: every motion goes through the six position actuators.
"""
from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np
import mujoco

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TCP_LOCAL = np.array([0.0225, 0.0, -0.086])   # centre of a 32 mm brick that touches the fixed pad, gripper frame
PAN_X = 0.0388                                # world x of the shoulder pan axis
Q_OPEN, Q_CLOSE = 1.0, 0.2                    # gripper targets; a 32 mm brick stops the jaw near 0.60, the torque limit sets the force
GRASP_DZ = 0.0034                             # TCP above the brick centre at the grasp (0.013 on the table)
MID_DZ, HOVER_DZ, LIFT1_DZ, LIFT2_DZ = 0.022, 0.042, 0.027, 0.057
APPROACH_OFF = 0.0045                         # start this far towards the fixed pad; closing pushes the brick home
FOOT_NEG, FOOT_POS, FOOT_HALF_W = 0.027, 0.052, 0.019   # gripper footprint around the brick centre
SLOT = (-0.0175, 0.0185)                      # part of the footprint taken by the brick itself
BOWL_KEEPOUT = 0.110                          # footprint stays this far from a bowl centre
BOWL_RADIUS = 0.095
PRESENT_A = np.array([0.5527, -1.2042, 0.8575, 1.0175, -2.3])
PRESENT_B = np.array([0.7468, -1.5394, 1.1230, 1.0625, 0.9507])
DROP_OFFSETS = ((0.03, 0.03), (-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03), (0.0, 0.0))
DROP_Z = 0.115
MAX_TRIES = 3
MAX_STIRS = 12                                # pushes through a blocked cluster per episode
STIR_Z = 0.016                                # TCP height of the closed gripper while it pushes
MAX_JAW_WIDTH = 0.040                         # widest thing the open jaw takes
MIN_PAD_FORCE = 0.2
HZ = 30


class MeasuredExpert:
    def __init__(self, env):
        self.env = env
        self.m, self.d = env.model, env.data
        m = self.m
        self.k = mujoco.MjData(m)             # scratch copy for kinematics; the simulation is never touched
        nid = lambda t, n: mujoco.mj_name2id(m, t, n)
        jid = [nid(mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM]
        self.qa = [int(m.jnt_qposadr[j]) for j in jid]
        self.da = [int(m.jnt_dofadr[j]) for j in jid]
        self.lo, self.hi = m.jnt_range[jid, 0] + 0.02, m.jnt_range[jid, 1] - 0.02
        self.gq = int(m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "gripper")])
        self.G = nid(mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.pads = (nid(mujoco.mjtObj.mjOBJ_GEOM, "fixed_grip_pad"), nid(mujoco.mjtObj.mjOBJ_GEOM, "moving_grip_pad"))
        self.bricks = [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("brick_")]
        self.defect = {}
        for s in getattr(env, "specs", []):
            b = nid(mujoco.mjtObj.mjOBJ_BODY, f"brick_{s.index}")
            if b >= 0: self.defect[b] = str(s.defect)
        self.bowl = {"good": nid(mujoco.mjtObj.mjOBJ_BODY, "good_bowl"), "defective": nid(mujoco.mjtObj.mjOBJ_BODY, "defective_bowl")}
        self.phase = "neutral"; self.current = None; self._events = []
        self._cmd = np.r_[[self.d.qpos[a] for a in self.qa], self.d.qpos[self.gq]].astype(float)
        self._gen = self._run()

    # ------------------------------------------------------------------ public
    def action(self):
        try: self._cmd = np.asarray(next(self._gen), float)
        except StopIteration: self.phase = "done"; self.current = None
        return self._cmd.copy()

    def events(self):
        out, self._events = self._events, []
        return out

    # ------------------------------------------------------------------ helpers
    def _emit(self, name, brick=None, **kw):
        row = {"event": name, "t": float(self.d.time), **kw}
        if brick is not None: row["brick"] = int(brick)
        self._events.append(row)

    def _fk(self, q):
        self.k.qpos[:] = self.d.qpos; self.k.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.k); mujoco.mj_comPos(self.m, self.k)
        R = self.k.xmat[self.G].reshape(3, 3)
        return self.k.xpos[self.G] + R @ TCP_LOCAL, R

    def _ik(self, pos, jaw=None, pitch=0.0, q0=None):
        """Damped least squares. Fingers point down, tilted outward by ``pitch``; ``jaw`` is the signed
        horizontal direction from the fixed pad to the moving pad. Returns (q, pos_err_m, finger_deg, jaw_deg)."""
        pos = np.asarray(pos, float)
        az = math.atan2(pos[1], pos[0] - PAN_X); rhat = np.array([math.cos(az), math.sin(az), 0.0])
        f_des = math.sin(pitch) * rhat + np.array([0.0, 0.0, -math.cos(pitch)])
        jd = None
        if jaw is not None:
            jd = np.asarray(jaw, float); jd = jd - jd.dot(f_des) * f_des; jd /= np.linalg.norm(jd)
        seeds = [np.asarray(q0, float)] if q0 is not None else []
        seeds += [np.array([-az, a, b, c, roll]) for roll in (0.0, -1.4, 1.4, -2.5, 2.5)
                  for a, b, c in ((-0.3, 0.6, 1.2), (0.2, 0.2, 1.2), (-0.8, 1.2, 1.1))]
        best = None
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        for q in seeds:
            q = np.clip(q, self.lo, self.hi)
            for _ in range(120):
                p, R = self._fk(q)
                e_rot = np.cross(-R[:, 2], f_des)
                if jd is not None: e_rot = e_rot + np.cross(R[:, 0], jd)
                e = np.r_[pos - p, 0.05 * e_rot]
                if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e_rot) < 2e-3: break
                mujoco.mj_jac(self.m, self.k, jp, jr, p, self.G)
                J = np.vstack([jp[:, self.da], 0.05 * jr[:, self.da]])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
                n = np.abs(dq).max()
                if n > 0.3: dq *= 0.3 / n
                q = np.clip(q + dq, self.lo, self.hi)
            p, R = self._fk(q)
            pe = float(np.linalg.norm(pos - p))
            fe = math.degrees(math.acos(float(np.clip((-R[:, 2]).dot(f_des), -1, 1))))
            je = 0.0 if jd is None else math.degrees(math.acos(float(np.clip(R[:, 0].dot(jd), -1, 1))))
            score = pe * 1000 + 0.2 * fe + 0.2 * je
            if best is None or score < best[0]: best = (score, q.copy(), pe, fe, je)
            if pe < 5e-4 and fe < 1.0 and je < 1.5: break
        return best[1], best[2], best[3], best[4]

    @staticmethod
    def _ok(sol): return sol[1] < 1e-3 and sol[2] < 2.0 and sol[3] < 3.0

    def _tcp(self):
        R = self.d.xmat[self.G].reshape(3, 3)
        return self.d.xpos[self.G] + R @ TCP_LOCAL

    def _pad_forces(self, body):
        out = [0.0, 0.0]; f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if body not in (int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])): continue
            for j, pad in enumerate(self.pads):
                if pad in (int(c.geom1), int(c.geom2)):
                    mujoco.mj_contactForce(self.m, self.d, i, f6); out[j] += abs(float(f6[0]))
        return out

    def _hand_force(self):
        """Total normal force between the hand (gripper and jaw bodies) and any brick."""
        hand = {int(self.m.geom_bodyid[g]) for g in self.pads}; bricks = set(self.bricks); f6 = np.zeros(6); total = 0.0
        for i in range(self.d.ncon):
            c = self.d.contact[i]; b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            if (b1 in hand and b2 in bricks) or (b2 in hand and b1 in bricks):
                mujoco.mj_contactForce(self.m, self.d, i, f6); total += abs(float(f6[0]))
        return total

    def _half(self, b):
        g = next(i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == b and self.m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX)
        return self.m.geom_size[g][:2].copy()

    def _bowl_xy(self): return {k: self.d.xpos[b][:2].copy() for k, b in self.bowl.items()}

    def _in_bowl(self, b):
        p = self.d.xpos[b]
        return any(np.linalg.norm(p[:2] - xy) < BOWL_RADIUS for xy in self._bowl_xy().values())

    def _clear(self, b, c, jaw):
        """True if the open gripper can come straight down on brick ``b`` with this jaw direction."""
        j = np.asarray(jaw[:2], float); j /= np.linalg.norm(j); perp = np.array([-j[1], j[0]])
        pts = np.array([c[:2] + a * j + w * perp for a in np.linspace(-FOOT_NEG, FOOT_POS, 12) if not SLOT[0] < a < SLOT[1]
                        for w in np.linspace(-FOOT_HALF_W, FOOT_HALF_W, 5)])
        for o in self.bricks:
            if o == b or self.d.xpos[o][2] < -0.01 or self._in_bowl(o): continue
            R = self.d.xmat[o].reshape(3, 3)
            if abs(R[2, 2]) > 0.95:
                hs = self._half(o) + 0.003; loc = (pts - self.d.xpos[o][:2]) @ R[:2, :2]
                if np.any((np.abs(loc[:, 0]) < hs[0]) & (np.abs(loc[:, 1]) < hs[1])): return False
            elif np.any(np.linalg.norm(pts - self.d.xpos[o][:2], axis=1) < float(np.linalg.norm(self._half3(o)[:2])) + 0.003): return False
        return not any(np.any(np.linalg.norm(pts - xy, axis=1) < BOWL_KEEPOUT) for xy in self._bowl_xy().values())

    def _half3(self, b):
        g = next(i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == b and self.m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX)
        return self.m.geom_size[g].copy()

    def _plan(self, b):
        """(centre, shifted target, jaw, z_grasp, q_grasp, q_hover, q_lift) or None. Works for a brick resting on any face."""
        R = self.d.xmat[b].reshape(3, 3); c0 = self.d.xpos[b].copy(); h = self._half3(b)
        up = int(np.argmax(np.abs(R[2, :])))
        if abs(R[2, up]) < 0.95 or c0[2] - h[up] > 0.035: return None          # tilted, or lying high on other bricks
        cands = []
        for i in range(3):
            if i == up: continue
            width = 2 * h[i] + (0.0032 if i == 2 else 0.0)                      # studs add to the height axis
            if width > MAX_JAW_WIDTH: continue
            for sgn in (1.0, -1.0): cands.append((sgn * R[:, i], width))
        for jaw, width in cands:
            jaw = np.array([jaw[0], jaw[1], 0.0]); jaw /= np.linalg.norm(jaw)
            if not self._clear(b, c0, jaw): continue
            # the TCP is the centre of a 32 mm object on the fixed pad; a narrower one sits closer to the fixed pad
            c = c0.copy(); c[:2] -= (APPROACH_OFF + (0.016 - width / 2)) * jaw[:2]; zg = c0[2] + GRASP_DZ
            sols = []; q0 = None
            for dz in (0.0, HOVER_DZ, LIFT2_DZ):
                s = self._ik((c[0], c[1], zg + dz), jaw=jaw, q0=q0)
                if not self._ok(s): break
                sols.append(s[0]); q0 = s[0]
            if len(sols) == 3: return c0, c, jaw, zg, sols[0], sols[1], sols[2]
        return None

    def _gap(self, p, skip=None):
        """Distance from table point ``p`` to the nearest brick outline (bricks in bowls do not count)."""
        best = 1.0
        for o in self.bricks:
            if o == skip or self._in_bowl(o) or self.d.xpos[o][2] < -0.01: continue
            R = self.d.xmat[o].reshape(3, 3); c = self.d.xpos[o][:2]
            if abs(R[2, 2]) > 0.95:
                loc = np.abs((p - c) @ R[:2, :2]) - self._half(o)
                dist = float(np.linalg.norm(np.maximum(loc, 0.0)))
            else:
                dist = max(0.0, float(np.linalg.norm(p - c)) - float(np.linalg.norm(self._half3(o)[:2])))
            best = min(best, dist)
        return best

    def _stir_plan(self, todo, n):
        """A straight push with the closed gripper against a blocked brick, to break a cluster or a pile apart.
        Pushes that end near the middle of the reach zone are tried first, so bricks are not shoved out of reach."""
        d = self.d
        order = todo[n % len(todo):] + todo[:n % len(todo)]
        for b in order:
            tgt = d.xpos[b][:2].copy(); cands = []
            for ang in np.arange(0.0, 2 * math.pi, math.pi / 6) + 0.37 * n:
                u = np.array([math.cos(ang), math.sin(ang)])
                for L in (0.05, 0.065, 0.08):
                    a, e = tgt - L * u, tgt + 0.045 * u
                    ra, re = math.hypot(a[0] - PAN_X, a[1]), math.hypot(e[0] - PAN_X, e[1])
                    if not (0.09 < ra < 0.272 and 0.11 < re < 0.235): continue
                    if abs(math.atan2(e[1], e[0] - PAN_X)) > 1.0: continue
                    if self._gap(a) < 0.032: continue                                   # room for the closed gripper to come down
                    if any(np.linalg.norm(q - xy) < BOWL_KEEPOUT + 0.02 for q in (a, e) for xy in self._bowl_xy().values()): continue
                    rhat = (tgt - np.array([PAN_X, 0.0])); rhat = rhat / max(np.linalg.norm(rhat), 1e-6)
                    out = max(0.0, float(u @ rhat))                                     # outward pushes shove bricks out of reach: last choice
                    cands.append((round(out, 1) + abs(re - 0.17), a, e)); break
            for _, a, e in sorted(cands, key=lambda c: c[0])[:4]:
                qs = []; q0 = None
                for f in np.linspace(0.0, 1.0, 5):
                    p = a + f * (e - a); s = self._ik((p[0], p[1], STIR_Z), q0=q0)
                    if not (s[1] < 1.5e-3 and s[2] < 3.0): break
                    qs.append(s[0]); q0 = s[0]
                if len(qs) < 5: continue
                for z, pitch in ((0.075, 0.0), (0.060, 0.0), (0.075, 0.4), (0.075, 0.7)):   # far starts need a lower or tilted approach
                    top = self._ik((a[0], a[1], z), pitch=pitch, q0=qs[0])
                    if top[1] < 2e-3: return b, top[0], qs
        return None

    def _label(self, b, since):
        """Inspector verdicts given for this brick in this cycle win; without an inspector the demonstrator uses ground truth."""
        said = [str(e.get("verdict")).lower() for e in getattr(self.env, "_events", [])
                if e.get("event") == "verdict" and e.get("brick") == b and e.get("t", 0.0) >= since]
        if said: return "defective" if any(v in ("defective", "-1", "bad") for v in said) else "good"
        return "good" if self.defect.get(b, "none") in ("none", "", "None") else "defective"

    # ------------------------------------------------------------------ motion primitives (generators of per-tick actions)
    def _move(self, q, grip, seconds, phase):
        self.phase = phase
        q0 = self._cmd[:5].copy(); g0 = float(self._cmd[5]); n = max(2, int(round(seconds * HZ)))
        for i in range(1, n + 1):
            a = i / n; a = 10 * a**3 - 15 * a**4 + 6 * a**5
            yield np.r_[q0 + a * (np.asarray(q) - q0), g0 + a * (grip - g0)]

    def _hold(self, seconds, phase=None):
        if phase: self.phase = phase
        for _ in range(max(1, int(round(seconds * HZ)))): yield self._cmd.copy()

    # ------------------------------------------------------------------ the cycle
    def _run(self):
        d = self.d
        yield from self._hold(0.2, "neutral")
        qN = self._ik((0.22, 0.0, 0.14), pitch=0.6)[0]
        drops = {}
        for k, xy in self._bowl_xy().items():
            sols = [self._ik((xy[0] + dx, xy[1] + dy, DROP_Z), pitch=0.6) for dx, dy in DROP_OFFSETS]
            drops[k] = [s[0] for s in sols if s[1] < 2e-3]
        n_drop = {"good": 0, "defective": 0}; tries = {}; stirs = 0
        yield from self._move(qN, Q_OPEN, 1.0, "neutral")
        while True:
            todo = [b for b in self.bricks if not self._in_bowl(b) and d.xpos[b][2] > -0.01 and tries.get(b, 0) < MAX_TRIES]
            todo.sort(key=lambda b: math.hypot(d.xpos[b][0] - PAN_X, d.xpos[b][1]))
            plan = b = None
            for b in todo:
                plan = self._plan(b)
                if plan: break
            if plan is None:
                stir = self._stir_plan(todo, stirs) if todo and stirs < MAX_STIRS else None
                if stir is None:
                    for b in todo: self._emit("ik_fail", b, reason="no_clear_grasp")
                    break
                stirs += 1; pushed, q_top, qs = stir
                self._emit("push", pushed, reason="blocked")
                yield from self._move(q_top, 0.0, 1.0, "push_reach")
                hit = False
                for act in self._move(qs[0], 0.0, 0.7, "push_reach"):
                    yield act
                    if self._hand_force() > 1.0: hit = True; break          # came down on a brick: back off, never squash it
                if hit:
                    yield from self._move(q_top, 0.0, 0.5, "push_abort")
                else:
                    for q in qs[1:]: yield from self._move(q, 0.0, 0.3, "push")
                yield from self._move(qN, Q_OPEN, 1.0, "neutral"); yield from self._hold(0.3)
                continue
            c0, c, jaw, zg, qg, qh, ql = plan
            tries[b] = tries.get(b, 0) + 1; t0 = float(d.time)
            self.current = SimpleNamespace(body=b, name=mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b), defect=self.defect.get(b, "none"))
            self._emit("reach", b)
            yield from self._move(qh, Q_OPEN, 1.0, "hover"); yield from self._hold(0.15)
            qm = self._ik((c[0], c[1], zg + MID_DZ), jaw=jaw, q0=self._cmd[:5])[0]
            hit = False
            for q in (qm, qg):
                for act in self._move(q, Q_OPEN, 0.35, "down"):
                    yield act
                    if self._hand_force() > 3.0: hit = True; break            # landed on a brick: stop, never squash it
                if hit: break
            if hit:
                self._emit("miss", b, reason="blocked_descent")
                yield from self._move(qh, Q_OPEN, 0.6, "retreat"); yield from self._move(qN, Q_OPEN, 1.0, "neutral")
                self.current = None; continue
            yield from self._hold(0.2, "touch")
            yield from self._move(qg, Q_CLOSE, 0.6, "close"); yield from self._hold(0.3)
            forces = self._pad_forces(b)
            if min(forces) < MIN_PAD_FORCE:
                self._emit("miss", b, pad_n=[round(f, 2) for f in forces])
                yield from self._move(qg, Q_OPEN, 0.3, "retreat"); yield from self._move(qh, Q_OPEN, 0.6, "retreat")
                yield from self._move(qN, Q_OPEN, 1.0, "neutral"); self.current = None; continue
            q1 = self._ik((c[0], c[1], zg + LIFT1_DZ), jaw=jaw, q0=self._cmd[:5])[0]
            yield from self._move(q1, Q_CLOSE, 0.4, "lift"); yield from self._move(ql, Q_CLOSE, 0.4, "lift")
            for pose, q, secs in (("a", PRESENT_A, 1.2), ("b", PRESENT_B, 1.0)):
                yield from self._move(q, Q_CLOSE, secs, "to_present_" + pose)
                yield from self._hold(0.2, "present_" + pose)
                self._emit("present", b, pose=pose)          # the brick is still: the inspector looks now
                yield from self._hold(0.2)
            if np.linalg.norm(d.xpos[b] - self._tcp()) > 0.02:
                self._emit("slip", b)
                yield from self._move(qN, Q_OPEN, 1.0, "neutral"); self.current = None; continue
            label = self._label(b, t0)
            if not drops[label]:
                self._emit("ik_fail", b, reason="bowl_unreachable")
                yield from self._move(qN, Q_CLOSE, 1.0, "neutral"); yield from self._move(qN, Q_OPEN, 0.4, "release")
                tries[b] = MAX_TRIES; self.current = None; continue
            qd = drops[label][n_drop[label] % len(drops[label])]; n_drop[label] += 1
            yield from self._move(qd, Q_CLOSE, 1.2, "drop"); yield from self._hold(0.2)
            self._cmd[5] = float(d.qpos[self.gq])        # start the release from where the jaw is, so the grip force fades before the pad moves
            yield from self._move(qd, Q_OPEN, 0.5, "release"); yield from self._hold(0.4)
            self._emit("drop", b, bowl=label)
            yield from self._move(qN, Q_OPEN, 1.0, "neutral"); self.current = None
        self.phase = "done"; self.current = None


# Public names retained for the StationEnv contract.
Expert = MeasuredExpert
__all__ = ["MeasuredExpert", "Expert"]
