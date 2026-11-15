#!/usr/bin/env python3
"""Export a compiled Dry Run MuJoCo station as a pose-ready glTF binary.

Every node is a MuJoCo body. Geometry is copied from the compiled model,
including mesh vertices/faces and generated faces for MuJoCo primitives. The
viewer only changes node world transforms from replay frames.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import mujoco

from dryrun.env import StationEnv


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(a) for a in q]
    qv = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def box(size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = [float(a) for a in size]
    v = np.array([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]], dtype=np.float32)
    f = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[4,0,3],[4,3,7]], dtype=np.uint32)
    return v, f


def cylinder(size: np.ndarray, segments: int = 24) -> tuple[np.ndarray, np.ndarray]:
    radius, half = float(size[0]), float(size[1])
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(angles) * radius, np.sin(angles) * radius], axis=1).astype(np.float32)
    v = np.concatenate([np.column_stack([ring, np.full(segments, -half)]), np.column_stack([ring, np.full(segments, half)]), [[0,0,-half],[0,0,half]]]).astype(np.float32)
    f = []
    for i in range(segments):
        j = (i + 1) % segments
        f.extend([[i, j, segments + j], [i, segments + j, segments + i], [2*segments, j, i], [2*segments + 1, segments + i, segments + j]])
    return v, np.asarray(f, dtype=np.uint32)


def mesh_geom(model: mujoco.MjModel, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    typ = int(model.geom_type[geom_id])
    if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[geom_id])
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        return np.asarray(model.mesh_vert[va:va + vn], dtype=np.float32).copy(), np.asarray(model.mesh_face[fa:fa + fn], dtype=np.uint32).copy()
    if typ == int(mujoco.mjtGeom.mjGEOM_BOX):
        return box(model.geom_size[geom_id])
    if typ == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return cylinder(model.geom_size[geom_id])
    if typ == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        # Stations use spheres only for small studs; a UV sphere keeps the
        # exported geometry independent of any browser-side primitive.
        r = float(model.geom_size[geom_id, 0]); rings, segs = 10, 20
        verts = []
        for i in range(rings + 1):
            p = np.pi * i / rings
            for j in range(segs):
                a = 2 * np.pi * j / segs
                verts.append([r*np.sin(p)*np.cos(a), r*np.sin(p)*np.sin(a), r*np.cos(p)])
        faces = []
        for i in range(rings):
            for j in range(segs):
                n = (j + 1) % segs; a = i*segs+j; b = i*segs+n; c = (i+1)*segs+n; d = (i+1)*segs+j
                if i: faces.append([a,b,d])
                if i < rings-1: faces.append([b,c,d])
        return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.uint32)
    return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)


def simplify_mesh(vertices: np.ndarray, faces: np.ndarray, max_triangles: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Keep the compiled visual mesh intact.

    The SO-101 STL meshes are already compact and contain the silhouette and
    printed-part detail that the browser needs. The old grid weld followed by
    every-Nth-face sampling tore holes through those surfaces and made the arm
    look like crumpled foil. There is no third-party quadric reducer in the
    deployment environment, so exporting the source visual triangles is the
    honest fallback. ``max_triangles`` remains as a compatibility argument for
    callers of older exporter versions.
    """
    del max_triangles
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def align4(data: bytearray) -> None:
    while len(data) % 4:
        data.append(0)


def split_crease_normals(vertices: np.ndarray, faces: np.ndarray, crease_degrees: float = 30.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split corners at creases while preserving smooth curved surfaces.

    MuJoCo's STL meshes share vertices across both flat and curved faces. A
    single smooth normal therefore rounds printed edges. Group incident faces
    at each vertex by a 30 degree normal crease, duplicate only the corners
    that cross a crease, and average each smooth group's area-weighted normals.
    """
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.uint32)
    if not len(f):
        return v, f, np.zeros_like(v, dtype=np.float32)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(b - a, c - a).astype(np.float32)
    length = np.linalg.norm(cross, axis=1, keepdims=True)
    face_normals = cross / np.maximum(length, 1e-12)
    cos_crease = float(np.cos(np.deg2rad(crease_degrees)))
    incident: list[list[int]] = [[] for _ in range(len(v))]
    for fi, tri in enumerate(f):
        for vi in tri:
            incident[int(vi)].append(fi)
    groups: list[dict[int, int]] = [{} for _ in range(len(v))]
    group_normals: list[list[np.ndarray]] = [[] for _ in range(len(v))]
    for vi, fs in enumerate(incident):
        if not fs:
            continue
        remaining = set(fs)
        gid = 0
        while remaining:
            seed = remaining.pop()
            stack = [seed]
            members = [seed]
            while stack:
                cur = stack.pop()
                joined = [other for other in remaining if float(np.dot(face_normals[cur], face_normals[other])) >= cos_crease]
                for other in joined:
                    remaining.remove(other)
                    stack.append(other)
                    members.append(other)
            summed = np.sum(cross[members], axis=0)
            summed /= max(float(np.linalg.norm(summed)), 1e-12)
            group_normals[vi].append(summed.astype(np.float32))
            for fi in members:
                groups[vi][fi] = gid
            gid += 1
    out_vertices: list[np.ndarray] = []
    out_normals: list[np.ndarray] = []
    out_faces = np.empty_like(f)
    corner_cache: dict[tuple[int, int], int] = {}
    for fi, tri in enumerate(f):
        for ci, vi_raw in enumerate(tri):
            vi = int(vi_raw)
            key = (vi, groups[vi].get(fi, 0))
            oi = corner_cache.get(key)
            if oi is None:
                oi = len(out_vertices)
                corner_cache[key] = oi
                out_vertices.append(v[vi])
                out_normals.append(group_normals[vi][key[1]] if group_normals[vi] else face_normals[fi])
            out_faces[fi, ci] = oi
    return np.asarray(out_vertices, dtype=np.float32), out_faces.astype(np.uint32), np.asarray(out_normals, dtype=np.float32)


def export(model: mujoco.MjModel, out: Path) -> None:
    binary = bytearray(); views = []; accessors = []; materials = []; material_index = {}
    meshes = []; nodes = []
    skip_body = {"top_camera_mount", "inspection_camera_mount"}
    arm_bodies = {"base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"}
    gripper_pads = {"fixed_grip_pad", "moving_grip_pad"}

    def material(rgba: np.ndarray, name: str) -> int:
        key = tuple(round(float(x), 4) for x in rgba)
        if key not in material_index:
            material_index[key] = len(materials)
            materials.append({"name": name or "mujoco_material", "pbrMetallicRoughness": {"baseColorFactor": list(key), "metallicFactor": 0.0, "roughnessFactor": 0.72}, "doubleSided": True})
        return material_index[key]

    def accessor(values: np.ndarray, component_type: int, typ: str, target: int | None = None) -> int:
        align4(binary); offset = len(binary); raw = np.asarray(values).tobytes(order="C"); binary.extend(raw)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None: view["target"] = target
        vi = len(views); views.append(view)
        acc = {"bufferView": vi, "componentType": component_type, "count": int(len(values)), "type": typ}
        if typ == "VEC3":
            acc["min"] = np.asarray(values).min(axis=0).astype(float).tolist(); acc["max"] = np.asarray(values).max(axis=0).astype(float).tolist()
        ai = len(accessors); accessors.append(acc); return ai

    for bid in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
        primitives = []; seen = set()
        if body_name not in skip_body:
            start = int(model.body_geomadr[bid]); end = start + int(model.body_geomnum[bid])
            for gid in range(start, end):
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                # The compiled model contains visual meshes followed by
                # collision hull copies for the arm. Export the complete
                # visual arm, while retaining the two black pad boxes and
                # the scene's non-arm display geometry (bricks and bowls).
                if body_name in arm_bodies and geom_name not in gripper_pads and (
                    int(model.geom_contype[gid]) != 0 or int(model.geom_conaffinity[gid]) != 0
                ):
                    continue
                # Keep the work surface in view, but omit the station's rear
                # collision/visual wall so the camera can frame the robot and
                # both bowls tightly.
                if geom_name in {"table_edge", "back_wall"}:
                    continue
                key = (int(model.geom_type[gid]), int(model.geom_dataid[gid]), tuple(np.round(model.geom_pos[gid], 7)), tuple(np.round(model.geom_quat[gid], 7)))
                if key in seen: continue
                seen.add(key)
                verts, faces = mesh_geom(model, gid)
                if len(verts) == 0: continue
                verts, faces = simplify_mesh(verts, faces)
                verts = np.asarray([quat_rotate(model.geom_quat[gid], v) + model.geom_pos[gid] for v in verts], dtype=np.float32)
                verts, faces, normals = split_crease_normals(verts, faces, 30.0)
                pos_acc = accessor(verts, 5126, "VEC3", 34962)
                norm_acc = accessor(normals, 5126, "VEC3", 34962)
                idx_acc = accessor(faces.reshape(-1), 5125, "SCALAR", 34963)
                mat_id = int(model.geom_matid[gid]); rgba = model.mat_rgba[mat_id] if mat_id >= 0 else np.array([.6,.6,.6,1])
                mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id) if mat_id >= 0 else geom_name
                primitives.append({"attributes": {"POSITION": pos_acc, "NORMAL": norm_acc}, "indices": idx_acc, "material": material(rgba, mat_name or geom_name), "mode": 4})
        mesh_idx = None
        if primitives:
            mesh_idx = len(meshes); meshes.append({"name": body_name + "_geometry", "primitives": primitives})
        node = {"name": body_name, "extras": {"mujoco_body_id": bid}}
        if mesh_idx is not None: node["mesh"] = mesh_idx
        nodes.append(node)

    doc = {"asset": {"version": "2.0", "generator": "dryrun MuJoCo geometry export"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes))), "name": "MuJoCo station"}], "nodes": nodes, "meshes": meshes, "materials": materials, "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(binary)}]}
    raw_json = json.dumps(doc, separators=(",", ":")).encode("utf-8"); raw_json += b" " * ((4 - len(raw_json) % 4) % 4); align4(binary)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(raw_json) + 8 + len(binary)))
        fh.write(struct.pack("<II", len(raw_json), 0x4E4F534A)); fh.write(raw_json)
        fh.write(struct.pack("<II", len(binary), 0x004E4942)); fh.write(binary)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1001); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    env = StationEnv(args.seed, tier="loose", render_size=2)
    try: export(env.model, args.out)
    finally: env.close()
    print(args.out, args.out.stat().st_size)


if __name__ == "__main__": main()
