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


def simplify_mesh(vertices: np.ndarray, faces: np.ndarray, max_triangles: int = 18000) -> tuple[np.ndarray, np.ndarray]:
    """Weld a dense visual mesh onto a deterministic grid and remove duplicates.

    MuJoCo's arm meshes contain many small visual facets. The viewer does not
    need their CAD tessellation, but it does need the silhouette and all body
    geometry. Grid welding keeps those surfaces intact while bringing the
    compiled scene under the page's download budget without browser-side
    geometry generation.
    """
    if len(faces) <= max_triangles:
        return vertices, faces
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.uint32)
    lo = v.min(axis=0); hi = v.max(axis=0); span = np.maximum(hi - lo, 1e-6)
    # Aim for roughly one welded vertex per two triangles. Allocate grid
    # resolution in proportion to each axis' extent.
    target_vertices = max(256, int(max_triangles * 0.52))
    scale = (target_vertices / float(np.prod(span))) ** (1.0 / 3.0)
    dims = np.maximum(2, np.round(span * scale).astype(np.int32))
    # Keep pathological skinny parts from producing a huge grid.
    while int(np.prod(dims)) > target_vertices * 2:
        axis = int(np.argmax(dims))
        dims[axis] = max(2, dims[axis] - 1)
    q = np.floor((v - lo) / span * (dims - 1)).astype(np.int64)
    keys = q[:, 0] + q[:, 1] * int(dims[0]) + q[:, 2] * int(dims[0] * dims[1])
    unique, inverse = np.unique(keys, return_inverse=True)
    nv = np.zeros((len(unique), 3), dtype=np.float32)
    np.add.at(nv, inverse, v)
    counts = np.bincount(inverse)
    nv /= counts[:, None]
    nf = inverse[f]
    keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = nf[keep]
    if len(nf):
        nf = np.sort(nf, axis=1)
        _, first = np.unique(nf, axis=0, return_index=True)
        nf = nf[np.sort(first)]
    if len(nf) > max_triangles:
        # Uniformly sample the welded surface as a final deterministic cap.
        take = np.linspace(0, len(nf) - 1, max_triangles, dtype=np.int64)
        nf = nf[take]
    used = np.unique(nf.reshape(-1)) if len(nf) else np.empty(0, dtype=np.int64)
    remap = np.full(len(nv), -1, dtype=np.int64); remap[used] = np.arange(len(used))
    return nv[used], remap[nf].astype(np.uint32)


def align4(data: bytearray) -> None:
    while len(data) % 4:
        data.append(0)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute smooth visual normals for the exported, simplified mesh."""
    normals = np.zeros_like(vertices, dtype=np.float32)
    if len(faces):
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        cross = np.cross(b - a, c - a).astype(np.float32)
        np.add.at(normals, faces[:, 0], cross)
        np.add.at(normals, faces[:, 1], cross)
        np.add.at(normals, faces[:, 2], cross)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1e-8)


def export(model: mujoco.MjModel, out: Path) -> None:
    binary = bytearray(); views = []; accessors = []; materials = []; material_index = {}
    meshes = []; nodes = []
    skip_body = {"top_camera_mount", "inspection_camera_mount"}

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
                pos_acc = accessor(verts, 5126, "VEC3", 34962)
                norm_acc = accessor(vertex_normals(verts, faces), 5126, "VEC3", 34962)
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
