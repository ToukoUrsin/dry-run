"""Render the generated station from all three camera views."""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image


def render(xml: str, out_dir: str, steps: int = 0) -> list[str]:
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=540, width=960)
    paths = []
    for camera in ("free_view", "top_view", "inspection_view"):
        renderer.update_scene(data, camera=camera)
        # Renderer.render() returns a view into its reusable buffer; copy it
        # before the next view (and before close()) so saved frames are real.
        image = renderer.render().copy()
        path = out / f"{camera}.png"
        Image.fromarray(image).save(path)
        paths.append(str(path))
    renderer.close()
    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("xml")
    p.add_argument("--out-dir", default="/bulk2/runs/dryrun/m1/frames")
    p.add_argument("--steps", type=int, default=0)
    a = p.parse_args()
    print("\n".join(render(a.xml, a.out_dir, a.steps)))
