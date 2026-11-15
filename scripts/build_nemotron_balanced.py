"""Build the four-stage Nemotron set from existing simulator-labeled failures.

The output is deliberately outside the repository under /bulk3.  Every row
keeps its source path and event evidence so the score can be audited without
trusting a model-generated label.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

OUT = Path("/bulk3/runs/dryrun/nemotron_balanced")
FAILURES = Path("data/nemotron_failures300.jsonl")
WRONG = Path("/bulk3/runs/dryrun/nemotron_balanced_candidates/wrong-bowl-candidates.jsonl")
PILE = Path("/bulk3/runs/dryrun/analyst_448_pile/pile-failures.jsonl")
MESSY = Path("/bulk3/runs/dryrun/analyst_448_messy/messy-failures.jsonl")
STAGES = ("never_touched_brick", "touched_but_did_not_lift", "lifted_then_lost", "wrong_bowl")
PANELS = ("before_grasp", "end_closing", "presentation", "end_cycle")


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _font(size: int = 18):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _sheet_from_frames(paths: list[str], out: Path, *, source_id: str) -> None:
    """Make a 2x2 448 px panel sheet from already-rendered frames."""
    labels = PANELS
    canvas = Image.new("RGB", (896, 896), "#101418")
    draw = ImageDraw.Draw(canvas)
    fnt = _font(18)
    for i, label in enumerate(labels):
        p = Path(paths[min(i, len(paths) - 1)])
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            im = Image.new("RGB", (32, 32), "#555")
        # Reserve a small caption strip, then scale the source into a fixed
        # 448x448 panel.  This preserves the contract even for older 32 px
        # artifacts; provenance records the original resolution separately.
        im.thumbnail((448, 420), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (448, 448), "#101418")
        panel.paste(im, ((448 - im.width) // 2, 28 + (420 - im.height) // 2))
        ImageDraw.Draw(panel).text((8, 5), label, fill="white", font=fnt)
        x = (i % 2) * 448
        y = (i // 2) * 448
        canvas.paste(panel, (x, y))
    canvas.save(out, format="PNG", optimize=True)


def _copy_sheet(src: str, out: Path) -> None:
    im = Image.open(src).convert("RGB")
    if im.size != (896, 896):
        im = im.resize((896, 896), Image.Resampling.LANCZOS)
    im.save(out, format="PNG", optimize=True)


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build() -> dict[str, Any]:
    sheets = OUT / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # The pile/messy 448 px sheets have explicit simulator ik_fail events.
    # Prefer them for the untouched class so this set does not depend only on
    # the old 32 px capture.
    high_res = [x for x in _load(PILE) + _load(MESSY) if x.get("failed_stage") == "no_clear_grasp"]
    high_res.sort(key=lambda x: str(x.get("id", "")))
    for x in high_res[:21]:
        rid = f"never-{x['id']}"
        dst = sheets / f"{rid}.png"
        _copy_sheet(str(x["sheet"]), dst)
        rows.append({
            "id": rid, "label": "never_touched_brick", "sheet": str(dst),
            "source_kind": "analyst_448", "source_id": x["id"],
            "source_rows": [str(PILE if str(x['id']).startswith('pile-') else MESSY)],
            "simulator_events": [x.get("event", {})],
            "source_resolution": x.get("render_size", 448),
        })

    failures = _load(FAILURES)
    touched = [x for x in failures if int(x.get("pad_contacts", 0)) > 0 and not bool(x.get("picked_by_contact"))]
    lifted = [x for x in failures if bool(x.get("picked_by_contact"))]
    touched.sort(key=lambda x: x["id"])
    lifted.sort(key=lambda x: x["id"])
    for label, source in (("touched_but_did_not_lift", touched[:21]), ("lifted_then_lost", lifted[:21])):
        for x in source:
            rid = f"{label}-{x['id']}"
            dst = sheets / f"{rid}.png"
            # Use four distinct event-aligned frames from the six-frame source.
            _sheet_from_frames([x["frames"][i] for i in (0, 2, 4, 5)], dst, source_id=x["id"])
            events = [{k: x[k] for k in ("failed_stage", "pad_contacts", "peak_pad_force_N", "picked_by_contact", "final_xyz", "in_bowl", "success") if k in x}]
            rows.append({
                "id": rid, "label": label, "sheet": str(dst),
                "source_kind": "nemotron_failures300", "source_id": x["id"],
                "source_rows": [str(FAILURES)], "simulator_events": events,
                "seed": x.get("seed"), "brick": x.get("brick"),
                "source_resolution": 32,
            })

    wrong = _load(WRONG)
    wrong.sort(key=lambda x: x["id"])
    for x in wrong[:21]:
        rid = f"wrong-{x['id']}"
        dst = sheets / f"{rid}.png"
        _copy_sheet(x["sheet"], dst)
        rows.append({
            "id": rid, "label": "wrong_bowl", "sheet": str(dst),
            "source_kind": "m4_modal_map", "source_id": x["id"],
            "source_rows": [x["source_shard"]],
            "simulator_events": x.get("simulator_events", []),
            "drop_event": x.get("drop_event"), "seed": x.get("seed"),
            "brick": x.get("brick"), "source_resolution": 64,
        })

    # Stable order makes the cache and audit reproducible.
    order = {s: i for i, s in enumerate(STAGES)}
    rows.sort(key=lambda x: (order[x["label"]], x["id"]))
    counts = {s: sum(x["label"] == s for x in rows) for s in STAGES}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"unbalanced rows: {counts}")
    (OUT / "input.jsonl").write_text("\n".join(json.dumps(x, separators=(",", ":")) for x in rows) + "\n")
    (OUT / "labels.jsonl").write_text("\n".join(json.dumps({"id": x["id"], "label": x["label"], "sheet": x["sheet"], "simulator_events": x["simulator_events"]}, separators=(",", ":")) for x in rows) + "\n")
    manifest = {
        "schema": "dryrun.nemotron_balanced.v1", "commit": _commit(),
        "model": "nvidia/nemotron-3.5-super-vl-120b-a12b", "render_size": 448,
        "panel_size": 448, "panel_count": 4, "stages": list(STAGES),
        "counts": counts, "n": len(rows), "sources": [str(FAILURES), str(PILE), str(MESSY), str(WRONG)],
        "note": "Labels are simulator-derived. 32/64 px legacy frames are resized into the fixed 448 px panel contract; source_resolution records that limitation.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
