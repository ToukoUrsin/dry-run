"""Run the render-trained inspector on a real phone photo."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
from .inspector import Inspector, DEFAULT_MODEL

def main() -> None:
    ap = argparse.ArgumentParser(description="Classify a phone photo of a Duplo brick")
    ap.add_argument("photo", type=Path); ap.add_argument("--model", type=Path, default=DEFAULT_MODEL); ap.add_argument("--out", type=Path); ap.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="tight top-face crop in source pixels") ; args = ap.parse_args()
    bgr = cv2.imread(str(args.photo), cv2.IMREAD_COLOR)
    if bgr is None: raise SystemExit(f"could not read {args.photo}")
    original_shape = list(bgr.shape[:2])
    if args.crop is not None:
        x, y, w, h = args.crop
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > bgr.shape[1] or y + h > bgr.shape[0]:
            raise SystemExit("--crop must be X Y W H inside the source image")
        bgr = bgr[y:y+h, x:x+w]
    result = Inspector(args.model)(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); heat = result.pop("heatmap")
    heat_path = args.out.with_name(args.out.stem + "_heatmap.png") if args.out else args.photo.with_name(args.photo.stem + "_heatmap.png")
    heat_path.parent.mkdir(parents=True, exist_ok=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(heat_path), heat):
        raise SystemExit(f"could not write heat map {heat_path}")
    result.update({"photo": str(args.photo), "heatmap": str(heat_path), "model": str(args.model), "crop": list(args.crop) if args.crop else None, "source_shape": original_shape, "note": "trained on MuJoCo renders; lighting, optics, scale and backgrounds differ from a phone photo"})
    payload = json.dumps(result, indent=2); print(payload)
    if args.out:
        try:
            args.out.write_text(payload)
        except OSError as exc:
            raise SystemExit(f"could not write result {args.out}: {exc}") from exc
if __name__ == "__main__": main()
