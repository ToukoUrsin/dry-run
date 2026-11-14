"""Threshold selection, held-out evaluation and CPU timing for Inspector."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import cv2, numpy as np, openvino as ov
from .defects import DEFECT_TYPES, HELD_OUT_TYPES, SEEN_FOR_THRESHOLD


def _score_histograms(rows: list[dict], bins: int = 20) -> dict:
    """Return compact, deterministic score histograms for the evidence page."""
    scores = [float(r["score"]) for r in rows if np.isfinite(float(r["score"]))]
    if not scores:
        edges = np.linspace(0.0, 1.0, bins + 1)
    else:
        lo, hi = min(scores), max(scores)
        # A constant score still needs non-zero-width bins for JSON consumers.
        if hi <= lo:
            pad = max(1e-6, abs(lo) * 1e-3)
            lo, hi = lo - pad, hi + pad
        edges = np.linspace(lo, hi, bins + 1)

    def counts(subset: list[dict]) -> list[int]:
        values = [float(r["score"]) for r in subset if np.isfinite(float(r["score"]))]
        return np.histogram(values, bins=edges)[0].astype(int).tolist()

    types = {typ: counts([r for r in rows if r["type"] == typ])
             for typ in ["good", *DEFECT_TYPES]}
    return {
        "edges": [float(x) for x in edges],
        "bins": [float(x) for x in edges[:-1]],
        "counts": counts(rows),
        "per_type": types,
    }


def page_payload(metrics: dict, *, source: str | Path | None = None) -> dict:
    """Project evaluation metrics into the stable page/API contract.

    The full evaluation retains model-specific fields for analysis.  The page
    only needs this compact shape: confusion counts, recall by defect,
    measured CPU latency, and score histograms.  ``source`` is included so a
    badge can identify the artifact that produced the numbers.
    """
    per_type = metrics.get("per_type") or {}
    latency = metrics.get("latency", metrics.get("latency_cpu", {})) or {}
    recalls = metrics.get("per_defect_recall")
    if recalls is None:
        recalls = {
            typ: float(row.get("recall", 0.0))
            for typ, row in per_type.items()
            if typ != "good"
        }
    hist = metrics.get("score_histograms")
    if hist is None:
        # Older artifacts did not persist individual scores.  Keep the key and
        # an honest empty shape instead of inventing a distribution.
        hist = {"edges": [], "bins": [], "counts": [], "per_type": {}}
    return {
        "schema": "dryrun.inspector.v1",
        "source": str(source) if source is not None else metrics.get("source"),
        "n": int(metrics.get("n", 0)),
        "auroc": metrics.get("auroc"),
        "threshold": metrics.get("threshold"),
        "confusion": metrics.get("confusion", {}),
        "per_defect_recall": recalls,
        "latency": latency,
        "score_histograms": hist,
        # Explicit aliases make the contract convenient for small clients and
        # preserve the names used by the original evaluation artifact.
        "per_type": per_type,
        "held_out": metrics.get("held_out", {}),
        "openvino_version": metrics.get("openvino_version"),
    }


def load_page_json(path: str | Path = "/bulk2/runs/dryrun/m2/evaluation.json") -> dict:
    """Load an evaluation artifact and return page-ready JSON.

    This helper is intentionally read-only; the web process can expose the
    result without importing the database or scanning image files.
    """
    path = Path(path)
    if not path.exists():
        return {
            "schema": "dryrun.inspector.v1",
            "status": "unavailable",
            "source": str(path),
            "message": "No inspector evaluation artifact has been recorded.",
            "confusion": {}, "per_defect_recall": {}, "latency": {},
            "score_histograms": {"edges": [], "bins": [], "counts": [], "per_type": {}},
        }
    try:
        metrics = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema": "dryrun.inspector.v1", "status": "unavailable",
            "source": str(path), "message": f"Could not read inspector evaluation: {exc}",
            "confusion": {}, "per_defect_recall": {}, "latency": {},
            "score_histograms": {"edges": [], "bins": [], "counts": [], "per_type": {}},
        }
    result = page_payload(metrics, source=path)
    result["status"] = "ok"
    return result

def _score(infer, inp, path: Path) -> tuple[float, np.ndarray]:
    b = cv2.imread(str(path)); rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB); x = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)[None].astype(np.float32) / 255
    out = infer.infer({inp: x}); return float(np.asarray(out["pred_score"]).reshape(-1)[0]), np.asarray(out["anomaly_map"]).reshape(256, 256)

def _rows(compiled, data: Path, split: str) -> list[dict]:
    infer = compiled.create_infer_request(); inp = compiled.input(0); rows = []
    good = data / "good" / split
    for p in sorted(good.glob("*.png")):
        s, _ = _score(infer, inp, p); rows.append({"path": str(p), "type": "good", "actual_defect": False, "score": s})
    for typ in DEFECT_TYPES:
        for p in sorted((data / "defects" / split / typ).glob("*.png")):
            s, _ = _score(infer, inp, p); rows.append({"path": str(p), "type": typ, "actual_defect": True, "score": s})
    return rows

def _threshold(rows: list[dict]) -> tuple[float, dict]:
    # Validation contains good plus only the four seen visual defect types.
    cand = sorted({float(r["score"]) for r in rows}); best = (-1., .5, {})
    for t in cand:
        tp = sum(r["actual_defect"] and r["score"] >= t for r in rows); tn = sum((not r["actual_defect"]) and r["score"] < t for r in rows)
        fp = sum((not r["actual_defect"]) and r["score"] >= t for r in rows); fn = sum(r["actual_defect"] and r["score"] < t for r in rows)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn); bal = .5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
        if (bal, f1) > (best[2].get("balanced_accuracy", -1), best[0]):
            best = (f1, float(t), {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "f1": f1, "balanced_accuracy": bal})
    return best[1], best[2]

def _metrics(rows: list[dict], threshold: float) -> dict:
    for r in rows: r["pred_defect"] = bool(r["score"] >= threshold)
    table = {}
    for typ in ["good", *DEFECT_TYPES]:
        rs = [r for r in rows if r["type"] == typ]; tp = sum(r["actual_defect"] and r["pred_defect"] for r in rs); tn = sum((not r["actual_defect"]) and (not r["pred_defect"]) for r in rs); fp = sum((not r["actual_defect"]) and r["pred_defect"] for r in rs); fn = sum(r["actual_defect"] and (not r["pred_defect"]) for r in rs)
        scores = [r["score"] for r in rs]
        table[typ] = {"n": len(rs), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                      "recall": tp / max(1, tp + fn),
                      "mean_score": float(np.mean(scores)) if scores else None}
    y = np.asarray([r["actual_defect"] for r in rows], dtype=np.int32); s = np.asarray([r["score"] for r in rows]); order = np.argsort(s); ranks = np.empty_like(order); ranks[order] = np.arange(len(order)); pos = y == 1; neg = y == 0; auroc = float((ranks[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / max(1, pos.sum() * neg.sum()))
    tp = sum(r["actual_defect"] and r["pred_defect"] for r in rows); tn = sum((not r["actual_defect"]) and (not r["pred_defect"]) for r in rows); fp = sum((not r["actual_defect"]) and r["pred_defect"] for r in rows); fn = sum(r["actual_defect"] and (not r["pred_defect"]) for r in rows)
    # Operationally, a false accept is a defective frame accepted as good
    # (FN over all defective frames), and a false reject is a good frame sent
    # to the defect bowl (FP over all good frames). Emit unambiguous FPR/FNR
    # aliases too, since some readers use FAR/FRR for the opposite convention.
    return {"n": len(rows), "auroc": auroc, "threshold": threshold,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                          "false_accept_rate": fn / max(1, fn + tp),
                          "false_reject_rate": fp / max(1, fp + tn),
                          "false_positive_rate": fp / max(1, fp + tn),
                          "false_negative_rate": fn / max(1, fn + tp)},
            "per_type": table, "rows": rows}

def _latency(model_path: Path) -> dict:
    core = ov.Core(); result = {}
    for threads in (1, 4):
        model = core.read_model(str(model_path)); model_input = model.input(0)
        # Comparison exports can retain the datamodule batch (for example
        # PatchCore exports with batch 2), while dynamic exports otherwise
        # choose an implementation default. Pin dynamic inputs to batch 1 so
        # latency is comparable and the inference request shape is explicit.
        if not model_input.partial_shape.is_static:
            shape = [int(dim.get_length()) if dim.is_static else (1 if i == 0 else 256)
                     for i, dim in enumerate(model_input.partial_shape)]
            model.reshape({model_input: shape})
        compiled = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads}); req = compiled.create_infer_request(); inp = compiled.input(0)
        shape = [int(dim) for dim in inp.shape]
        x = np.zeros(shape, np.float32)
        for _ in range(20): req.infer({inp: x})
        samples = []
        for _ in range(100):
            t = time.perf_counter_ns(); req.infer({inp: x}); samples.append((time.perf_counter_ns() - t) / 1e6)
        result[str(threads)] = {"threads": threads, "median_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)), "n": len(samples)}
    return result

def _contact_sheet(rows: list[dict], compiled, out: Path, threshold: float) -> str:
    chosen = []
    for typ in ["good", *DEFECT_TYPES]:
        rs = [r for r in rows if r["type"] == typ]; chosen += rs[:1]
        miss = [r for r in rs if r["actual_defect"] != r["pred_defect"]]
        if miss: chosen.append(miss[0])
    tiles = []
    req = compiled.create_infer_request(); inp = compiled.input(0)
    for r in chosen:
        b = cv2.imread(r["path"]); rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB); x = cv2.resize(rgb, (256, 256)).transpose(2, 0, 1)[None].astype(np.float32) / 255; outv = req.infer({inp: x}); amap = np.asarray(outv["anomaly_map"]).reshape(256, 256); amap = cv2.normalize(amap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8); heat = cv2.applyColorMap(amap, cv2.COLORMAP_JET); tile = cv2.hconcat([cv2.resize(b, (180, 180)), cv2.resize(heat, (180, 180))]); cv2.putText(tile, f"{r['type']} {'MISS' if r['actual_defect'] != r['pred_defect'] else ''} {r['score']:.2f}", (4, 174), cv2.FONT_HERSHEY_SIMPLEX, .34, (255, 255, 255), 1, cv2.LINE_AA); tiles.append(tile)
    cols = 3; rows_n = (len(tiles) + cols - 1) // cols; sheet = np.zeros((rows_n * 180, cols * 360, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles): sheet[(i // cols) * 180:(i // cols + 1) * 180, (i % cols) * 360:(i % cols + 1) * 360] = tile
    path = out / "contact_sheet.png"; cv2.imwrite(str(path), sheet); return str(path)

def evaluate(model_path: Path, data: Path, out: Path) -> dict:
    core = ov.Core(); compiled = core.compile_model(core.read_model(str(model_path)), "CPU")
    val = _rows(compiled, data, "val"); threshold, threshold_metrics = _threshold([r for r in val if r["type"] == "good" or r["type"] in SEEN_FOR_THRESHOLD])
    sidecar = model_path.with_name("threshold.json"); sidecar.write_text(json.dumps({"threshold": threshold, "validation": threshold_metrics, "seen_types": list(SEEN_FOR_THRESHOLD), "held_out_types": list(HELD_OUT_TYPES)}, indent=2))
    test = _rows(compiled, data, "test"); metrics = _metrics(test, threshold); metrics["validation"] = {"n": len(val), "threshold": threshold, "selected_on": list(SEEN_FOR_THRESHOLD), "metrics": threshold_metrics}; metrics["held_out"] = {t: metrics["per_type"][t] for t in HELD_OUT_TYPES}; metrics["latency_cpu"] = _latency(model_path); metrics["openvino_version"] = ov.__version__; metrics["contact_sheet"] = _contact_sheet(test, compiled, out.parent, threshold); metrics.pop("rows", None)
    out.write_text(json.dumps(metrics, indent=2)); print(json.dumps(metrics, indent=2)); return metrics

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--model", type=Path, default=Path("/bulk2/runs/dryrun/m2/model/openvino/weights/openvino/dryrun_inspector.xml")); ap.add_argument("--data", type=Path, default=Path("/bulk2/runs/dryrun/m2/data")); ap.add_argument("--out", type=Path, default=Path("/bulk2/runs/dryrun/m2/evaluation.json")); a = ap.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True); evaluate(a.model, a.data, a.out)
if __name__ == "__main__": main()
