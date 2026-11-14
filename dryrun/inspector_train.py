"""Train and export the good-only Anomalib inspector."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import openvino as ov
from .defects import build_dataset
from .inspector_eval import _latency

def _datamodule(dataset: Path, batch_size: int = 16):
    from anomalib.data import Folder
    return Folder(name="dryrun_bricks", root=dataset, normal_dir="good/train",
                  normal_test_dir="good/val", abnormal_dir="defects/val",
                  train_batch_size=batch_size, eval_batch_size=batch_size, num_workers=0,
                  normal_split_ratio=0.0, seed=42)

def train_padim(dataset: Path, out: Path) -> dict:
    from anomalib.engine import Engine
    from anomalib.models import Padim
    dm = _datamodule(dataset)
    model = Padim(backbone="resnet18", layers=["layer1"], pre_trained=False, n_features=10, visualizer=False)
    engine = Engine(default_root_dir=str(out), logger=False, enable_checkpointing=True,
                    accelerator="cpu", devices=1, max_epochs=1, num_sanity_val_steps=0,
                    limit_train_batches=None)
    started = time.perf_counter(); engine.fit(model=model, datamodule=dm)
    exported = engine.export(model=model, export_type="openvino", export_root=str(out / "openvino"),
                             model_file_name="dryrun_inspector", input_size=(256, 256))
    ckpts = sorted(out.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return {"name": "Padim", "fit_seconds": time.perf_counter() - started,
            "checkpoint": str(ckpts[-1]) if ckpts else None, "openvino": str(exported),
            "anomalib": "2.7.0", "trained_on": "good/train only"}

def compare_anomalib(dataset: Path, out: Path) -> list[dict]:
    """Run the three requested Anomalib families on the same data."""
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd, Padim, Patchcore
    dm = _datamodule(dataset)
    specs = [("Padim", lambda: Padim(backbone="resnet18", layers=["layer1"], pre_trained=False, n_features=10, visualizer=False)),
             ("PatchCore", lambda: Patchcore(backbone="resnet18", layers=["layer1"], pre_trained=False, coreset_sampling_ratio=0.01, num_neighbors=3, visualizer=False)),
             # EfficientAD requires a one-image train loader and expects raw
             # tensors; Folder's default normalization would fail its hook.
             ("EfficientAD", lambda: EfficientAd(model_size="small", pre_processor=False, visualizer=False))]
    results = []
    for name, factory in specs:
        row = {"name": name, "dataset": "same good-only train / validation", "status": "started"}
        try:
            model = factory(); root = out / "comparison" / name
            run_dm = _datamodule(dataset, batch_size=1) if name == "EfficientAD" else dm
            engine = Engine(default_root_dir=str(root), logger=False, enable_checkpointing=True,
                            accelerator="cpu", devices=1, max_epochs=1, num_sanity_val_steps=0, limit_train_batches=1)
            t = time.perf_counter(); engine.fit(model=model, datamodule=run_dm); row["fit_seconds"] = time.perf_counter() - t
            row["validation"] = engine.test(model=model, datamodule=run_dm, ckpt_path=None)
            try:
                row["openvino"] = str(engine.export(model=model, export_type="openvino", export_root=str(root / "openvino"), model_file_name=name.lower(), input_size=(256, 256)))
                row["cpu_latency_ms"] = openvino_latency(Path(row["openvino"]))
            except Exception as exc: row["export_error"] = f"{type(exc).__name__}: {exc}"
            # The milestone compares deployable CPU cost, not just Lightning
            # timing. Only report it when export produced a concrete model.
            if "openvino" in row:
                exported = Path(row["openvino"])
                if exported.exists():
                    row["latency_cpu"] = _latency(exported)
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "error"; row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
    (out / "model_comparison.json").write_text(json.dumps(results, indent=2, default=str)); return results

def openvino_latency(model_path: Path, threads: int = 1) -> dict:
    core = ov.Core(); compiled = core.compile_model(core.read_model(str(model_path)), "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads}); req = compiled.create_infer_request(); inp = compiled.input(0); shape = [int(d.get_length()) if d.is_static else 1 for d in inp.partial_shape]; x = np.zeros(shape, np.float32)
    for _ in range(10): req.infer({inp: x})
    samples = []
    for _ in range(50):
        t = time.perf_counter_ns(); req.infer({inp: x}); samples.append((time.perf_counter_ns() - t) / 1e6)
    return {"threads": threads, "median_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)), "n": len(samples), "openvino": ov.__version__}

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", type=Path, default=Path("/bulk2/runs/dryrun/m2/data")); ap.add_argument("--out", type=Path, default=Path("/bulk2/runs/dryrun/m2/model")); ap.add_argument("--render", action="store_true"); ap.add_argument("--compare", action="store_true")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); report = {}
    if a.render or not (a.dataset / "good/train").exists(): report["dataset"] = build_dataset(a.dataset)
    report["training"] = train_padim(a.dataset, a.out)
    if a.compare: report["comparison"] = compare_anomalib(a.dataset, a.out)
    (a.out / "training.json").write_text(json.dumps(report, indent=2, default=str)); print(json.dumps(report, indent=2, default=str))
if __name__ == "__main__": main()
