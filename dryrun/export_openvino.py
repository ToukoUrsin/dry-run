"""Export the final ACT checkpoint and measure OpenVINO CPU inference."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch, openvino as ov
from .train_policy import load_policy, ROOT


class ACTExport(torch.nn.Module):
    def __init__(self, policy): super().__init__(); self.model = policy.model
    def forward(self, top, inspection, state):
        actions, _ = self.model({"observation.images": [top, inspection], "observation.state": state})
        return actions[:, 0, :]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", type=Path, default=ROOT / "act-final.pt"); ap.add_argument("--out", type=Path, default=ROOT / "openvino"); ap.add_argument("--iterations", type=int, default=20); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    policy = load_policy(a.checkpoint); wrapper = ACTExport(policy).eval(); top = torch.zeros(1, 3, 64, 64); inspection = torch.zeros_like(top); state = torch.zeros(1, 7)
    onnx = a.out / "policy.onnx"; torch.onnx.export(wrapper, (top, inspection, state), onnx, input_names=["top", "inspection", "state"], output_names=["action"], opset_version=17)
    core = ov.Core(); xml = a.out / "policy.xml"; ov_model = ov.convert_model(str(onnx)); ov.save_model(ov_model, str(xml)); timings = {}
    inputs = {"top": np.zeros((1,3,64,64), np.float32), "inspection": np.zeros((1,3,64,64), np.float32), "state": np.zeros((1,7), np.float32)}
    for threads in (1, 4):
        compiled = core.compile_model(str(xml), "CPU", {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY"}); req = compiled.create_infer_request();
        for _ in range(3): req.infer(inputs)
        start = time.perf_counter()
        for _ in range(a.iterations): req.infer(inputs)
        timings[str(threads)] = {"iterations": a.iterations, "latency_ms": (time.perf_counter() - start) * 1000 / a.iterations}
    result = {"checkpoint": str(a.checkpoint), "onnx": str(onnx), "openvino": str(xml), "device": "CPU", "threads": timings, "commit": os.popen("git rev-parse HEAD").read().strip()}
    (a.out / "timing.json").write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
