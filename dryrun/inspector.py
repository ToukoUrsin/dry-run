"""Small inference API shared by the benchmark, web page and photo script."""
from __future__ import annotations
import json, os
from pathlib import Path
import cv2, numpy as np, openvino as ov

# The real-present-frame model is the benchmark and page default.  Keep this
# on the /bulk3 volume because /bulk2 is a compatibility symlink and may be
# full; Modal receives the same files at its mounted /runs path.
DEFAULT_MODEL = Path("/bulk3/runs/dryrun/m2_rebuild/default_model/openvino/weights/openvino/dryrun_inspector.xml")

class Inspector:
    """Load an OpenVINO inspector once and classify RGB frames."""
    def __init__(self, model_path: str | Path | None = None, threshold: float | None = None, threads: int = 1):
        self.model_path = Path(model_path or os.environ.get("DRYRUN_INSPECTOR_MODEL", DEFAULT_MODEL))
        if not self.model_path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {self.model_path}")
        threads = int(threads)
        if threads < 1:
            raise ValueError("threads must be at least 1")
        self.core = ov.Core()
        config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads}
        self.compiled = self.core.compile_model(self.core.read_model(str(self.model_path)), "CPU", config)
        self.request = self.compiled.create_infer_request(); self.input = self.compiled.input(0)
        # The exported model is currently 256x256. Keep this derived from the
        # model where possible so a future export with another fixed size does
        # not silently reshape its anomaly map incorrectly.
        ps = self.input.partial_shape
        self.batch_size = int(ps[0].get_length()) if len(ps) == 4 and ps[0].is_static else 1
        self.input_height = int(ps[2].get_length()) if len(ps) == 4 and ps[2].is_static else 256
        self.input_width = int(ps[3].get_length()) if len(ps) == 4 and ps[3].is_static else 256
        self.score_output = self.compiled.output("pred_score")
        self.heatmap_output = self.compiled.output("anomaly_map")
        sidecar = self.model_path.with_name("threshold.json")
        selected = threshold if threshold is not None else (json.loads(sidecar.read_text()).get("threshold", .5) if sidecar.exists() else .5)
        self.threshold = float(selected)
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")

    @staticmethod
    def _rgb_input(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("frame must be HxWx3 RGB")
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError("frame must contain numeric RGB values")
        is_float = np.issubdtype(arr.dtype, np.floating)
        arr = np.asarray(arr, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("frame contains NaN or infinite values")
        # Accept both the uint8 convention used by the station and float RGB
        # in the common [0, 1] range.  The old code divided [0, 1] input by
        # 255 a second time, producing nearly black inference frames.
        if is_float and arr.size and float(arr.max()) <= 1.0:
            arr *= 255.0
        return np.ascontiguousarray(np.clip(arr, 0.0, 255.0) / 255.0, dtype=np.float32)

    def inspect(self, frame: np.ndarray) -> dict:
        """Return score, verdict and a uint8 heat map for an RGB frame."""
        frame = self._rgb_input(frame)
        h, w = frame.shape[:2]; x = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)[None]
        if self.batch_size > 1: x = np.repeat(x, self.batch_size, axis=0)
        outputs = self.request.infer({self.input: x})
        score = float(np.asarray(outputs[self.score_output]).reshape(-1)[0])
        amap = np.asarray(outputs[self.heatmap_output])
        if amap.ndim == 4: amap = amap[0, 0]
        elif amap.ndim == 3: amap = amap[0]
        amap = np.squeeze(amap)
        if amap.ndim != 2:
            raise RuntimeError(f"unexpected anomaly_map shape: {np.asarray(outputs[self.heatmap_output]).shape}")
        amap = cv2.resize(amap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        lo, hi = float(amap.min()), float(amap.max()); heat = np.zeros_like(amap, dtype=np.uint8) if hi <= lo else np.uint8(np.clip((amap - lo) * 255 / (hi - lo), 0, 255))
        # StationEnv can emit its physical ``present`` event as soon as a
        # held brick first enters the presentation envelope.  That frame is
        # before pose A/B and contains only a small, distant brick; it must
        # not decide the bowl.  Use the fixed-camera border colour to measure
        # the largest foreground component.  The calibrated full present
        # crop occupies materially more than this, while the pre-present
        # frame is about 10% of the image.  Return an explicit invalid verdict
        # so rollout keeps the event auditable without letting it override the
        # later full-frame decisions.
        border = np.concatenate((
            frame[: max(1, h // 32)].reshape(-1, 3),
            frame[-max(1, h // 32):].reshape(-1, 3),
            frame[:, : max(1, w // 32)].reshape(-1, 3),
            frame[:, -max(1, w // 32):].reshape(-1, 3),
        ), axis=0)
        background = np.median(border, axis=0)
        foreground = (np.linalg.norm(frame - background, axis=2) > (80.0 / 255.0)).astype(np.uint8)
        nlabels, _, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        bbox_fraction = 0.0
        if nlabels > 1:
            largest = stats[1:][np.argmax(stats[1:, cv2.CC_STAT_AREA])]
            bbox_fraction = float(largest[cv2.CC_STAT_WIDTH] * largest[cv2.CC_STAT_HEIGHT]) / float(h * w)
        if bbox_fraction < 0.13:
            return {"score": score, "threshold": self.threshold, "verdict": "invalid", "heatmap": heat,
                    "frame_valid": False, "reason": "underfilled_pre_present",
                    "foreground_bbox_fraction": bbox_fraction}
        return {"score": score, "threshold": self.threshold, "verdict": "defective" if score >= self.threshold else "good", "heatmap": heat,
                "frame_valid": True, "foreground_bbox_fraction": bbox_fraction}
    __call__ = inspect
