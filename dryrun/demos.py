"""LeRobot demonstration recorder used by the MuJoCo and hardware rollouts.

The recorder deliberately keeps a brick cycle in memory until the expert has
finished it.  Failed cycles are therefore never written to the training set.
The rollout owns the physics and calls :meth:`record_step` at its 30 Hz
sampling boundary; this module only owns the on-disk LeRobot format.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FEATURES = {
    "observation.images.top": {"dtype": "image", "shape": (64, 64, 3)},
    "observation.images.inspection": {"dtype": "image", "shape": (64, 64, 3)},
    "observation.state": {"dtype": "float32", "shape": (6,)},
    "action": {"dtype": "float32", "shape": (6,)},
    # The inspector output is the only defect information exposed to policy.
    # 0 means not inspected yet, +1 good and -1 defective.
    "inspection.verdict": {"dtype": "int8", "shape": (1,)},
    "success": {"dtype": "bool", "shape": (1,)},
}


@dataclass
class Cycle:
    task: str
    frames: list[dict[str, Any]]


class DemoRecorder:
    """Write successful expert brick cycles as current-format LeRobot data.

    ``record_step`` accepts uint8 HWC images, six-joint state and six-joint
    target action.  ``inspection`` is optional before the present step and is
    represented by a zero verdict.  At the end of a cycle call ``end_cycle``
    with the physical success flag; unsuccessful cycles are discarded.
    """

    def __init__(self, root: str | Path, repo_id: str = "dryrun-so101", fps: int = 30):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        # ``LeRobotDataset.create`` owns creation of the dataset directory and
        # intentionally requires it not to exist.  Create only its parent so
        # accidental appends cannot silently produce a corrupt metadata tree.
        self.root.parent.mkdir(parents=True, exist_ok=True)
        if self.root.exists():
            raise FileExistsError(
                f"dataset root already exists: {self.root}; choose a new run directory "
                "or remove the old run explicitly"
            )
        self.repo_id = repo_id
        self.fps = int(fps)
        self._dataset = LeRobotDataset.create(
            repo_id,
            fps=self.fps,
            features=FEATURES,
            root=self.root,
            robot_type="so101",
            use_videos=False,
            image_writer_processes=0,
            image_writer_threads=0,
        )
        self._cycle: Cycle | None = None
        self.cycles_seen = 0
        self.cycles_written = 0
        self.frames_written = 0

    @property
    def dataset(self):
        return self._dataset

    def begin_cycle(self, task: str = "sort brick") -> None:
        if self._cycle is not None:
            raise RuntimeError("a demo cycle is already open")
        self._cycle = Cycle(task=str(task), frames=[])

    # Adapter used directly as ``record=recorder`` by dryrun.rollout.run.
    def cycle_start(self, brick: str | None = None, task: str = "sort brick") -> None:
        self.begin_cycle(task=f"{task} ({brick})" if brick is not None else task)

    def cycle_end(self, success: bool) -> bool:
        return self.end_cycle(bool(success))

    def record(self, payload: dict[str, Any]) -> None:
        """Consume the callback payload emitted at each 30 Hz boundary."""
        if payload.get("type") in {"cycle_start", "cycle_end"}:
            if payload["type"] == "cycle_start":
                self.cycle_start(payload.get("brick"))
            else:
                self.cycle_end(bool(payload.get("success", False)))
            return
        obs = payload["observation"]
        self.record_step(
            state=obs["joints"], action=payload["action"], top=obs["top"],
            inspection=obs.get("inspection"), verdict=int(payload.get("verdict", 0)),
            success=False,
        )

    @staticmethod
    def _image(value: Any, name: str) -> np.ndarray:
        image = np.asarray(value)
        if image.shape != (64, 64, 3):
            raise ValueError(f"{name} must have shape (64, 64, 3), got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def record_step(
        self,
        state: Any,
        action: Any,
        top: Any,
        inspection: Any | None = None,
        verdict: int = 0,
        success: bool = False,
    ) -> None:
        """Stage one 30 Hz frame for the open cycle.

        ``verdict`` is intentionally an explicit argument rather than a scene
        label.  A caller must pass 0 until inspection, then +1/-1 from the
        inspector camera.
        """
        if self._cycle is None:
            raise RuntimeError("call begin_cycle before record_step")
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.shape != (6,) or action.shape != (6,):
            raise ValueError(f"state/action must each have shape (6,), got {state.shape}/{action.shape}")
        if verdict not in (-1, 0, 1):
            raise ValueError("verdict must be -1, 0 or +1")
        if inspection is None:
            inspection = np.zeros((64, 64, 3), dtype=np.uint8)
        self._cycle.frames.append(
            {
                "observation.images.top": self._image(top, "top"),
                "observation.images.inspection": self._image(inspection, "inspection"),
                "observation.state": state,
                "action": action,
                "inspection.verdict": np.asarray([verdict], dtype=np.int8),
                "success": np.asarray([bool(success)], dtype=np.bool_),
            }
        )

    def end_cycle(self, success: bool) -> bool:
        """Commit a successful cycle and return whether it was written."""
        if self._cycle is None:
            raise RuntimeError("no demo cycle is open")
        cycle = self._cycle
        self._cycle = None
        self.cycles_seen += 1
        if not success or not cycle.frames:
            return False
        for frame in cycle.frames:
            frame["success"] = np.asarray([True], dtype=np.bool_)
            self._dataset.add_frame({**frame, "task": cycle.task})
        self._dataset.save_episode()
        self.cycles_written += 1
        self.frames_written += len(cycle.frames)
        return True

    def close(self) -> dict[str, int]:
        """Finalize metadata and return a compact write summary."""
        if self._cycle is not None:
            raise RuntimeError("cannot close with an open demo cycle")
        self._dataset.finalize()
        return {
            "cycles_seen": self.cycles_seen,
            "cycles_written": self.cycles_written,
            "frames_written": self.frames_written,
        }


__all__ = ["DemoRecorder", "FEATURES"]
