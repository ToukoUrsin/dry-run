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
    brick: int | None = None
    success: bool = False


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
        self._cycles: list[Cycle] = []
        self._seen_events: set[tuple[str, str, str]] = set()
        self._scene_success = False
        self._finalized = False
        # ``rollout.run`` opens one scene-level boundary and emits brick
        # boundaries as physics ``reach`` events.  Keep that scene staged
        # until its result is available: a scene-level success must never
        # turn every brick into a successful training episode.
        self._scene_open = False
        self.cycles_seen = 0
        self.cycles_written = 0
        self.frames_written = 0

    @property
    def dataset(self):
        return self._dataset

    def begin_cycle(self, task: str = "sort brick") -> None:
        if self._cycle is not None:
            raise RuntimeError("a demo cycle is already open")
        self._cycles = []
        self._seen_events.clear()
        self._scene_success = False
        self._finalized = False
        self._scene_open = True
        self._cycle = Cycle(task=str(task), frames=[])

    # Adapter used directly as ``record=recorder`` by dryrun.rollout.run.
    def cycle_start(self, brick: str | None = None, task: str = "sort brick") -> None:
        self.begin_cycle(task=f"{task} ({brick})" if brick is not None else task)
        if brick is not None:
            self._cycle.brick = self._brick_id(brick)
            # A caller that names the brick is using the direct, one-brick
            # recorder API rather than the scene-level rollout adapter.
            self._scene_open = False

    def cycle_end(self, success: bool) -> bool:
        # The rollout's cycle_end is scene-scoped.  Defer all writes until
        # ``finalize_result`` can match each staged cycle to result["correct"].
        if self._scene_open:
            self._stage_current()
            self._scene_success = bool(success)
            return False
        return self.end_cycle(bool(success))

    @staticmethod
    def _brick_id(value: Any) -> int | None:
        """Normalize body ids such as ``3`` and ``"brick_3"``."""
        if value is None:
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        text = str(value)
        try:
            return int(text)
        except ValueError:
            if text.startswith("brick_"):
                try:
                    return int(text.rsplit("_", 1)[1])
                except ValueError:
                    pass
        return None

    @staticmethod
    def _event_key(event: dict[str, Any]) -> tuple[str, str, str]:
        return (str(event.get("event", "")), str(event.get("brick", "")), str(event.get("t", "")))

    def _start_brick(self, brick: int, task: str = "sort brick") -> None:
        brick_id = self._brick_id(brick)
        if brick_id is None:
            return
        if self._cycle is not None:
            # Avoid splitting a cycle if a cumulative event tail repeats the
            # same reach event.  The event-key de-duplication normally covers
            # this, but this guard also handles callers without timestamps.
            if self._cycle.brick == brick_id:
                return
            if self._cycle.frames:
                self._cycles.append(self._cycle)
        self._cycle = Cycle(task=f"{task} ({brick_id})", frames=[], brick=brick_id)

    def _consume_events(self, events: Any) -> None:
        for event in events or []:
            if not isinstance(event, dict):
                continue
            key = self._event_key(event)
            if key in self._seen_events:
                continue
            self._seen_events.add(key)
            if event.get("event") == "reach" and event.get("brick") is not None:
                self._start_brick(int(event["brick"]))

    def record(self, payload: dict[str, Any]) -> None:
        """Consume the callback payload emitted at each 30 Hz boundary."""
        if payload.get("type") in {"cycle_start", "cycle_end"}:
            if payload["type"] == "cycle_start":
                self.cycle_start(payload.get("brick"), str(payload.get("task", "sort brick")))
            else:
                self.cycle_end(bool(payload.get("success", False)))
            return
        self._consume_events(payload.get("events", []))
        if self._cycle is None:
            self._cycle = Cycle(task="sort brick", frames=[])
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

    def _stage_current(self) -> None:
        if self._cycle is None:
            return
        if self._cycle.frames:
            self._cycles.append(self._cycle)
        self._cycle = None

    def _write_cycle(self, cycle: Cycle) -> bool:
        if not cycle.success or not cycle.frames:
            return False
        for frame in cycle.frames:
            frame["success"] = np.asarray([True], dtype=np.bool_)
            self._dataset.add_frame({**frame, "task": cycle.task})
        self._dataset.save_episode()
        self.cycles_written += 1
        self.frames_written += len(cycle.frames)
        return True

    def end_cycle(self, success: bool) -> bool:
        """Commit a direct, already-scoped brick cycle.

        This is retained for the older callback API used by render fixtures.
        Scene rollouts call ``cycle_end`` and then ``finalize_result`` so they
        can be segmented by brick.
        """
        if self._cycle is None:
            raise RuntimeError("no demo cycle is open")
        cycle = self._cycle
        self._cycle = None
        self.cycles_seen += 1
        cycle.success = bool(success)
        return self._write_cycle(cycle)

    def finalize_result(self, result: dict[str, Any]) -> dict[str, int]:
        """Finalize the staged scene against physical per-brick results.

        ``StationEnv.result`` returns body ids in ``correct``.  Only cycles
        whose ``brick`` id is in that set are written.  A legacy scene with no
        reach events is handled as one unnamed cycle and is written only when
        the scene-level result is truthy.
        """
        if self._finalized:
            return {"cycles_seen": self.cycles_seen, "cycles_written": self.cycles_written,
                    "frames_written": self.frames_written}
        self._stage_current()
        correct_values = result.get("correct", []) if isinstance(result, dict) else []
        correct = {self._brick_id(value) for value in (correct_values or [])}
        correct.discard(None)
        scene_ok = bool(correct) or bool(result.get("success", False)) if isinstance(result, dict) else False
        # If no physical ids are available, preserve the old one-cycle
        # callback behavior without claiming a multi-brick scene succeeded.
        for cycle in self._cycles:
            if cycle.brick is None:
                cycle.success = bool(scene_ok and len(self._cycles) == 1)
            else:
                cycle.success = cycle.brick in correct
            self.cycles_seen += 1
            self._write_cycle(cycle)
        self._cycles = []
        self._finalized = True
        self._scene_open = False
        return {"cycles_seen": self.cycles_seen, "cycles_written": self.cycles_written,
                "frames_written": self.frames_written}

    def close(self) -> dict[str, int]:
        """Finalize metadata and return a compact write summary."""
        # A caller that used the scene adapter must explicitly provide the
        # physical result.  Discarding an unfinalized scene is safer than
        # silently training on a scene-level success flag.
        if self._cycle is not None or self._cycles:
            if not self._finalized:
                self._stage_current()
                self._cycles = []
            else:
                raise RuntimeError("cannot close with an open demo cycle")
        self._dataset.finalize()
        return {
            "cycles_seen": self.cycles_seen,
            "cycles_written": self.cycles_written,
            "frames_written": self.frames_written,
        }


__all__ = ["DemoRecorder", "FEATURES"]
