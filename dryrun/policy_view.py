"""Shared camera preprocessing for the cropped policy experiment.

The top camera sees a large amount of empty table.  The crop is expressed as
fractions of the rendered square so recording and closed-loop evaluation use
the same pixels at any render resolution.  Bounds include both side bowls and
the full loose work envelope, with a small margin for seeded variation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


# At the 224 px full view this is approximately x=[-0.25, 0.35] m and
# y=[-0.40, 0.40] m for the compiled top camera.
WORK_AREA_CROP = (0.09, 0.10, 0.70, 0.90)  # left, top, right, bottom


def crop_work_area(frame: Any, output_size: int = 224) -> np.ndarray:
    """Crop the top view to the work area, then resize to a square."""

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got {array.shape}")
    height, width = array.shape[:2]
    left, top, right, bottom = WORK_AREA_CROP
    x0 = max(0, min(width - 1, int(round(left * width))))
    x1 = max(x0 + 1, min(width, int(round(right * width))))
    y0 = max(0, min(height - 1, int(round(top * height))))
    y1 = max(y0 + 1, min(height, int(round(bottom * height))))
    cropped = array[y0:y1, x0:x1]
    image = Image.fromarray(np.asarray(cropped, dtype=np.uint8), mode="RGB")
    image = image.resize((int(output_size), int(output_size)), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def resize_rgb(frame: Any, output_size: int = 224) -> np.ndarray:
    """Resize an RGB frame without applying the top-view crop."""

    array = np.asarray(frame)
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    return np.asarray(image.resize((int(output_size), int(output_size)), Image.Resampling.BILINEAR), dtype=np.uint8)


__all__ = ["WORK_AREA_CROP", "crop_work_area", "resize_rgb"]
