"""
Coordinate bookkeeping between the original scan and the pre-processed page.

Pre-processing crops away scanner borders and rotates the page to remove skew, but the
IIIF canvas (and Cantaloupe) use the original scan. Every geometric change is recorded
as one 2x3 affine matrix mapping original -> processed pixels, so any box found on the
processed page (or in a crop of it) can be mapped back onto the canvas.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2
XYWH = Tuple[int, int, int, int]


def _to_3x3(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([matrix, [0.0, 0.0, 1.0]])


def crop_matrix(x0: float, y0: float) -> np.ndarray:
    """Affine matrix for cropping at offset (x0, y0): the crop's origin moves to (0, 0)."""
    return np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0]])


def rotation_matrix(angle_deg: float, width: int, height: int) -> Tuple[np.ndarray, int, int]:
    """
    Affine matrix rotating an image about its centre, with the canvas expanded so nothing is clipped.

    Uses the same convention as cv2.getRotationMatrix2D (positive angle = counter-clockwise).

    Returns:
        (matrix, new_width, new_height)
    """
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    new_w = int(np.ceil(abs(width * cos) + abs(height * sin)))
    new_h = int(np.ceil(abs(width * sin) + abs(height * cos)))
    cx, cy = width / 2.0, height / 2.0
    matrix = np.array(
        [
            [cos, sin, (1 - cos) * cx - sin * cy + (new_w / 2.0 - cx)],
            [-sin, cos, sin * cx + (1 - cos) * cy + (new_h / 2.0 - cy)],
        ]
    )
    return matrix, new_w, new_h


@dataclass
class PageTransform:
    """
    Mapping from original scan pixels to processed page pixels.

    Attributes:
        matrix: 2x3 affine matrix, original -> processed
        original_size: (width, height) of the original scan
        processed_size: (width, height) of the processed page
        angle: deskew rotation applied, in degrees (for reporting)
        content_box: crop applied before rotation, in original pixels (x1, y1, x2, y2)
    """

    matrix: np.ndarray
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    angle: float = 0.0
    content_box: Box = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def identity(cls, width: int, height: int) -> "PageTransform":
        return cls(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (width, height), (width, height), 0.0, (0, 0, width, height))

    def then(self, matrix: np.ndarray, new_size: Tuple[int, int]) -> "PageTransform":
        """Compose another original->processed step after this one."""
        combined = (_to_3x3(matrix) @ _to_3x3(self.matrix))[:2]
        return PageTransform(combined, self.original_size, new_size, self.angle, self.content_box)

    @property
    def inverse(self) -> np.ndarray:
        """2x3 affine matrix, processed -> original."""
        return np.linalg.inv(_to_3x3(self.matrix))[:2]

    def points_to_original(self, points: np.ndarray) -> np.ndarray:
        """Map an (N, 2) array of processed-page points onto the original scan."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        return pts @ self.inverse[:, :2].T + self.inverse[:, 2]

    def points_to_processed(self, points: np.ndarray) -> np.ndarray:
        """Map an (N, 2) array of original-scan points onto the processed page."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        return pts @ self.matrix[:, :2].T + self.matrix[:, 2]

    def box_to_original(self, box: Box, offset: Tuple[float, float] = (0.0, 0.0)) -> XYWH:
        """
        Map an axis-aligned processed-page box to an axis-aligned box on the original scan.

        Rotated boxes become the smallest axis-aligned box containing all four corners,
        clamped to the scan.

        Args:
            box: (x1, y1, x2, y2) in processed pixels, or in crop pixels when offset is given
            offset: position of the crop on the processed page

        Returns:
            (x, y, w, h) in integer original-scan pixels, w and h at least 1
        """
        x1, y1, x2, y2 = box
        ox, oy = offset
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float) + [ox, oy]
        mapped = self.points_to_original(corners)
        width, height = self.original_size
        left = int(np.clip(np.floor(mapped[:, 0].min()), 0, width - 1))
        top = int(np.clip(np.floor(mapped[:, 1].min()), 0, height - 1))
        right = int(np.clip(np.ceil(mapped[:, 0].max()), left + 1, width))
        bottom = int(np.clip(np.ceil(mapped[:, 1].max()), top + 1, height))
        return left, top, right - left, bottom - top

    def to_json(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "original_size": list(self.original_size),
            "processed_size": list(self.processed_size),
            "angle": self.angle,
            "content_box": list(self.content_box),
        }

    @classmethod
    def from_json(cls, data: dict) -> "PageTransform":
        return cls(
            np.array(data["matrix"], dtype=float),
            tuple(data["original_size"]),
            tuple(data["processed_size"]),
            float(data.get("angle", 0.0)),
            tuple(data.get("content_box", (0, 0, *data["original_size"]))),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PageTransform":
        return cls.from_json(json.loads(Path(path).read_text()))


def union_box(boxes: Iterable[Box]) -> Box:
    """Smallest box containing all the given (x1, y1, x2, y2) boxes."""
    arr = np.asarray(list(boxes), dtype=float)
    return float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].max()), float(arr[:, 3].max())
