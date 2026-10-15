"""
Stage 3: layout detection with DocLayout-YOLO, then region building (stages/regions.py):
duplicate removal, overlap splitting at whitespace, recovery of missed text and XY-cut
reading order. config.region_builder = "legacy" uses the original paper's post-processing
from function_modules instead.

The model is loaded once and used from a single worker thread. Inference is about 0.2s
per page on an M-series Mac, so this stage is never the bottleneck and needs no batching.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from newspaper_pipeline.config import PipelineConfig
from newspaper_pipeline.stages.regions import build_regions

logger = logging.getLogger(__name__)


class LayoutDetector:
    """DocLayout-YOLO wrapper that turns a processed page into post-processed regions with crops on disk."""

    def __init__(self, config: PipelineConfig) -> None:
        # Imported lazily so worker processes that never run layout don't pay for torch
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        try:
            # The freshness check against the Hub can take 30s+ when it is slow or rate limited;
            # a cached copy is all we need
            weights = hf_hub_download(repo_id=config.yolo_repo, filename=config.yolo_file, local_files_only=True)
        except Exception:
            logger.info(f"Downloading {config.yolo_repo}/{config.yolo_file}")
            weights = hf_hub_download(repo_id=config.yolo_repo, filename=config.yolo_file)
        self.model = YOLOv10(weights)
        self.config = config
        self.device = config.yolo_device
        if self.device == "mps":
            import torch

            if not torch.backends.mps.is_available():
                logger.warning("MPS not available, running DocLayout-YOLO on CPU")
                self.device = "cpu"

    def detect(self, page_id: str, processed_path: str) -> pd.DataFrame:
        """
        Detect, post-process and crop regions for one page.

        Args:
            page_id: Page identifier
            processed_path: Pre-processed greyscale page

        Returns:
            DataFrame of regions in reading order, including x1..y2 (processed pixels),
            class, box_page_id, reading_order, recovered, crop_path and crop_pad (white
            margin added around each crop; crop pixel (pad, pad) is region pixel (x1, y1))
        """
        gray = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Could not read {processed_path}")
        height, width = gray.shape
        result = self.model.predict(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
            imgsz=self.config.yolo_image_size,
            conf=self.config.yolo_conf,
            device=self.device,
            verbose=False,
        )[0]

        rows: List[Dict] = []
        for (x1, y1, x2, y2), cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            name = result.names[int(cls)]
            rows.append(
                {
                    "filename": f"{page_id}.png",
                    "page_id": page_id,
                    "class": name,  # raw DocStructBench label; mapped by the region builder
                    "confidence": float(conf),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "page_height": height,
                    "page_width": width,
                }
            )

        if not rows:
            logger.warning(f"{page_id}: no layout regions detected")
            return pd.DataFrame(columns=["page_id", "box_page_id", "class", "reading_order", "x1", "y1", "x2", "y2", "crop_path"])

        if self.config.region_builder == "legacy":
            regions = self._legacy_regions(pd.DataFrame(rows), width, height)
        else:
            regions = build_regions(pd.DataFrame(rows), gray)

        crop_dir = Path(processed_path).parent / "regions"
        crop_dir.mkdir(exist_ok=True)
        crop_paths = []
        for row in regions.itertuples():
            x1, y1, x2, y2 = int(row.x1), int(row.y1), int(np.ceil(row.x2)), int(np.ceil(row.y2))
            path = crop_dir / f"{row.reading_order:04d}_{row.box_page_id}.png"
            # A white margin (not page context, which would pull in neighbouring text) helps
            # Tesseract and chandra with tightly cropped display type
            pad = self.config.crop_padding
            crop = cv2.copyMakeBorder(gray[y1:y2, x1:x2], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            cv2.imwrite(str(path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            crop_paths.append(str(path))
        regions["x1"] = regions["x1"].astype(int)
        regions["y1"] = regions["y1"].astype(int)
        regions["x2"] = np.ceil(regions["x2"]).astype(int)
        regions["y2"] = np.ceil(regions["y2"]).astype(int)
        regions["crop_path"] = crop_paths
        regions["crop_pad"] = self.config.crop_padding
        return regions

    @staticmethod
    def _legacy_regions(detections: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
        """The original paper's post-processing (function_modules.bbox_functions), kept for comparison."""
        from function_modules.bbox_functions import postprocess_bbox

        detections = detections.copy()
        detections["class"] = np.where(detections["class"] == "plain text", "text", detections["class"])
        regions = postprocess_bbox(detections, 10, width_multiplier=1.5, remove_abandon=True, fill_columns=True)
        regions = regions.sort_values("reading_order").reset_index(drop=True)
        regions[["x1", "x2"]] = regions[["x1", "x2"]].clip(0, width)
        regions[["y1", "y2"]] = regions[["y1", "y2"]].clip(0, height)
        regions["recovered"] = False
        return regions[(regions["x2"] - regions["x1"] >= 2) & (regions["y2"] - regions["y1"] >= 2)].reset_index(drop=True)


def silence_bbox_progress_bars() -> None:
    """function_modules.bbox_functions uses tqdm per page; with many concurrent pages the bars are just noise."""
    os.environ.setdefault("TQDM_DISABLE", "1")
