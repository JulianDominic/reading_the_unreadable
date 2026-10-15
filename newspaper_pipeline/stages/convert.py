"""
Stage 1: decode a source scan (JPEG2000, TIFF, JPG, PNG) into a working PNG.

PNG sources are used in place without copying. The original pixel dimensions are
recorded because they define the IIIF canvas that Cantaloupe serves.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # newspaper scans are legitimately huge

logger = logging.getLogger(__name__)


@dataclass
class ConvertedPage:
    """Result of the convert stage."""

    page_id: str
    source_path: str
    source_relpath: str
    png_path: str
    width: int
    height: int
    converted: bool

    def to_dict(self) -> Dict:
        return asdict(self)


def to_8bit_rgb_or_l(image: Image.Image) -> Image.Image:
    """
    Normalise any Pillow image mode to 8-bit L or RGB.

    Archive TIFFs are frequently 16-bit greyscale, CMYK or palette images, and some JP2s carry an alpha band.
    """
    if image.mode in ("L", "RGB"):
        return image
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(image, dtype=np.float32)
        hi = arr.max() if arr.size else 1.0
        scale = 255.0 / (65535.0 if hi > 255 else max(hi, 1.0))
        return Image.fromarray(np.clip(arr * scale, 0, 255).astype(np.uint8), mode="L")
    if image.mode == "1":
        return image.convert("L")
    if image.mode in ("LA", "PA"):
        return image.convert("L")
    return image.convert("RGB")


def convert_page(source_path: str, source_relpath: str, page_id: str, work_dir: str) -> Dict:
    """
    Decode a scan and write it as PNG into the page's work folder.

    Args:
        source_path: Absolute path of the scan
        source_relpath: Path relative to the input folder (used as the Cantaloupe identifier)
        page_id: Unique page identifier
        work_dir: Root work folder; files go into work_dir/page_id

    Returns:
        ConvertedPage as a dict (dicts pickle cheaply between processes)
    """
    page_dir = Path(work_dir) / page_id
    page_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source_path)

    with Image.open(src) as image:
        if getattr(image, "n_frames", 1) > 1:
            logger.warning(f"{src.name} has {image.n_frames} frames; only the first is processed")
        width, height = image.size
        if src.suffix.lower() == ".png" and image.mode in ("L", "RGB"):
            return ConvertedPage(page_id, str(src), source_relpath, str(src), width, height, False).to_dict()
        image.load()
        normalised = to_8bit_rgb_or_l(image)
        png_path = page_dir / "page.png"
        # compress_level=1: these files are deleted after the run, so write speed matters more than size
        normalised.save(png_path, format="PNG", compress_level=1)

    return ConvertedPage(page_id, str(src), source_relpath, str(png_path), width, height, True).to_dict()
