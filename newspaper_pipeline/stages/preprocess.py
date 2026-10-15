"""
Stage 2: page-level image clean-up before layout detection and OCR.

Operations, in order:
1. Border masking: find the paper against a darker scanner bed, crop to it and paint
   anything outside the paper (rulers, colour cards, bed) white.
2. Deskew: estimate the page skew once with a projection-profile search and rotate.
3. Enhancement: flatten the paper background (staining, uneven lighting), lift faint
   bleed-through to white with a levels curve and denoise with a bilateral filter.

Geometric steps (1, 2) are recorded in a PageTransform so results map back onto the original scan.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from newspaper_pipeline.transforms import Box, PageTransform, crop_matrix, rotation_matrix

logger = logging.getLogger(__name__)

ANALYSIS_WIDTH = 1500  # pixels; skew and border analysis run on a downscaled copy


def _downscale(gray: np.ndarray, target_width: int = ANALYSIS_WIDTH) -> Tuple[np.ndarray, float]:
    scale = min(1.0, target_width / gray.shape[1])
    if scale == 1.0:
        return gray, 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return small, scale


def find_paper(gray: np.ndarray, min_fraction: float = 0.5, max_fraction: float = 0.985) -> Optional[Tuple[Box, np.ndarray]]:
    """
    Locate the paper against a darker scanner background.

    Args:
        gray: Full resolution greyscale page
        min_fraction: Reject detections covering less of the image than this (likely a mistake)
        max_fraction: Treat detections covering more than this as "no border present"

    Returns:
        ((x1, y1, x2, y2) of the paper in full resolution pixels, full resolution paper mask), or None
    """
    small, scale = _downscale(gray)
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close text and rules into solid paper before looking for the outline
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    paper = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    fraction = (w * h) / float(small.shape[0] * small.shape[1])
    if fraction > max_fraction or fraction < min_fraction:
        return None

    mask_small = np.zeros_like(small)
    cv2.drawContours(mask_small, [largest], -1, 255, thickness=cv2.FILLED)
    # Pull the outline in by a few analysis pixels: the outline is only accurate to about one
    # downscaled pixel, and a surviving sliver of dark scanner bed (or the shadowed paper edge)
    # is a strong straight line that biases the skew estimate.
    inset = 3
    mask_small = cv2.erode(mask_small, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * inset + 1, 2 * inset + 1)))
    mask = cv2.resize(mask_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    box = (
        (x + inset) / scale,
        (y + inset) / scale,
        min((x + w - inset) / scale, gray.shape[1]),
        min((y + h - inset) / scale, gray.shape[0]),
    )
    return box, mask


def estimate_skew(gray: np.ndarray, max_angle: float = 5.0) -> float:
    """
    Estimate the rotation (degrees, cv2 convention) that makes text lines horizontal.

    Uses a coarse-to-fine projection profile search: the correct rotation gives the
    sharpest row profile of ink pixels, i.e. the highest variance of row sums.

    Args:
        gray: Greyscale page (any resolution)
        max_angle: Largest skew searched in either direction

    Returns:
        Angle to pass to rotation_matrix to straighten the page
    """
    small, _ = _downscale(gray)
    ink = cv2.adaptiveThreshold(small, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15).astype(np.float32)
    h, w = ink.shape
    centre = (w / 2.0, h / 2.0)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rotated.sum(axis=1)))

    coarse = np.arange(-max_angle, max_angle + 1e-9, 0.25)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.25, best + 0.25 + 1e-9, 0.02)
    return float(round(max(fine, key=score), 3))


def enhance(gray: np.ndarray, clahe_clip_limit: float = 0.0, denoise: bool = True) -> np.ndarray:
    """
    Flatten paper background, lift faint bleed-through to white and remove speckle.

    1. Background flattening divides by a smooth estimate of the paper colour, removing
       staining and lighting gradients.
    2. A levels curve maps the page's 25th grey percentile (clamped to 190-240) to white.
       Ink is far darker than that, while bleed-through from the reverse side sits just below paper.
    3. An edge-preserving bilateral filter removes speckle without blurring strokes.

    Tested against Tesseract word confidence on the 1874 sample page (30 regions, Sauvola):
    raw 81.6 -> flatten+levels+bilateral 82.8 mean confidence; CLAHE made it worse
    (73.1 at clip 1.5) because it amplifies bleed-through, so it is off by default.

    Args:
        gray: Greyscale page
        clahe_clip_limit: CLAHE clip limit; 0 disables CLAHE
        denoise: Apply the bilateral filter

    Returns:
        Enhanced greyscale page (uint8)
    """
    small, _ = _downscale(gray, 1000)
    # Dilation removes dark text so only the paper colour remains, then blur smooths it
    background = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    background = cv2.medianBlur(background, 21)
    background = cv2.resize(background, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    flattened = cv2.divide(gray, background, scale=255)

    white_point = float(np.clip(np.percentile(flattened[::8, ::8], 25), 190, 240))
    lut = np.clip(np.arange(256) * 255.0 / white_point, 0, 255).astype(np.uint8)
    enhanced = cv2.LUT(flattened, lut)

    if clahe_clip_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(16, 16))
        enhanced = clahe.apply(enhanced)

    if denoise:
        enhanced = cv2.bilateralFilter(enhanced, 5, 30, 5)
    return enhanced


def preprocess_page(
    page_id: str,
    png_path: str,
    work_dir: str,
    mask_borders: bool = True,
    deskew: bool = True,
    max_deskew_angle: float = 5.0,
    enhance_page: bool = True,
    clahe_clip_limit: float = 0.0,
    denoise: bool = True,
) -> Dict:
    """
    Run all page-level pre-processing and save the processed page and its transform.

    Args:
        page_id: Page identifier
        png_path: Converted page from the convert stage
        work_dir: Root work folder
        mask_borders: Crop to the paper and mask the scanner background
        deskew: Estimate and remove page skew
        max_deskew_angle: Largest skew searched (degrees)
        enhance_page: Apply background flattening, levels and (optionally) CLAHE and denoise
        clahe_clip_limit: CLAHE clip limit; 0 disables it
        denoise: Apply the bilateral denoise

    Returns:
        Dict with processed_path, transform_path, angle, content_box and processed size
    """
    page_dir = Path(work_dir) / page_id
    page_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read {png_path}")
    height, width = gray.shape
    transform = PageTransform.identity(width, height)

    if mask_borders:
        found = find_paper(gray)
        if found is not None:
            (x1, y1, x2, y2), mask = found
            gray = gray.copy()
            gray[mask == 0] = 255
            x1, y1, x2, y2 = int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))
            gray = gray[y1:y2, x1:x2]
            transform = transform.then(crop_matrix(x1, y1), (x2 - x1, y2 - y1))
            transform.content_box = (x1, y1, x2, y2)
            logger.info(f"{page_id}: masked scanner border, paper is {x2 - x1}x{y2 - y1} at ({x1}, {y1})")

    if deskew:
        angle = estimate_skew(gray, max_deskew_angle)
        if abs(angle) >= 0.05:
            h, w = gray.shape
            matrix, new_w, new_h = rotation_matrix(angle, w, h)
            gray = cv2.warpAffine(gray, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=255)
            transform = transform.then(matrix, (new_w, new_h))
            transform.angle = angle
        logger.info(f"{page_id}: skew {angle:+.2f} deg{'' if abs(angle) >= 0.05 else ' (not corrected)'}")

    if enhance_page:
        gray = enhance(gray, clahe_clip_limit, denoise)

    processed_path = page_dir / "processed.png"
    transform_path = page_dir / "transform.json"
    cv2.imwrite(str(processed_path), gray, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    transform.save(transform_path)

    return {
        "processed_path": str(processed_path),
        "transform_path": str(transform_path),
        "angle": transform.angle,
        "content_box": list(transform.content_box),
        "processed_size": list(transform.processed_size),
    }
