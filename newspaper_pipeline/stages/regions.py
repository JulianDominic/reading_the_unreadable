"""
Turn raw DocLayout-YOLO detections into clean, non-overlapping OCR regions in reading order.

This replaces function_modules.bbox_functions.postprocess_bbox for the IIIF pipeline (the
old function is left untouched for run_sync_pipeline.py and the published results). On
the 1874 sample page the old post-processing covered 82% of the region area two or more
times: boxes it could not fit into an equal-width column grid were assigned "column 0"
and stretched to the full print width, and every box was extended down to the next box
in its column. Each duplicated area was transcribed twice.

Steps:
1. Normalise classes (DocStructBench has ten; we keep text, title, figure, table) and drop "abandon".
2. Drop near-duplicate and nested detections.
3. Split partial overlaps along the whitespace row/column with the least ink, so cuts
   fall between text lines rather than through them.
4. Recover text YOLO missed: ink not covered by any region is grouped into new text regions.
5. Reading order by recursive XY-cut, trying vertical gutters before horizontal gaps,
   which matches newspaper layout (masthead, then columns, with headings that span
   several columns read before the text beneath them).

Figures are allowed to overlap text: they are described rather than transcribed and
never get word annotations, so they cannot duplicate text.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLASS_MAP = {
    "plain text": "text",
    "text": "text",
    "title": "title",
    "figure": "figure",
    "table": "table",
    "figure_caption": "text",
    "table_caption": "text",
    "table_footnote": "text",
    "isolate_formula": "text",
    "formula_caption": "text",
}

INK_SCALE = 4  # the ink mask is built at 1/4 resolution; plenty for locating text blocks


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float, float]:
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """
    Binary mask (1 = ink) of text-like marks at 1/INK_SCALE resolution.

    Printed rules (column separators, double rules under the masthead, borders) are removed
    so they don't count as content: as long straight segments found by morphological
    opening, and as connected components far longer than any letter and only a few pixels
    thick, which catches textured or broken rules the opening misses.
    """
    small = cv2.resize(gray, (gray.shape[1] // INK_SCALE, gray.shape[0] // INK_SCALE), interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    v_rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    ink = (ink & ~(h_rules | v_rules)).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    rule_like = ((w >= 25) & (w >= 8 * h)) | ((h >= 25) & (h >= 8 * w))
    rule_like[0] = False
    if rule_like.any():
        ink[rule_like[labels]] = 0
    return ink


def _ink_profile(ink: np.ndarray, box: Sequence[float], axis: int) -> np.ndarray:
    x1, y1, x2, y2 = (int(v // INK_SCALE) for v in box)
    window = ink[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
    return window.sum(axis=axis)  # axis=1: per row, axis=0: per column


def _best_cut(lo: float, hi: float, profile: np.ndarray) -> float:
    """Full-resolution coordinate between lo and hi with the least ink, preferring the middle on ties."""
    if profile.size == 0 or hi - lo < 2 * INK_SCALE:
        return (lo + hi) / 2.0
    centre = (profile.size - 1) / 2.0
    penalty = np.abs(np.arange(profile.size) - centre) / max(profile.size, 1) * 0.5  # tie-breaker only
    index = int(np.argmin(profile + penalty))
    return float(np.clip(lo + (index + 0.5) * INK_SCALE, lo, hi))


def remove_nested(boxes: List[Dict], contain_threshold: float = 0.6) -> List[Dict]:
    """
    Drop detections that are mostly inside another detection of a text-like class.

    The smaller box goes when contain_threshold of its area lies inside the larger one; for
    near-identical boxes the lower-confidence one goes. Figures are never removed and never
    remove others.
    """
    keep = [True] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda i: -_area(boxes[i]["bbox"]))
    for a_pos, i in enumerate(order):
        if not keep[i] or boxes[i]["class"] == "figure":
            continue
        for j in order[a_pos + 1:]:
            if not keep[j] or boxes[j]["class"] == "figure":
                continue
            inter = _area(_intersection(boxes[i]["bbox"], boxes[j]["bbox"]))
            if inter <= 0:
                continue
            area_i, area_j = _area(boxes[i]["bbox"]), _area(boxes[j]["bbox"])
            if inter / area_j >= contain_threshold:
                if area_j / area_i > 0.8 and boxes[j]["confidence"] > boxes[i]["confidence"]:
                    keep[i] = False  # near-identical: keep the more confident one
                    break
                keep[j] = False
    return [b for b, k in zip(boxes, keep) if k]


def split_overlaps(boxes: List[Dict], ink: np.ndarray, min_size: float = 8.0, max_rounds: int = 20) -> List[Dict]:
    """
    Remove partial overlaps between non-figure boxes by moving one shared edge to the
    emptiest row (boxes stacked vertically) or column (boxes side by side) inside the overlap.
    """
    for _ in range(max_rounds):
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a["class"] == "figure" or b["class"] == "figure":
                    continue
                ox1, oy1, ox2, oy2 = _intersection(a["bbox"], b["bbox"])
                if ox2 - ox1 <= 0.5 or oy2 - oy1 <= 0.5:
                    continue
                ab, bb = list(a["bbox"]), list(b["bbox"])
                if a["recovered"] != b["recovered"]:
                    # Detections win: only the recovered box gives way, along its shorter overlap side
                    rec, det = (ab, bb) if a["recovered"] else (bb, ab)
                    if (ox2 - ox1) >= (oy2 - oy1):
                        if rec[1] < det[1]:
                            rec[3] = det[1]
                        else:
                            rec[1] = det[3]
                    elif rec[0] < det[0]:
                        rec[2] = det[0]
                    else:
                        rec[0] = det[2]
                elif (ox2 - ox1) >= (oy2 - oy1):
                    upper, lower = (ab, bb) if (ab[1], ab[3]) <= (bb[1], bb[3]) else (bb, ab)
                    cut = _best_cut(oy1, oy2, _ink_profile(ink, (ox1, oy1, ox2, oy2), axis=1))
                    upper[3], lower[1] = cut, cut
                else:
                    left, right = (ab, bb) if (ab[0], ab[2]) <= (bb[0], bb[2]) else (bb, ab)
                    cut = _best_cut(ox1, ox2, _ink_profile(ink, (ox1, oy1, ox2, oy2), axis=0))
                    left[2], right[0] = cut, cut
                a["bbox"], b["bbox"] = tuple(ab), tuple(bb)
                changed = True
        boxes = [b for b in boxes if b["bbox"][2] - b["bbox"][0] >= min_size and b["bbox"][3] - b["bbox"][1] >= min_size]
        if not changed:
            break
    return boxes


def _merge_recovered(found: List[Tuple[int, int, int, int]], blocked: np.ndarray, ink: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Join recovered pieces (1/INK_SCALE boxes x1, y1, x2, y2) into lines and blocks.

    Pieces on the same line join across a word space: any gap up to 0.8x the text height
    (letters of a masthead), or up to 2.5x when the gap is not a column gutter. A gutter is
    a strip with no ink for several line heights above and below; a word space has the
    neighbouring lines' text above and below it. Lines join into blocks when stacked
    closely with overlapping x ranges. A join never covers an existing region (blocked).
    """

    def is_gutter(x1: int, x2: int, y1: int, y2: int) -> bool:
        reach = 5 * (y2 - y1)
        window = ink[max(0, y1 - reach):y2 + reach, x1:x2]
        return window.size == 0 or window.sum() == 0

    def can_join(a, b) -> bool:
        ha, hb = a[3] - a[1], b[3] - b[1]
        small = min(ha, hb)
        v_overlap = min(a[3], b[3]) - max(a[1], b[1])
        h_overlap = min(a[2], b[2]) - max(a[0], b[0])
        gap = max(a[0], b[0]) - min(a[2], b[2])
        similar_size = max(ha, hb) <= 2.5 * small  # stops specks chaining unrelated text together
        same_line = similar_size and v_overlap >= 0.5 * small and (
            gap <= 0.8 * small
            or (gap <= 2.5 * small and not is_gutter(min(a[2], b[2]), max(a[0], b[0]), max(a[1], b[1]), min(a[3], b[3])))
        )
        stacked = h_overlap >= 0.6 * min(a[2] - a[0], b[2] - b[0]) and (max(a[1], b[1]) - min(a[3], b[3])) <= 0.8 * small
        if not (same_line or stacked):
            return False
        x1, y1, x2, y2 = min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
        return not blocked[y1:y2, x1:x2].any()

    boxes = sorted(found, key=lambda b: (b[1], b[0]))
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[j][1] > boxes[i][3] + (boxes[i][3] - boxes[i][1]):
                    break  # sorted by top edge: nothing further down can touch box i
                if can_join(boxes[i], boxes[j]):
                    a, b = boxes[i], boxes[j]
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del boxes[j]
                    merged = True
                    break
            if merged:
                boxes.sort(key=lambda b: (b[1], b[0]))
                break
    return boxes


def recover_missed_text(
    boxes: List[Dict],
    ink: np.ndarray,
    page_size: Tuple[int, int],
    min_ink_pixels: int = 150,
    min_height: int = 4,
    min_piece_ink: int = 12,
) -> List[Dict]:
    """
    Add text regions for ink that no detection covers (headings or notices YOLO missed).

    Uncovered ink is grouped into words with a small dilation, the words are joined into
    lines and blocks (_merge_recovered), and blocks with too little ink are dropped as noise.

    Args:
        boxes: Current regions (full resolution)
        ink: Mask from ink_mask
        page_size: (width, height) of the processed page
        min_ink_pixels: Minimum ink pixels (at 1/INK_SCALE) for a block to count as text
        min_height: Minimum block height at 1/INK_SCALE
        min_piece_ink: Pieces (roughly words) with less ink than this are specks and ignored

    Returns:
        New text regions (not yet merged with boxes)
    """
    covered = np.zeros_like(ink)
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v / INK_SCALE)) for v in b["bbox"])
        covered[max(y1, 0):y2, max(x1, 0):x2] = 1
    uncovered = ink & (1 - covered)
    words = cv2.dilate(uncovered, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))) & (1 - covered)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(words, connectivity=8)
    # Ink per piece; specks (dust, bleed-through) are dropped before joining so they can't bridge gutters
    ink_per_label = np.bincount(labels[uncovered.astype(bool)], minlength=count)
    pieces = []
    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        if covered[y:y + h, x:x + w].any():
            # The piece wraps around an existing region (e.g. an illustration touching a text
            # column): break it back into separate marks and let the joining rebuild words and
            # lines without crossing that region
            part = (uncovered[y:y + h, x:x + w] & (labels[y:y + h, x:x + w] == label)).astype(np.uint8)
            n, _, sub, _ = cv2.connectedComponentsWithStats(part, connectivity=8)
            pieces.extend((int(x + sx), int(y + sy), int(x + sx + sw), int(y + sy + sh)) for sx, sy, sw, sh, area in sub[1:] if area >= 3)
        elif ink_per_label[label] >= min_piece_ink:
            pieces.append((int(x), int(y), int(x + w), int(y + h)))
    width, height = page_size
    found = []
    for x1, y1, x2, y2 in _merge_recovered(pieces, covered, ink):
        if y2 - y1 < min_height or x2 - x1 < 2 * min_height:
            continue
        if int(uncovered[y1:y2, x1:x2].sum()) < min_ink_pixels:
            continue
        bbox = (max(0, (x1 - 1) * INK_SCALE), max(0, (y1 - 1) * INK_SCALE), min(width, (x2 + 1) * INK_SCALE), min(height, (y2 + 1) * INK_SCALE))
        found.append({"class": "text", "confidence": 0.0, "bbox": bbox, "recovered": True})
    return found


def _clusters(intervals: List[Tuple[float, float, int]]) -> List[List[int]]:
    """Group indices whose [start, end) intervals overlap (sweep line)."""
    groups: List[List[int]] = []
    end = -np.inf
    for start, stop, index in sorted(intervals):
        if groups and start < end:
            groups[-1].append(index)
            end = max(end, stop)
        else:
            groups.append([index])
            end = stop
    return groups


def xy_cut_order(boxes: List[Dict], x_tolerance: float, y_tolerance: float) -> List[Tuple[int, int, int]]:
    """
    Reading order by recursive XY-cut, adapted to newspapers.

    At each level the boxes are first split at vertical gutters (x gaps no box crosses).
    If there are none they are split at horizontal gaps, but consecutive bands are merged
    back together while a vertical gutter still runs through all of them: under a heading
    that spans two columns, both columns are then read top to bottom one after the other
    instead of row by row. If neither cut exists the boxes are read top to bottom.
    Tolerances shrink each box slightly so small overhangs don't hide a gutter.

    Returns:
        (box index, band, column) in reading order, where band is the index of the first
        horizontal cut and column the index of the first vertical cut on the box's path
    """
    result: List[Tuple[int, int, int]] = []

    def x_groups(indices: List[int]) -> List[List[int]]:
        spans = []
        for i in indices:
            x1, _, x2, _ = boxes[i]["bbox"]
            start = x1 + x_tolerance
            spans.append((start, max(x2 - x_tolerance, start + 1e-3), i))
        groups = _clusters(spans)
        return sorted(groups, key=lambda g: min(boxes[i]["bbox"][0] for i in g))

    def y_groups(indices: List[int]) -> List[List[int]]:
        spans = []
        for i in indices:
            _, y1, _, y2 = boxes[i]["bbox"]
            start = y1 + y_tolerance
            spans.append((start, max(y2 - y_tolerance, start + 1e-3), i))
        bands = sorted(_clusters(spans), key=lambda g: min(boxes[i]["bbox"][1] for i in g))
        merged: List[List[int]] = []
        for band in bands:
            if merged and len(x_groups(merged[-1])) > 1 and len(x_groups(merged[-1] + band)) > 1:
                merged[-1] = merged[-1] + band
            else:
                merged.append(band)
        return merged

    def recurse(indices: List[int], band: Optional[int], column: Optional[int]) -> None:
        if len(indices) == 1:
            result.append((indices[0], band or 0, column or 0))
            return
        groups = x_groups(indices)
        if len(groups) > 1:
            for n, group in enumerate(groups):
                recurse(group, band, n if column is None else column)
            return
        groups = y_groups(indices)
        if len(groups) > 1:
            for n, group in enumerate(groups):
                recurse(group, n if band is None else band, column)
            return
        for i in sorted(indices, key=lambda i: (boxes[i]["bbox"][1], boxes[i]["bbox"][0])):
            result.append((i, band or 0, column or 0))

    if boxes:
        recurse(list(range(len(boxes))), None, None)
    return result


def build_regions(detections: pd.DataFrame, gray: np.ndarray, recover_missed: bool = True) -> pd.DataFrame:
    """
    Clean raw detections into OCR regions.

    Args:
        detections: Raw YOLO output with columns class, confidence, x1, y1, x2, y2 (processed page pixels)
        gray: The processed greyscale page the detections were made on
        recover_missed: Add regions for uncovered ink

    Returns:
        DataFrame with class, confidence, x1, y1, x2, y2 (ints), reading_order (1-based),
        page_block, column_number, box_page_id and recovered (bool), sorted by reading order
    """
    height, width = gray.shape
    boxes = []
    for row in detections.to_dict("records"):
        cls = CLASS_MAP.get(row["class"])
        if cls is None:
            continue
        x1, y1 = max(0.0, float(row["x1"])), max(0.0, float(row["y1"]))
        x2, y2 = min(float(width), float(row["x2"])), min(float(height), float(row["y2"]))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            boxes.append({"class": cls, "confidence": float(row["confidence"]), "bbox": (x1, y1, x2, y2), "recovered": False})

    ink = ink_mask(gray)
    boxes = remove_nested(boxes)
    boxes = split_overlaps(boxes, ink)
    if recover_missed:
        boxes += recover_missed_text(boxes, ink, (width, height))
        boxes = split_overlaps(boxes, ink)

    rows = []
    if not boxes:
        return pd.DataFrame(columns=["class", "confidence", "x1", "y1", "x2", "y2", "reading_order", "page_block", "column_number", "box_page_id", "recovered"])
    text_widths = [b["bbox"][2] - b["bbox"][0] for b in boxes if b["class"] == "text"] or [width / 6]
    order = xy_cut_order(boxes, x_tolerance=0.02 * float(np.median(text_widths)), y_tolerance=2.0)
    for rank, (index, band, column) in enumerate(order, start=1):
        b = boxes[index]
        x1, y1, x2, y2 = b["bbox"]
        rows.append(
            {
                "class": b["class"],
                "confidence": b["confidence"],
                "x1": int(np.floor(x1)),
                "y1": int(np.floor(y1)),
                "x2": int(np.ceil(x2)),
                "y2": int(np.ceil(y2)),
                "reading_order": rank,
                "page_block": band,
                "column_number": column,
                "box_page_id": f"B{band}C{column}R{rank}",
                "recovered": b["recovered"],
            }
        )
    return pd.DataFrame(rows)
