"""
Stage 6 (side outputs): debug overlay and flat exports alongside the IIIF documents.

    overlay.jpg       downscaled original scan with region (by class) and word boxes; because
                      the boxes are already mapped back to original-scan pixels, this is also
                      a visual check of the deskew/crop inverse mapping
    page.txt          region texts in reading order
    regions.parquet   one row per region (chandra text, html, labels, alignment stats, suspect reason)
    lines.parquet     one row per aligned line
    words.parquet     one row per aligned word
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

CLASS_COLOURS = {"text": (0, 90, 255), "title": (220, 0, 0), "figure": (0, 160, 0), "table": (255, 140, 0)}
WORD_COLOUR = (255, 0, 200)
SUSPECT_COLOUR = (150, 150, 150)


def _open_downscaled(source_path: str, max_width: int) -> Image.Image:
    image = Image.open(source_path)
    if image.format == "JPEG2000":
        # Decode a lower resolution level directly instead of the full 50 megapixels
        reduce = 0
        while image.size[0] / (2 ** (reduce + 1)) >= max_width and reduce < 5:
            reduce += 1
        image.reduce = reduce
    image.draft("RGB", (max_width, max_width * 4))  # no-op except for JPEG
    image = image.convert("RGB")
    image.thumbnail((max_width, max_width * 4))
    return image


def write_overlay(source_path: str, original_size: tuple, regions: List[Dict], out_path: Path, max_width: int = 2400) -> None:
    """Draw region and word boxes (original-scan pixels) on a downscaled copy of the original scan."""
    image = _open_downscaled(source_path, max_width)
    scale = image.size[0] / float(original_size[0])
    draw = ImageDraw.Draw(image)

    def rect(box: Dict) -> tuple:
        return (box["x"] * scale, box["y"] * scale, (box["x"] + box["w"]) * scale, (box["y"] + box["h"]) * scale)

    for region in regions:
        for word in region.get("words", []):
            draw.rectangle(rect(word["box"]), outline=WORD_COLOUR, width=1)
    for region in regions:
        suspect = bool(region.get("suspect"))
        colour = SUSPECT_COLOUR if suspect else CLASS_COLOURS.get(region["class"], (90, 90, 90))
        draw.rectangle(rect(region["box"]), outline=colour, width=3)
        x, y, _, _ = rect(region["box"])
        draw.text((x + 4, y + 2), f"{region['reading_order']} {region['class']}{' SUSPECT' if suspect else ''}", fill=colour)
    image.save(out_path, quality=85)


def write_flat_exports(page_dir: Path, page: Dict, regions: List[Dict]) -> None:
    """Write page.txt and the regions/lines/words parquet files for one page."""
    page_id = page["page_id"]
    ordered = sorted(regions, key=lambda r: r["reading_order"])

    with open(page_dir / "page.txt", "w", encoding="utf-8") as handle:
        for region in ordered:
            if region.get("text") and not region.get("suspect"):
                handle.write(f"[{region['reading_order']} {region['box_page_id']} {region['class']}]\n{region['text']}\n\n")

    region_rows, line_rows, word_rows = [], [], []
    for region in ordered:
        box = region["box"]
        region_rows.append(
            {
                "page_id": page_id,
                "box_page_id": region["box_page_id"],
                "reading_order": region["reading_order"],
                "class": region["class"],
                "recovered": region.get("recovered", False),  # added from uncovered ink, not detected by YOLO
                "x": box["x"],
                "y": box["y"],
                "w": box["w"],
                "h": box["h"],
                "text": region.get("text", ""),
                "html": region.get("html", ""),
                "ocr_class": region.get("ocr_class", region["class"]),  # how chandra read it: figures as "text" by default
                "chandra_labels": ",".join(region.get("labels", [])),
                "image_descriptions": "\n".join(region.get("image_descriptions", [])),
                "finish_reasons": ",".join(str(f) for f in region.get("finish_reasons", [])),
                "prompt_tokens": region.get("prompt_tokens", 0),
                "completion_tokens": region.get("completion_tokens", 0),
                "segments": region.get("segments", 0),
                "alignment_score": region.get("alignment_score", 0.0),
                "n_chandra_tokens": region.get("n_chandra_tokens", 0),
                "n_tesseract_words": region.get("n_tesseract_words", 0),
                "n_matched": region.get("n_matched", 0),
                "n_estimated": region.get("n_estimated", 0),
                "error": region.get("error", ""),
                "suspect": region.get("suspect", ""),
            }
        )
        for n, line in enumerate(region.get("lines", [])):
            line_rows.append({"page_id": page_id, "box_page_id": region["box_page_id"], "line": n, "text": line["text"], **line["box"]})
        for n, word in enumerate(region.get("words", [])):
            word_rows.append(
                {
                    "page_id": page_id,
                    "box_page_id": region["box_page_id"],
                    "word": n,
                    "line": word["line"],
                    "text": word["text"],
                    "tesseract_text": word["tesseract_text"],
                    "tesseract_conf": word["conf"],
                    "similarity": word["similarity"],
                    "match": word["match"],
                    **word["box"],
                    # continuation on the next line of a word hyphenated across lines
                    **{f"cont_{k}": (word["part_boxes"][1][k] if word.get("part_boxes") else None) for k in ("x", "y", "w", "h")},
                }
            )

    pd.DataFrame(region_rows).to_parquet(page_dir / "regions.parquet", index=False)
    pd.DataFrame(line_rows, columns=["page_id", "box_page_id", "line", "text", "x", "y", "w", "h"]).to_parquet(page_dir / "lines.parquet", index=False)
    pd.DataFrame(
        word_rows,
        columns=[
            "page_id", "box_page_id", "word", "line", "text", "tesseract_text", "tesseract_conf", "similarity", "match",
            "x", "y", "w", "h", "cont_x", "cont_y", "cont_w", "cont_h",
        ],
    ).to_parquet(page_dir / "words.parquet", index=False)
