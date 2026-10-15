"""
Flag chandra output that is probably not a transcription.

On the 1874 sample page chandra occasionally:
- invented hundreds of words for a 20px tall rule line (padded to a readable aspect ratio),
- described the image ("No newspaper title text is visible as the image is heavily cropped...")
  instead of transcribing it,
- answered in its layout JSON format ([{"label": ..., "bbox": ...}]) with no text at all,
- repeated itself (coordinates, counting, rows of dots) until the token limit.

Flagged regions keep their raw output in regions.parquet (with the reason) but are left
out of the IIIF annotations and page.txt, so invented text is never published.
"""

import json
import re
from typing import Optional, Sequence

META_COMMENTARY = re.compile(
    r"\b(?:the|this) image\b|\bis (?:not )?visible\b|\bappears to (?:be|show)\b|\bheavily cropped\b"
    r"|\bno (?:legible|readable|visible) text\b|\bI (?:cannot|can't|am unable)\b",
    re.IGNORECASE,
)


def is_layout_json(content: str) -> bool:
    """True when chandra answered with its layout format (a JSON list of {label, bbox}) instead of text."""
    stripped = (content or "").strip()
    if not stripped.startswith("["):
        return False
    try:
        data = json.loads(stripped)
    except ValueError:
        return False
    return isinstance(data, list) and bool(data) and all(isinstance(item, dict) and "bbox" in item for item in data)


def assess_region(
    region_class: str,
    text: str,
    raw: str,
    width: float,
    height: float,
    word_area: float,
    n_chandra_tokens: int,
    n_tesseract_words: int,
    alignment_score: float,
    finish_reasons: Sequence[str] = (),
) -> Optional[str]:
    """
    Decide whether a region's chandra output looks like something other than a transcription.

    Args:
        region_class: Layout class; figures are expected to be descriptions and only get the JSON check
        text: Plain text derived from the chandra output
        raw: Raw chandra output
        width, height: Region size in processed-page pixels
        word_area: Typical Tesseract word box area on this page (pixels), used to estimate capacity
        n_chandra_tokens, n_tesseract_words, alignment_score: From the alignment stage
        finish_reasons: Per-request finish reasons; "length" means the output hit its token cap

    Returns:
        Reason string when suspect, otherwise None
    """
    if is_layout_json(raw):
        return "layout JSON instead of text"
    if any(reason == "length" for reason in finish_reasons):
        return "output hit its token limit (repetition loop)"
    if region_class == "figure" or not text.strip():
        return None
    if META_COMMENTARY.search(text):
        return "model commentary instead of a transcription"
    capacity = (width * height) / max(word_area, 1.0)
    if n_chandra_tokens > max(20.0, 3.0 * capacity):
        return f"{n_chandra_tokens} tokens but the region holds about {capacity:.0f} words"
    if n_tesseract_words and alignment_score < 0.2 and n_chandra_tokens > 3 * n_tesseract_words + 10:
        return f"{n_chandra_tokens} tokens against {n_tesseract_words} Tesseract words with alignment {alignment_score:.2f}"
    return None
