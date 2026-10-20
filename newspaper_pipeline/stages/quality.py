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

# Phrases a model uses about the image, never about the world: generic ones like "I cannot" or
# "appears to be" are everyday Victorian prose ("I cannot conceive...") and withheld real text.
# Commentary comes first in an answer, so only the start of the output is checked.
META_COMMENTARY = re.compile(
    r"\b(?:the|this) (?:image|picture|photograph|scan)\b|\bheavily cropped\b"
    r"|\bno (?:legible|readable|visible) text\b"
    r"|\bI (?:cannot|can't|am unable to) (?:read|transcribe|make out|decipher|see|identify)\b"
    r"|\b(?:text|image|writing) is (?:not |too )?(?:visible|legible|readable|blurry|faded|unclear)\b",
    re.IGNORECASE,
)
COMMENTARY_WINDOW = 300  # characters


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


def fragment_share(text: str) -> float:
    """
    Share of the tokens with letters that have at most 3 of them. Prose is about 40-50%; the
    cut-off ends of lines from a neighbouring column or the facing page ("to e, in le een, ed all
    he") are 85-100%. Numbers don't count either way, so tables and advertisements stay low.
    """
    tokens = [t for t in text.split() if any(c.isalpha() for c in t)]
    if len(tokens) < 3:
        return 0.0
    return sum(1 for t in tokens if sum(c.isalpha() for c in t) <= 3) / len(tokens)


def transcription_problem(region_class: str, text: str, raw: str, finish_reasons: Sequence[str] = ()) -> Optional[str]:
    """The checks that need nothing but chandra's output: worth a retry when they fail."""
    if is_layout_json(raw):
        return "layout JSON instead of text"
    if any(reason == "length" for reason in finish_reasons):
        return "output hit its token limit (repetition loop)"
    if region_class != "figure" and text.strip() and META_COMMENTARY.search(text[:COMMENTARY_WINDOW]):
        return "model commentary instead of a transcription"
    return None


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
    at_page_edge: bool = False,
    from_figure: bool = False,
    figure_min_alignment: float = 0.2,
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
        at_page_edge: The region touches the left or right edge of the scan
        from_figure: A figure region read as text. Asked to transcribe a picture with no text,
            chandra invents some ("THE NEW YORK TIMES, FRIDAY, MAY 15, 1931"), so figure text is
            only published when Tesseract independently found the same words
        figure_min_alignment: How much of a figure's text Tesseract must confirm

    Returns:
        Reason string when suspect, otherwise None
    """
    problem = transcription_problem(region_class, text, raw, finish_reasons)
    if problem or region_class == "figure" or not text.strip():
        return problem
    # A strip of cut-off text is a few letters wide. Calibrated on the edge regions of 250 BLN600
    # excerpts and 9 full pages: real text at an edge is at least 2.9 word sizes wide, and under 6
    # word sizes with 60% fragments caught 35 of 53 strips and none of 893 real regions
    if from_figure and alignment_score < figure_min_alignment:
        return "figure text not confirmed by Tesseract"
    word_size = max(word_area, 1.0) ** 0.5
    if at_page_edge and width < 2.5 * word_size:
        return "cut-off text at the page edge"  # too narrow for a column, whatever was read in it
    if at_page_edge and width < 6.0 * word_size and fragment_share(text) >= 0.6:
        return "cut-off text at the page edge"
    capacity = (width * height) / max(word_area, 1.0)
    if n_chandra_tokens > max(20.0, 3.0 * capacity):
        return f"{n_chandra_tokens} tokens but the region holds about {capacity:.0f} words"
    if n_tesseract_words and alignment_score < 0.2 and n_chandra_tokens > 3 * n_tesseract_words + 10:
        return f"{n_chandra_tokens} tokens against {n_tesseract_words} Tesseract words with alignment {alignment_score:.2f}"
    return None
