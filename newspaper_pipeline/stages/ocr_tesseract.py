"""
Stage 4b: word-level coordinates from Tesseract (via tesserocr, in-process).

Each worker process keeps one TessBaseAPI instance alive. Tesseract binarises the crop
itself with its Sauvola implementation (thresholding_method=2), which measured slightly
better than Otsu on the sample page, so the crop it receives is the same pre-processed
greyscale crop chandra gets.
"""

import os
from typing import Dict, List, Optional

_API = None  # one TessBaseAPI per worker process
_API_KEY: Optional[tuple] = None


def init_worker() -> None:
    """Process pool initializer: one thread per Tesseract process, since the pool already provides the parallelism."""
    os.environ["OMP_THREAD_LIMIT"] = "1"


def _get_api(tessdata: str, lang: str):
    global _API, _API_KEY
    import tesserocr

    if _API is None or _API_KEY != (tessdata, lang):
        if _API is not None:
            _API.End()
        _API = tesserocr.PyTessBaseAPI(path=tessdata, lang=lang, psm=tesserocr.PSM.AUTO)
        _API.SetVariable("thresholding_method", "2")  # Sauvola
        _API_KEY = (tessdata, lang)
    return _API


def tesseract_words(crop_path: str, tessdata: str, lang: str = "eng") -> List[Dict]:
    """
    Recognise a region crop and return its words with geometry.

    Args:
        crop_path: Path to the region crop
        tessdata: Folder containing <lang>.traineddata
        lang: Tesseract language code

    Returns:
        Words in Tesseract reading order. Each word has text, conf, line (index unique
        within the region), block, and x1/y1/x2/y2 in crop pixels.
    """
    import tesserocr
    from PIL import Image

    api = _get_api(tessdata, lang)
    with Image.open(crop_path) as image:
        api.SetImage(image)
        api.Recognize()

    words: List[Dict] = []
    iterator = api.GetIterator()
    if iterator is None:
        return words

    level = tesserocr.RIL.WORD
    block, line = -1, -1
    for word in tesserocr.iterate_level(iterator, level):
        if word.IsAtBeginningOf(tesserocr.RIL.BLOCK):
            block += 1
        if word.IsAtBeginningOf(tesserocr.RIL.TEXTLINE):
            line += 1
        try:
            text = word.GetUTF8Text(level)
        except RuntimeError:  # tesserocr raises instead of returning "" for empty words
            continue
        if not text or not text.strip():
            continue
        box = word.BoundingBox(level)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        words.append(
            {
                "text": text.strip(),
                "conf": float(word.Confidence(level)),
                "block": max(block, 0),
                "line": max(line, 0),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }
        )
    return words
