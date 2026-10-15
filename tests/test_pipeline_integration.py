"""
End-to-end run on part of the 1874 sample page with real DocLayout-YOLO and Tesseract.

Chandra is replaced by a fake that transcribes each crop with Tesseract, so the run is
offline and deterministic. The scan is rotated and placed on a dark scanner bed first,
which exercises border masking, deskew and the inverse coordinate mapping.
"""

import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import tesserocr  # imported on the main thread: its cysignals dependency installs signal handlers on import
from PIL import Image

from newspaper_pipeline.config import PipelineConfig
from newspaper_pipeline.orchestrator import SUCCESS_MARKER, Pipeline
from newspaper_pipeline.stages.iiif import validate

SAMPLE = Path(__file__).resolve().parent.parent / "input_images" / "testing_page_1874.png"
ROTATION = 1.2
BORDER = 160

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not SAMPLE.exists(), reason="sample page not available")]


class TesseractChandra:
    """Stands in for the chandra endpoint: answers with chandra-style HTML produced by Tesseract."""

    def __init__(self, tessdata):
        self.tessdata = tessdata
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    async def create(self, model, messages, max_tokens=None, temperature=None):
        self.calls += 1
        url = messages[0]["content"][1]["image_url"]["url"]
        png = base64.b64decode(url.split(",", 1)[1])
        text = await asyncio.to_thread(self._ocr, png)
        html = '<div data-bbox="0 0 1000 1000" data-label="Text"><p>' + "<br>".join(text.splitlines()) + "</p></div>"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=html), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=len(text.split())),
        )

    def _ocr(self, png):
        with tesserocr.PyTessBaseAPI(path=self.tessdata, lang="eng") as api:
            api.SetVariable("thresholding_method", "2")  # same Sauvola setting as the pipeline's Tesseract stage
            api.SetImage(Image.open(io.BytesIO(png)))
            return api.GetUTF8Text()


def make_scan(path: Path) -> np.ndarray:
    """Top-left part of the sample page, rotated by ROTATION degrees, on a dark scanner bed, saved as TIFF."""
    with Image.open(SAMPLE) as image:
        page = np.asarray(image.convert("L").crop((0, 2600, 3200, 5600)))
    h, w = page.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ROTATION, 1.0)
    rotated = cv2.warpAffine(page, m, (w, h), borderValue=int(np.median(page)))
    scan = np.full((h + 2 * BORDER, w + 2 * BORDER), 30, np.uint8)
    scan[BORDER:BORDER + h, BORDER:BORDER + w] = rotated
    Image.fromarray(scan).save(path, compression="tiff_lzw")
    return scan


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    (root / "scans").mkdir()
    scan = make_scan(root / "scans" / "colonist_p1.tif")
    config = PipelineConfig(
        input_dir=root / "scans",
        output_dir=root / "out",
        work_dir=root / "work",
        cache_dir=root / "cache",
        tesseract_workers=4,
        yolo_device="cpu",
    )
    fake = TesseractChandra(config.tessdata)
    summary = asyncio.run(Pipeline(config, chandra_client=fake).run())
    return SimpleNamespace(root=root, config=config, summary=summary, fake=fake, scan=scan, out=root / "out" / "colonist_p1")


def test_page_completes_and_temp_files_are_removed(run):
    assert run.summary.pages_succeeded == ["colonist_p1"], run.summary
    assert (run.out / SUCCESS_MARKER).exists()
    assert not (run.root / "work").exists()
    for name in ("manifest.json", "annotations/blocks.json", "annotations/lines.json", "annotations/words.json",
                 "overlay.jpg", "page.txt", "regions.parquet", "lines.parquet", "words.parquet", "report.json"):
        assert (run.out / name).exists(), name
    assert (run.root / "out" / "collection.json").exists()


def test_preprocessing_found_border_and_skew(run):
    report = json.loads((run.out / "report.json").read_text())
    # the crop has its own small skew (paper warps), so measure it rather than assuming zero
    from newspaper_pipeline.stages.preprocess import estimate_skew

    with Image.open(SAMPLE) as image:
        intrinsic = estimate_skew(np.asarray(image.convert("L").crop((0, 2600, 3200, 5600))))
    assert report["deskew_angle"] == pytest.approx(intrinsic - ROTATION, abs=0.12)
    x1, y1, x2, y2 = report["content_box"]
    assert BORDER <= x1 <= BORDER + 25 and BORDER <= y1 <= BORDER + 25
    assert report["regions"] > 10 and report["words"] > 300


def test_outputs_validate_and_use_original_canvas(run):
    manifest = json.loads((run.out / "manifest.json").read_text())
    assert validate(manifest) == []
    canvas = manifest["items"][0]
    assert (canvas["width"], canvas["height"]) == (run.scan.shape[1], run.scan.shape[0])
    for name in ("blocks", "lines", "words"):
        page = json.loads((run.out / "annotations" / f"{name}.json").read_text())
        assert validate(page) == []
        assert page["items"]


def test_alignment_quality(run):
    import pandas as pd

    regions = pd.read_parquet(run.out / "regions.parquet")
    text_regions = regions[(regions["class"] != "figure") & (regions["n_chandra_tokens"] > 0) & (regions["n_tesseract_words"] > 0)]
    # The fake chandra *is* Tesseract with the same settings, so nearly every token should land on a box
    assert text_regions["alignment_score"].median() > 0.95
    report = json.loads((run.out / "report.json").read_text())
    assert report["alignment_score"] > 0.9 and 0 < report["word_box_coverage"] <= report["alignment_score"]


def test_word_boxes_land_on_the_same_words_in_the_original_scan(run):
    """Strongest check of the inverse mapping: re-OCR word boxes cut from the original rotated scan."""
    import pandas as pd
    from rapidfuzz import fuzz

    words = pd.read_parquet(run.out / "words.parquet")
    candidates = words[(words["tesseract_conf"] > 90) & (words["text"].str.len() >= 5) & words["text"].str.isalpha()]
    sample = candidates.sample(min(40, len(candidates)), random_state=0)
    hits = 0
    with tesserocr.PyTessBaseAPI(path=run.config.tessdata, lang="eng", psm=tesserocr.PSM.SINGLE_WORD) as api:
        for row in sample.itertuples():
            pad = 6
            crop = run.scan[max(row.y - pad, 0):row.y + row.h + pad, max(row.x - pad, 0):row.x + row.w + pad]
            api.SetImage(Image.fromarray(crop))
            if fuzz.ratio(api.GetUTF8Text().strip().lower(), row.text.lower()) >= 80:
                hits += 1
    assert len(sample) >= 20
    assert hits / len(sample) >= 0.8, f"only {hits}/{len(sample)} word boxes matched"


def test_rerun_skips_finished_pages(run):
    calls_before = run.fake.calls
    summary = asyncio.run(Pipeline(run.config, chandra_client=run.fake).run())
    assert summary.pages_skipped == 1 and summary.pages_succeeded == []
    assert run.fake.calls == calls_before


def test_forced_rerun_uses_chandra_cache(run):
    calls_before = run.fake.calls
    summary = asyncio.run(Pipeline(run.config, chandra_client=run.fake).run(force=True))
    assert summary.pages_succeeded == ["colonist_p1"]
    assert run.fake.calls == calls_before  # every region came from the cache
    assert summary.chandra["cache_hits"] > 10


def test_regions_do_not_duplicate_text(run):
    """
    No word is transcribed by two regions. (Region outlines on the original canvas can touch
    slightly on a rotated scan, since deskewed rectangles map back to slightly larger boxes,
    but the crops OCR sees never overlap.)
    """
    import pandas as pd

    words = pd.read_parquet(run.out / "words.parquet")
    boxes = words[["x", "y", "w", "h"]].to_numpy(float)
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    regions = words["box_page_id"].to_numpy()
    duplicated = 0
    for i in range(len(words)):
        ix = np.clip(np.minimum(x2[i], x2) - np.maximum(boxes[i, 0], boxes[:, 0]), 0, None)
        iy = np.clip(np.minimum(y2[i], y2) - np.maximum(boxes[i, 1], boxes[:, 1]), 0, None)
        overlap = ix * iy / (boxes[i, 2] * boxes[i, 3])
        if ((overlap > 0.5) & (regions != regions[i])).any():
            duplicated += 1
    assert duplicated / len(words) < 0.01, f"{duplicated} of {len(words)} words also appear in another region"
    assert pd.read_parquet(run.out / "regions.parquet")["reading_order"].is_unique
