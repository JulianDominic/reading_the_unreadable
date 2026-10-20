"""
Configuration for the concurrent newspaper pipeline.

Worker counts default to values that fit a 24GB Apple Silicon machine: a decoded
newspaper scan is roughly 50 megapixels, so stages that hold whole pages are kept small,
while Tesseract (which only sees region crops) gets most of the CPU cores.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

SUPPORTED_SUFFIXES = (".jp2", ".j2k", ".jpx", ".tif", ".tiff", ".png", ".jpg", ".jpeg")

DEFAULT_PROMPTS: Dict[str, str] = {
    "text": "Transcribe the text from this newspaper image. Include linebreaks. Use plain text only, no markdown. Do not add commentary.",
    "title": "Transcribe the title/headline from this newspaper image. Use plain text only, no markdown. Do not add commentary.",
    "figure": "Describe the graphic from this newspaper image. Do not add commentary.",
    "table": "Extract the table from this image as tab-separated values (TSV). Do not add commentary.",
}


def _default_tessdata() -> str:
    """Tesseract language data location; the tesserocr wheel bundles Tesseract but no tessdata."""
    if os.getenv("TESSDATA_PREFIX"):
        return os.environ["TESSDATA_PREFIX"]
    for candidate in ("/opt/homebrew/share/tessdata", "/usr/local/share/tessdata", "/usr/share/tesseract-ocr/5/tessdata"):
        if Path(candidate).is_dir():
            return candidate
    return "/opt/homebrew/share/tessdata"


@dataclass(frozen=True)
class PipelineConfig:
    """
    All settings for a pipeline run.

    Attributes:
        input_dir: Folder of source scans (JP2, TIFF, PNG, JPG); searched recursively
        output_dir: Folder for manifests, annotations, overlays and flat exports
        work_dir: Folder for temporary per-page files, removed during cleanup
        cache_dir: Folder for cached chandra responses (survives runs so re-runs are cheap)
        image_base_url: Base of the IIIF Image API 3 service, e.g. Cantaloupe's /iiif/3
        manifest_base_url: URL where output_dir is served, used for manifest/annotation ids
        slash_substitute: Cantaloupe slash_substitute for identifiers in subfolders; None URL-encodes "/"
    """

    input_dir: Path = Path("input_images")
    output_dir: Path = Path("output/iiif")
    work_dir: Path = Path("output/work")
    cache_dir: Path = Path("output/cache/chandra")

    # IIIF
    image_base_url: str = "http://localhost:8182/iiif/3"
    manifest_base_url: str = "http://localhost:8000"
    slash_substitute: Optional[str] = None
    language: str = "en"

    # Chandra (remote, OpenAI compatible)
    chandra_api_base: Optional[str] = None
    chandra_api_key: Optional[str] = None
    chandra_model: str = "chandra-ocr-2-vllm"
    chandra_concurrency: int = 2
    chandra_timeout: float = 600.0
    chandra_max_tokens: int = 8192
    # Greedy decoding: at the server's default temperature the same crop gave 70, 71, 119 and
    # 70 words over four requests (the 119 run invented text); at 0 it gave 70 every time
    chandra_temperature: float = 0.0
    # Per-request token caps (see ChandraPool.token_cap): repetition loops stop early
    chandra_figure_max_tokens: int = 512
    chandra_tokens_per_word: float = 6.0
    chandra_min_word_area: float = 2000.0  # pixels; smaller than a typical word box, so the cap is generous
    chandra_retries: int = 4
    transcribe_figures: bool = True  # read figure regions (mostly adverts) as text instead of describing them
    chandra_quality_retries: int = 2  # re-ask when the output isn't a transcription (loop, commentary, layout JSON)
    chandra_retry_temperature: float = 0.3
    chandra_retry_min_wait: float = 4.0
    chandra_retry_max_wait: float = 60.0
    prompts: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PROMPTS))
    # Regions taller than max_ratio * width are sent to chandra in overlapping segments
    split_max_ratio: float = 1.5
    split_overlap_fraction: float = 0.2
    # Wider crops are padded with white: the Qwen3-VL processor behind chandra rejects extreme
    # aspect ratios (a 5814x20 rule line returned HTTP 400)
    max_aspect_ratio: float = 20.0

    # Layout
    yolo_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    yolo_file: str = "doclayout_yolo_docstructbench_imgsz1024.pt"
    yolo_image_size: int = 1056
    yolo_conf: float = 0.2
    yolo_device: str = "mps"
    # "xycut": stages/regions.py (non-overlapping regions, recovered missed text, XY-cut order)
    # "legacy": function_modules.bbox_functions.postprocess_bbox from the original paper
    region_builder: str = "xycut"
    crop_padding: int = 24  # white margin around each region crop, in pixels

    # Tesseract
    tessdata: str = field(default_factory=_default_tessdata)
    tesseract_lang: str = "eng"

    # Pre-processing
    mask_borders: bool = True
    deskew: bool = True
    max_deskew_angle: float = 5.0
    enhance: bool = True
    clahe_clip_limit: float = 0.0  # CLAHE amplifies bleed-through; measured worse on the sample page
    denoise: bool = True

    # Concurrency
    convert_workers: int = 2
    preprocess_workers: int = 3
    tesseract_workers: int = 6
    max_pages_in_flight: int = 3

    # Behaviour
    keep_temp: bool = False
    overlay_max_width: int = 2400
