"""
Run the concurrent newspaper OCR pipeline: scans in, IIIF Presentation 3 manifests out.

Each page is converted, pre-processed (border mask, deskew, enhancement), laid out with
DocLayout-YOLO, transcribed region by region with chandra-ocr-2 (text) and Tesseract
(word boxes), aligned, and written as a IIIF manifest with block/line/word annotations.
Pages flow through the stages independently; see newspaper_pipeline/orchestrator.py.

Usage:
    python run_pipeline.py                              # input_images/ -> output/iiif/
    python run_pipeline.py --input scans --output out --image-base-url http://host:8182/iiif/3

Requires OPENAI_API_BASE and OPENAI_API_KEY (in the environment or .env) for chandra.
Re-running skips finished pages; chandra responses are cached, so retrying is cheap.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from newspaper_pipeline.config import PipelineConfig
from newspaper_pipeline.orchestrator import Pipeline


def parse_args() -> argparse.Namespace:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=defaults.input_dir, help="folder of scans (searched recursively)")
    parser.add_argument("--output", type=Path, default=defaults.output_dir, help="folder for manifests and exports")
    parser.add_argument("--work", type=Path, default=defaults.work_dir, help="folder for temporary files")
    parser.add_argument("--cache", type=Path, default=defaults.cache_dir, help="folder for cached chandra responses")
    parser.add_argument("--image-base-url", default=defaults.image_base_url, help="IIIF Image API base (Cantaloupe /iiif/3)")
    parser.add_argument("--manifest-base-url", default=defaults.manifest_base_url, help="URL where --output is served")
    parser.add_argument("--slash-substitute", default=None, help="Cantaloupe slash_substitute for scans in subfolders")
    parser.add_argument("--model", default=defaults.chandra_model, help="chandra model name on the endpoint")
    parser.add_argument("--chandra-concurrency", type=int, default=defaults.chandra_concurrency)
    parser.add_argument("--tesseract-workers", type=int, default=defaults.tesseract_workers)
    parser.add_argument("--pages-in-flight", type=int, default=defaults.max_pages_in_flight)
    parser.add_argument("--device", default=defaults.yolo_device, help="DocLayout-YOLO device: mps or cpu")
    parser.add_argument("--region-builder", choices=["xycut", "legacy"], default=defaults.region_builder,
                        help="xycut: non-overlapping regions in column reading order; legacy: the original paper's post-processing")
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--no-border-mask", action="store_true")
    parser.add_argument("--no-enhance", action="store_true")
    parser.add_argument("--describe-figures", action="store_true",
                        help="ask chandra to describe figure regions instead of transcribing their text")
    parser.add_argument("--keep-temp", action="store_true", help="keep per-page temporary files for debugging")
    parser.add_argument("--force", action="store_true", help="re-process pages that already finished")
    parser.add_argument("--limit", type=int, default=None, help="process at most N pages")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger = logging.getLogger("run_pipeline")

    load_dotenv()
    api_base, api_key = os.getenv("OPENAI_API_BASE"), os.getenv("OPENAI_API_KEY")
    if not api_base:
        logger.error("Missing OPENAI_API_BASE (set it in the environment or .env)")
        return 2
    if not args.input.exists():
        logger.error(f"Input folder '{args.input}' not found")
        return 2

    config = PipelineConfig(
        input_dir=args.input,
        output_dir=args.output,
        work_dir=args.work,
        cache_dir=args.cache,
        image_base_url=args.image_base_url,
        manifest_base_url=args.manifest_base_url,
        slash_substitute=args.slash_substitute,
        chandra_api_base=api_base,
        chandra_api_key=api_key,
        chandra_model=args.model,
        chandra_concurrency=args.chandra_concurrency,
        tesseract_workers=args.tesseract_workers,
        max_pages_in_flight=args.pages_in_flight,
        yolo_device=args.device,
        region_builder=args.region_builder,
        deskew=not args.no_deskew,
        mask_borders=not args.no_border_mask,
        enhance=not args.no_enhance,
        transcribe_figures=not args.describe_figures,
        keep_temp=args.keep_temp,
    )
    logger.info(f"chandra: {config.chandra_model} at {api_base} (max {config.chandra_concurrency} concurrent)")

    summary = asyncio.run(Pipeline(config).run(force=args.force, limit=args.limit))

    logger.info("=" * 60)
    logger.info(
        f"Finished in {summary.seconds:.0f}s: {len(summary.pages_succeeded)} complete, {len(summary.pages_incomplete)} incomplete, "
        f"{len(summary.pages_failed)} failed, {summary.pages_skipped} skipped (already done)"
    )
    if summary.chandra:
        logger.info(f"chandra: {summary.chandra}")
    for page_id, error in summary.pages_failed.items():
        logger.error(f"  {page_id}: {error}")
    logger.info(f"Manifests: {config.output_dir}/<page_id>/manifest.json, collection: {config.output_dir}/collection.json")
    return 1 if summary.pages_failed or summary.pages_incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
