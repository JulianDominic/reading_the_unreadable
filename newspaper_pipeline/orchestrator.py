"""
Asynchronous orchestration of the pipeline stages.

Every stage has its own fixed group of workers:

    convert       process pool (config.convert_workers)
    preprocess    process pool (config.preprocess_workers)
    layout        one thread holding the DocLayout-YOLO model
    tesseract     process pool (config.tesseract_workers)
    chandra       config.chandra_concurrency async workers on a priority queue
    finalize      threads (alignment, IIIF, exports, cleanup)

Each page is a coroutine that walks through the stages on its own, so a page moves to the
next stage as soon as it is ready, never waiting for other pages. Within a page every
region runs chandra and Tesseract concurrently; the page is finalised when its last
region completes. A semaphore caps pages in flight to bound memory and temp disk use;
it is sized so chandra (the bottleneck) always has queued work.
"""

import asyncio
import json
import logging
import multiprocessing
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from newspaper_pipeline.config import SUPPORTED_SUFFIXES, PipelineConfig
from newspaper_pipeline.stages import cleanup, iiif
from newspaper_pipeline.stages.align import align_region
from newspaper_pipeline.stages.convert import convert_page
from newspaper_pipeline.stages.export import write_flat_exports, write_overlay
from newspaper_pipeline.stages.ocr_chandra import ChandraPool
from newspaper_pipeline.stages.ocr_tesseract import PSM_AUTO, PSM_SINGLE_BLOCK, PSM_SPARSE_TEXT, init_worker, tesseract_words
from newspaper_pipeline.stages.preprocess import preprocess_page
from newspaper_pipeline.stages.quality import assess_region, transcription_problem
from newspaper_pipeline.stages.regions import figures_with_own_text
from newspaper_pipeline.transforms import PageTransform

logger = logging.getLogger(__name__)

SUCCESS_MARKER = "_SUCCESS"


@dataclass
class SourcePage:
    page_id: str
    source_path: Path
    source_relpath: str
    seq: int


@dataclass
class RunSummary:
    pages_total: int = 0
    pages_skipped: int = 0
    pages_succeeded: List[str] = field(default_factory=list)
    pages_incomplete: List[str] = field(default_factory=list)
    pages_failed: Dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0
    chandra: Dict[str, int] = field(default_factory=dict)


# Figure text needs this much Tesseract support (see quality.assess_region). Measured on the 25
# figures the first version withheld: real advert text reached 0.25-1.00 with the readings below,
# text chandra invented for pictures 0.00-0.03.
FIGURE_MIN_ALIGNMENT = 0.2
# Tesseract readings tried for figure text, cheapest first: display type is often too large, and
# scattered words read better in sparse-text mode. Stops once the support is this good.
FIGURE_READINGS = [(psm, scale) for psm in (PSM_AUTO, PSM_SPARSE_TEXT, PSM_SINGLE_BLOCK) for scale in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15)][1:]
FIGURE_GOOD_ENOUGH = 0.8


def _token_share(regions: Iterable[Dict], with_estimated: bool = False) -> float:
    """Matched (and optionally estimated) chandra tokens / all chandra tokens over the given regions."""
    regions = list(regions)
    total = sum(r.get("n_chandra_tokens", 0) for r in regions)
    placed = sum(r.get("n_matched", 0) + (r.get("n_estimated", 0) if with_estimated else 0) for r in regions)
    return round(placed / total, 4) if total else 0.0


def make_page_id(relpath: Path) -> str:
    """File-system and URL safe page id from a path relative to the input folder."""
    stem = str(relpath.with_suffix("")).replace("\\", "/").replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "page"


def discover_sources(input_dir: Path) -> List[SourcePage]:
    """Find supported scans under input_dir (recursively), with unique page ids, in sorted order."""
    input_dir = Path(input_dir)
    paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith("."))
    pages: List[SourcePage] = []
    used: Dict[str, int] = {}
    for seq, path in enumerate(paths):
        relpath = path.relative_to(input_dir)
        page_id = make_page_id(relpath)
        if page_id in used:
            used[page_id] += 1
            page_id = f"{page_id}_{path.suffix.lower().lstrip('.')}{used[page_id]}"
        used.setdefault(page_id, 0)
        pages.append(SourcePage(page_id, path.resolve(), relpath.as_posix(), seq))
    return pages


class Pipeline:
    """
    Runs the full pipeline over a folder of scans.

    Args:
        config: Pipeline settings
        chandra_client: Optional OpenAI-compatible async client (tests inject a fake)
        layout_factory: Optional callable returning an object with detect(page_id, processed_path)
    """

    def __init__(self, config: PipelineConfig, chandra_client: Any = None, layout_factory: Any = None) -> None:
        self.config = config
        self.chandra_client = chandra_client
        self.layout_factory = layout_factory
        self._detector: Any = None
        self.summary = RunSummary()

    # ---- stage helpers -------------------------------------------------------------------

    def _detect(self, page_id: str, processed_path: str) -> pd.DataFrame:
        """Runs in the single layout thread; the model is created there on first use."""
        if self._detector is None:
            if self.layout_factory is not None:
                self._detector = self.layout_factory(self.config)
            else:
                from newspaper_pipeline.stages.layout import LayoutDetector, silence_bbox_progress_bars

                silence_bbox_progress_bars()
                self._detector = LayoutDetector(self.config)
        return self._detector.detect(page_id, processed_path)

    async def _process_region(self, page: SourcePage, region: Dict, transcribe_figure: bool = False) -> Dict:
        loop = asyncio.get_running_loop()
        # Figures are mostly advertisements set as pictures: read them like text when no other region
        # already covers their text (see regions.figures_with_own_text). Any illustration chandra
        # describes goes to image_descriptions, not the text.
        region_class = "text" if region["class"] == "figure" and transcribe_figure else region["class"]
        priority = (page.seq, int(region["reading_order"]))
        chandra_task = asyncio.create_task(self.chandra.ocr_region(region["crop_path"], region_class, priority))

        words: List[Dict] = []
        tesseract_error = ""
        if region_class != "figure":
            try:
                words = await loop.run_in_executor(self.tess_pool, tesseract_words, region["crop_path"], self.config.tessdata, self.config.tesseract_lang)
            except Exception as exc:
                tesseract_error = f"tesseract: {exc}"
                logger.warning(f"{page.page_id} {region['box_page_id']}: {tesseract_error}")

        result: Dict[str, Any] = {"error": tesseract_error, "ocr_class": region_class}
        try:
            chandra = await chandra_task
        except Exception as exc:
            logger.error(f"{page.page_id} {region['box_page_id']}: chandra failed: {exc!r}")
            result["error"] = "; ".join(filter(None, [tesseract_error, f"chandra: {exc!r}"]))
            result["chandra_failed"] = True
            return result

        # Output that isn't a transcription would be withheld and its text lost: ask again first
        for attempt in range(1, self.config.chandra_quality_retries + 1):
            problem = transcription_problem(region_class, chandra["text"], chandra["html"], chandra["finish_reasons"])
            if problem is None:
                break
            logger.info(f"{page.page_id} {region['box_page_id']}: {problem}; asking chandra again (retry {attempt})")
            try:
                chandra = await self.chandra.ocr_region(region["crop_path"], region_class, priority, attempt)
            except Exception as exc:
                logger.warning(f"{page.page_id} {region['box_page_id']}: chandra retry failed: {exc!r}")
                break

        result.update(chandra, n_chandra_tokens=len(chandra["text"].split()), n_tesseract_words=len(words))
        alignment = None
        if region_class != "figure" and words:
            alignment = await asyncio.to_thread(align_region, chandra["text"], words, crop_path=region["crop_path"])
        if region["class"] == "figure" and region_class != "figure" and chandra["text"].strip():
            # Display type in adverts (letters 100-200 px tall) and scattered words often defeat the
            # default reading: try others and keep whichever agrees best with chandra
            for psm, scale in FIGURE_READINGS:
                if alignment is not None and alignment["score"] >= FIGURE_GOOD_ENOUGH:
                    break
                try:
                    other = await loop.run_in_executor(
                        self.tess_pool, tesseract_words, region["crop_path"], self.config.tessdata, self.config.tesseract_lang, scale, psm
                    )
                except Exception as exc:
                    logger.warning(f"{page.page_id} {region['box_page_id']}: tesseract psm {psm} at {scale}x: {exc}")
                    continue
                if not other:
                    continue
                candidate = await asyncio.to_thread(align_region, chandra["text"], other, crop_path=region["crop_path"])
                if alignment is None or candidate["score"] > alignment["score"]:
                    alignment = candidate
        if alignment is not None:
            result.update(
                words=alignment["words"],
                lines=alignment["lines"],
                alignment_score=alignment["score"],
                n_chandra_tokens=alignment["n_chandra_tokens"],
                n_tesseract_words=alignment["n_tesseract_words"],
                n_matched=alignment["n_matched"],
                n_estimated=alignment["n_estimated"],
            )
        return result

    def _finalize_page(self, page: SourcePage, converted: Dict, prep: Dict, regions: pd.DataFrame, results: List[Dict], timings: Dict) -> Dict:
        """Map everything back to original-scan pixels and write IIIF, overlay and flat exports."""
        transform = PageTransform.load(Path(prep["transform_path"]))

        def box(entry: Dict, offset: tuple = (0, 0)) -> Dict:
            x, y, w, h = transform.box_to_original((entry["x1"], entry["y1"], entry["x2"], entry["y2"]), offset)
            return {"x": x, "y": y, "w": w, "h": h}

        # Typical word size on this page, for judging how much text a region can physically hold
        word_areas = [(w["x2"] - w["x1"]) * (w["y2"] - w["y1"]) for r in results for w in r.get("words", [])]
        word_area = float(np.median(word_areas)) if word_areas else 3000.0

        page_regions: List[Dict] = []
        errors: List[str] = []
        suspects: List[Dict] = []
        for region, result in zip(regions.to_dict("records"), results):
            pad = region.get("crop_pad", 0)
            offset = (region["x1"] - pad, region["y1"] - pad)  # crop pixels -> processed page pixels
            entry = {
                "box_page_id": region["box_page_id"],
                "class": region["class"],
                "reading_order": int(region["reading_order"]),
                "recovered": bool(region.get("recovered", False)),
                "box": box(region),
                **{k: v for k, v in result.items() if k not in ("words", "lines")},
                "lines": [{**line, "box": box(line, offset)} for line in result.get("lines", [])],
                "words": [
                    {
                        **word,
                        "box": box(word, offset),
                        # a word printed on two lines (hyphenated): both parts, head first
                        **({"part_boxes": [box(dict(zip(("x1", "y1", "x2", "y2"), part)), offset) for part in word["parts"]]} if word.get("parts") else {}),
                    }
                    for word in result.get("words", [])
                ],
            }
            if result.get("error"):
                errors.append(f"{region['box_page_id']}: {result['error']}")
            reason = assess_region(
                result.get("ocr_class", region["class"]),
                result.get("text", ""),
                result.get("html", ""),
                region["x2"] - region["x1"],
                region["y2"] - region["y1"],
                word_area,
                result.get("n_chandra_tokens", len(result.get("text", "").split())),
                result.get("n_tesseract_words", 0),
                result.get("alignment_score", 0.0),
                result.get("finish_reasons", []),
                # touching the left or right edge of the scan (original-scan coordinates: deskewing
                # shifts a strip at x=0 several pixels inwards)
                at_page_edge=entry["box"]["x"] <= 0.02 * converted["width"] or entry["box"]["x"] + entry["box"]["w"] >= 0.98 * converted["width"],
                from_figure=region["class"] == "figure" and result.get("ocr_class") == "text",
                figure_min_alignment=FIGURE_MIN_ALIGNMENT,
            )
            if reason:
                # Keep the raw output for review in regions.parquet, but publish nothing from it
                entry.update(suspect=reason, lines=[], words=[])
                suspects.append({"box_page_id": region["box_page_id"], "reason": reason})
            page_regions.append(entry)

        page_info = {
            "page_id": page.page_id,
            "source_relpath": page.source_relpath,
            "width": converted["width"],
            "height": converted["height"],
        }
        documents = iiif.build_page_documents(page_info, page_regions, self.config)
        for name, document in documents.items():
            for problem in iiif.validate(document)[:5]:
                errors.append(f"IIIF {name}: {problem}")

        out_dir = Path(self.config.output_dir) / page.page_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / SUCCESS_MARKER).unlink(missing_ok=True)
        iiif.write_documents(out_dir, documents)
        write_flat_exports(out_dir, page_info, page_regions)
        write_overlay(str(page.source_path), (converted["width"], converted["height"]), page_regions, out_dir / "overlay.jpg", self.config.overlay_max_width)

        published = [r for r in page_regions if not r.get("suspect") and r.get("ocr_class", r["class"]) != "figure"]
        report = {
            "page_id": page.page_id,
            "source": page.source_relpath,
            "size": [converted["width"], converted["height"]],
            "deskew_angle": prep["angle"],
            "content_box": prep["content_box"],
            "regions": len(page_regions),
            "words": sum(len(r["words"]) for r in page_regions),
            "lines": sum(len(r["lines"]) for r in page_regions),
            # share of chandra tokens placed on a Tesseract box, in regions where Tesseract found words
            "alignment_score": _token_share(r for r in published if r.get("n_tesseract_words")),
            # share of all published chandra text that has word boxes, matched or estimated
            "word_box_coverage": _token_share(published, with_estimated=True),
            "estimated_words": sum(r.get("n_estimated", 0) for r in published),
            "chandra_cached_regions": sum(1 for r in page_regions if r.get("cached")),
            "suspect_regions": suspects,
            "timings_seconds": timings,
            "errors": errors,
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        if not errors:
            (out_dir / SUCCESS_MARKER).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        return report

    async def _process_page(self, page: SourcePage) -> None:
        loop = asyncio.get_running_loop()
        config = self.config
        async with self.page_slots:
            timings: Dict[str, float] = {}
            start = time.perf_counter()

            def lap(stage: str, since: float) -> float:
                now = time.perf_counter()
                timings[stage] = round(now - since, 2)
                return now

            try:
                async with self.convert_slots:
                    t = time.perf_counter()
                    converted = await loop.run_in_executor(
                        self.convert_pool, convert_page, str(page.source_path), page.source_relpath, page.page_id, str(config.work_dir)
                    )
                    lap("convert", t)
                async with self.preprocess_slots:
                    t = time.perf_counter()
                    prep = await loop.run_in_executor(
                        self.preprocess_pool,
                        partial(
                            preprocess_page,
                            page.page_id,
                            converted["png_path"],
                            str(config.work_dir),
                            mask_borders=config.mask_borders,
                            deskew=config.deskew,
                            max_deskew_angle=config.max_deskew_angle,
                            enhance_page=config.enhance,
                            clahe_clip_limit=config.clahe_clip_limit,
                            denoise=config.denoise,
                        ),
                    )
                    lap("preprocess", t)
                t = time.perf_counter()
                regions = await loop.run_in_executor(self.layout_executor, self._detect, page.page_id, prep["processed_path"])
                lap("layout", t)
                logger.info(f"{page.page_id}: {len(regions)} regions, skew {prep['angle']:+.2f} deg; queued for OCR")

                t = time.perf_counter()
                own_text = figures_with_own_text(regions) if config.transcribe_figures else set()
                results = await asyncio.gather(
                    *(self._process_region(page, region, region["box_page_id"] in own_text) for region in regions.to_dict("records"))
                )
                lap("ocr_and_alignment", t)

                t = time.perf_counter()
                report = await asyncio.to_thread(self._finalize_page, page, converted, prep, regions, results, timings)
                lap("iiif_and_exports", t)
                timings["total"] = round(time.perf_counter() - start, 2)
                (Path(config.output_dir) / page.page_id / "report.json").write_text(json.dumps({**report, "timings_seconds": timings}, indent=2))

                if report["errors"]:
                    self.summary.pages_incomplete.append(page.page_id)
                    logger.warning(f"{page.page_id}: written with {len(report['errors'])} problem(s); re-run to retry (see report.json)")
                else:
                    self.summary.pages_succeeded.append(page.page_id)
                    logger.info(
                        f"{page.page_id}: done in {timings['total']:.0f}s - {report['regions']} regions, {report['words']} words, "
                        f"alignment {report['alignment_score']:.2f}, word box coverage {report['word_box_coverage']:.2f}, "
                        f"{len(report['suspect_regions'])} suspect region(s) withheld"
                    )
            except Exception as exc:
                self.summary.pages_failed[page.page_id] = repr(exc)
                logger.exception(f"{page.page_id}: failed")
            finally:
                if not config.keep_temp:
                    cleanup.cleanup_page(config.work_dir, page.page_id)

    async def _report_progress(self, total: int, interval: float = 30.0) -> None:
        while True:
            await asyncio.sleep(interval)
            done = len(self.summary.pages_succeeded) + len(self.summary.pages_incomplete) + len(self.summary.pages_failed)
            stats = self.chandra.stats
            logger.info(
                f"progress: {done}/{total} pages, chandra queue {self.chandra.pending}, "
                f"requests {stats['requests']}, cache hits {stats['cache_hits']}, failures {stats['failures']}"
            )

    # ---- entry point -----------------------------------------------------------------------

    def write_collection(self) -> Optional[Path]:
        """Rebuild collection.json from every manifest in the output folder."""
        manifests = []
        for path in sorted(Path(self.config.output_dir).glob("*/manifest.json")):
            try:
                manifest = json.loads(path.read_text())
                manifests.append({"id": manifest["id"], "label": manifest["label"]})
            except (OSError, json.JSONDecodeError, KeyError):
                logger.warning(f"Skipping unreadable manifest {path}")
        if not manifests:
            return None
        out = Path(self.config.output_dir) / "collection.json"
        out.write_text(json.dumps(iiif.build_collection(manifests, self.config), indent=1))
        return out

    async def run(self, sources: Optional[List[SourcePage]] = None, force: bool = False, limit: Optional[int] = None) -> RunSummary:
        """
        Process every page that hasn't completed yet.

        Args:
            sources: Pages to process (defaults to discover_sources(config.input_dir))
            force: Re-process pages even if they already have a _SUCCESS marker
            limit: Process at most this many pages

        Returns:
            RunSummary
        """
        config = self.config
        started = time.perf_counter()
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.work_dir).mkdir(parents=True, exist_ok=True)
        cleanup.remove_partial_files(Path(config.output_dir))

        sources = discover_sources(Path(config.input_dir)) if sources is None else sources
        todo = [p for p in sources if force or not (Path(config.output_dir) / p.page_id / SUCCESS_MARKER).exists()]
        self.summary.pages_total = len(sources)
        self.summary.pages_skipped = len(sources) - len(todo)
        if limit is not None:
            todo = todo[:limit]
        logger.info(f"{len(sources)} scans found, {self.summary.pages_skipped} already done, processing {len(todo)}")
        if not todo:
            self.write_collection()
            return self.summary

        spawn = multiprocessing.get_context("spawn")
        self.convert_pool = ProcessPoolExecutor(config.convert_workers, mp_context=spawn)
        self.preprocess_pool = ProcessPoolExecutor(config.preprocess_workers, mp_context=spawn)
        self.tess_pool = ProcessPoolExecutor(config.tesseract_workers, mp_context=spawn, initializer=init_worker)
        self.layout_executor = ThreadPoolExecutor(1, thread_name_prefix="layout")
        self.page_slots = asyncio.Semaphore(config.max_pages_in_flight)
        self.convert_slots = asyncio.Semaphore(config.convert_workers)
        self.preprocess_slots = asyncio.Semaphore(config.preprocess_workers)
        self.chandra = ChandraPool(config, self.chandra_client)
        await self.chandra.start()
        progress = asyncio.create_task(self._report_progress(len(todo)))
        try:
            await asyncio.gather(*(self._process_page(page) for page in todo))
        finally:
            progress.cancel()
            await self.chandra.close()
            for pool in (self.convert_pool, self.preprocess_pool, self.tess_pool):
                pool.shutdown(wait=True, cancel_futures=True)
            self.layout_executor.shutdown(wait=True, cancel_futures=True)
            cleanup.cleanup_run(Path(config.work_dir), [p.page_id for p in todo], config.keep_temp)

        self.write_collection()
        self.summary.seconds = round(time.perf_counter() - started, 1)
        self.summary.chandra = dict(self.chandra.stats)
        return self.summary
