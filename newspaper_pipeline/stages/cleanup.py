"""
Stage 7: remove temporary files.

Each page keeps everything temporary (converted PNG, processed page, transform, region
crops) under work_dir/page_id, so cleanup is a single folder removal once the page's
outputs are written. Source scans are never touched: they are what Cantaloupe serves.
"""

import logging
import shutil
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def cleanup_page(work_dir: Path, page_id: str) -> None:
    """Delete one page's temporary folder."""
    page_dir = Path(work_dir) / page_id
    if page_dir.exists():
        shutil.rmtree(page_dir, ignore_errors=True)


def cleanup_run(work_dir: Path, page_ids: Iterable[str], keep_temp: bool) -> None:
    """
    End of run sweep: remove the page folders this run created, then the work folder if it is empty.

    Only named page folders are deleted, so pointing work_dir at a folder that holds other
    files never loses them.
    """
    work_dir = Path(work_dir)
    if not work_dir.exists():
        return
    if keep_temp:
        logger.info(f"Temporary files kept in {work_dir}")
        return
    for page_id in page_ids:
        cleanup_page(work_dir, page_id)
    try:
        work_dir.rmdir()
        logger.info(f"Removed temporary folder {work_dir}")
    except OSError:
        logger.info(f"Left {work_dir} in place because it contains other files")


def remove_partial_files(output_dir: Path) -> None:
    """Remove half-written *.tmp files left behind by an interrupted run."""
    for tmp in Path(output_dir).rglob("*.tmp"):
        tmp.unlink(missing_ok=True)
