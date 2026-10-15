"""Orchestration behaviour with a fake layout model and fake chandra: concurrency, ordering, failures, resume."""

import asyncio
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

from newspaper_pipeline.config import PipelineConfig
from newspaper_pipeline.orchestrator import SUCCESS_MARKER, Pipeline


class FakeLayout:
    """Three fixed regions per page; records when each page reached layout."""

    events = []

    def __init__(self, config):
        pass

    def detect(self, page_id, processed_path):
        FakeLayout.events.append(("layout", page_id, time.perf_counter()))
        gray = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
        rows = []
        for n, (y1, y2) in enumerate([(50, 150), (200, 300), (350, 450)], start=1):
            path = f"{processed_path}.{n}.png"
            cv2.imwrite(path, gray[y1:y2, 50:550])
            rows.append({"box_page_id": f"B0C1R{n}", "class": "text", "reading_order": n, "x1": 50, "y1": y1, "x2": 550, "y2": y2, "crop_path": path})
        return pd.DataFrame(rows)


class FakeChandra:
    def __init__(self, fail_text=None, delay=0.15):
        self.active = 0
        self.max_active = 0
        self.calls = []
        self.fail_text = fail_text
        self.delay = delay
        self.chat = SimpleNamespace(completions=self)

    async def create(self, model, messages, max_tokens=None, temperature=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            self.calls.append(time.perf_counter())
            image = messages[0]["content"][1]["image_url"]["url"]
            if self.fail_text and self.fail_text in image:
                raise ValueError("model rejected the image")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="<p>sample words</p>"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
            )
        finally:
            self.active -= 1


def make_pages(folder, count):
    folder.mkdir()
    for n in range(count):
        # unique noise per page, so no two regions share a chandra cache entry
        page = np.random.default_rng(n).integers(215, 245, (600, 600)).astype(np.uint8)
        cv2.putText(page, f"page {n} sample words", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3)
        cv2.imwrite(str(folder / f"p{n}.png"), page)


def config_for(tmp_path):
    return PipelineConfig(
        input_dir=tmp_path / "scans",
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        tesseract_workers=2,
        max_pages_in_flight=3,
        deskew=False,
    )


def test_pages_flow_through_stages_concurrently(tmp_path):
    make_pages(tmp_path / "scans", 4)
    FakeLayout.events = []
    fake = FakeChandra()
    summary = asyncio.run(Pipeline(config_for(tmp_path), chandra_client=fake, layout_factory=FakeLayout).run())

    assert sorted(summary.pages_succeeded) == ["p0", "p1", "p2", "p3"]
    assert fake.max_active == 2  # never more than the endpoint allows
    assert len(fake.calls) == 12
    # Later pages reached layout while chandra was still working on earlier ones
    layout_times = sorted(t for _, _, t in FakeLayout.events)
    assert layout_times[2] < fake.calls[-1] and layout_times[1] < fake.calls[3]
    for n in range(4):
        assert (tmp_path / "out" / f"p{n}" / SUCCESS_MARKER).exists()
    assert not (tmp_path / "work").exists()
    assert (tmp_path / "out" / "collection.json").exists()


def test_failed_region_leaves_page_unfinished_until_rerun(tmp_path):
    make_pages(tmp_path / "scans", 1)
    config = config_for(tmp_path)

    # Every request fails on the first run (all images are data:image/png URLs)
    broken = FakeChandra(fail_text="data:image/png")
    first = asyncio.run(Pipeline(config, chandra_client=broken, layout_factory=FakeLayout).run())
    assert first.pages_incomplete == ["p0"]
    assert not (tmp_path / "out" / "p0" / SUCCESS_MARKER).exists()
    assert (tmp_path / "out" / "p0" / "manifest.json").exists()  # partial output is still written
    assert "chandra" in (tmp_path / "out" / "p0" / "report.json").read_text()

    working = FakeChandra()
    second = asyncio.run(Pipeline(config, chandra_client=working, layout_factory=FakeLayout).run())
    assert second.pages_succeeded == ["p0"] and second.pages_skipped == 0
    assert len(working.calls) == 3
    assert (tmp_path / "out" / "p0" / SUCCESS_MARKER).exists()


def test_keep_temp_preserves_work_files(tmp_path):
    make_pages(tmp_path / "scans", 1)
    config = PipelineConfig(**{**config_for(tmp_path).__dict__, "keep_temp": True})
    asyncio.run(Pipeline(config, chandra_client=FakeChandra(delay=0), layout_factory=FakeLayout).run())
    assert (tmp_path / "work" / "p0" / "processed.png").exists()
    assert (tmp_path / "work" / "p0" / "transform.json").exists()
