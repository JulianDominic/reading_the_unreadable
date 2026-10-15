import asyncio
import io
from types import SimpleNamespace

import httpx
import openai
import pytest
from PIL import Image

from newspaper_pipeline.config import PipelineConfig
from newspaper_pipeline.stages.ocr_chandra import ChandraPool, html_to_text, knit_texts, pad_to_aspect, split_offsets

CHANDRA_SAMPLE = """<div data-bbox="41 63 989 900" data-label="Table">
<table border="0"><thead><tr><th colspan="2">Notes in circulation.....</th><th>$</th><th>76,223</th></tr></thead>
<tbody><tr><td colspan="2">Gold Bullion.....</td><td></td><td>4,807</td></tr></tbody></table></div>
<div data-bbox="0 900 1000 1000" data-label="Text"><p>The undersigned having<br>purchased the <b>Colonial</b> Hotel</p></div>"""


def test_html_to_text_tables_lines_and_labels():
    text, labels, dropped = html_to_text(CHANDRA_SAMPLE)
    assert dropped == []
    assert labels == ["Table", "Text"]
    assert text.split("\n") == [
        "Notes in circulation.....\t$\t76,223",
        "Gold Bullion.....\t\t4,807",
        "The undersigned having",
        "purchased the Colonial Hotel",
    ]


def test_html_to_text_passes_plain_text_and_strips_fences():
    assert html_to_text("Plain words\nsecond line") == ("Plain words\nsecond line", [], [])
    assert html_to_text("```html\n<p>x</p>\n```")[0] == "x"
    assert html_to_text('<div data-label="Image"><img alt="A ship at anchor"></div>') == ("A ship at anchor", ["Image"], [])
    assert html_to_text("") == ("", [], [])


def test_html_to_text_drops_image_descriptions_from_transcriptions():
    html = (
        '<div data-label="Text"><p>(Signed) WM. C. WARD, Manager.</p></div>'
        '<div data-label="Image"><img alt="Royal coat of arms"/>The image is a detailed illustration.</div>'
    )
    text, labels, dropped = html_to_text(html, drop_labels=("Image",))
    assert text == "(Signed) WM. C. WARD, Manager."
    assert labels == ["Text", "Image"]
    assert dropped == ["Royal coat of arms The image is a detailed illustration."]


def test_knit_texts_joins_on_overlap():
    a = "first line of the column\nthe overlapping part appears here"
    b = "the overlapping part appears here\nand the column continues"
    assert knit_texts(a, b) == "first line of the column\nthe overlapping part appears here\nand the column continues"
    assert knit_texts("no overlap at all", "something else") == "no overlap at all\nsomething else"


def test_split_offsets():
    assert split_offsets(100, 100, 1.5, 0.2) == [(0, 100)]
    offsets = split_offsets(1000, 100, 1.5, 0.2)
    assert offsets[0] == (0, 150) and offsets[-1][1] == 1000
    assert all(b - a == 150 for a, b in offsets)
    assert all(offsets[i + 1][0] < offsets[i][1] for i in range(len(offsets) - 1))  # segments overlap


def png_bytes(color=0, size=(40, 20)):
    buffer = io.BytesIO()
    Image.new("L", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeCompletions:
    """Records concurrency and call order; optionally fails the first N calls."""

    def __init__(self, fail_first=0, delay=0.02):
        self.active = 0
        self.max_active = 0
        self.calls = []
        self.fail_first = fail_first
        self.delay = delay

    async def create(self, model, messages, max_tokens=None, temperature=None):
        self.temperatures = getattr(self, "temperatures", []) + [temperature]
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            self.calls.append(messages[0]["content"][0]["text"])
            if self.fail_first > 0:
                self.fail_first -= 1
                raise openai.APIConnectionError(request=httpx.Request("POST", "http://test"))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=f"<p>{messages[0]['content'][0]['text']}</p>"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )
        finally:
            self.active -= 1


def make_pool(tmp_path, completions, concurrency=2):
    config = PipelineConfig(cache_dir=tmp_path / "cache", chandra_concurrency=concurrency, chandra_retry_min_wait=0, chandra_retry_max_wait=0)
    return ChandraPool(config, SimpleNamespace(chat=SimpleNamespace(completions=completions)))


def test_pool_respects_concurrency_and_priority(tmp_path):
    completions = FakeCompletions()

    async def scenario():
        pool = make_pool(tmp_path, completions, concurrency=2)
        # Queue before starting so the workers pick strictly by priority
        futures = [pool.submit(png_bytes(i), f"job {p}", (p,)) for i, p in enumerate([5, 1, 4, 2, 3, 0])]
        await pool.start()
        results = await asyncio.gather(*futures)
        await pool.close()
        return results

    results = asyncio.run(scenario())
    assert completions.max_active == 2
    assert completions.calls == ["job 0", "job 1", "job 2", "job 3", "job 4", "job 5"]
    assert set(completions.temperatures) == {0.0}  # deterministic decoding for OCR
    assert all(r["finish_reason"] == "stop" and not r["cached"] for r in results)


def test_pool_caches_and_retries(tmp_path):
    completions = FakeCompletions(fail_first=2)

    async def scenario():
        pool = make_pool(tmp_path, completions, concurrency=1)
        await pool.start()
        first = await pool.submit(png_bytes(), "prompt", (0,))
        second = await pool.submit(png_bytes(), "prompt", (0,))
        await pool.close()
        return pool, first, second

    pool, first, second = asyncio.run(scenario())
    assert len(completions.calls) == 3  # two failures retried, then success; the repeat came from cache
    assert first["cached"] is False and second["cached"] is True
    assert pool.stats["cache_hits"] == 1


def test_ocr_region_splits_tall_crops(tmp_path):
    completions = FakeCompletions()
    crop = tmp_path / "tall.png"
    Image.new("L", (100, 400), 255).save(crop)

    async def scenario():
        pool = make_pool(tmp_path, completions)
        await pool.start()
        result = await pool.ocr_region(str(crop), "text", (0, 1))
        figure = await pool.ocr_region(str(crop), "figure", (0, 2))
        await pool.close()
        return result, figure

    result, figure = asyncio.run(scenario())
    assert result["segments"] == len(split_offsets(400, 100, 1.5, 0.2)) > 1
    assert figure["segments"] == 1  # figures are never split
    assert "Describe the graphic" in figure["text"]


def test_pool_surfaces_non_retryable_errors(tmp_path):
    class Broken:
        async def create(self, **kwargs):
            raise ValueError("bad request")

    async def scenario():
        pool = make_pool(tmp_path, Broken())
        await pool.start()
        try:
            with pytest.raises(ValueError):
                await pool.submit(png_bytes(), "p", (0,))
        finally:
            await pool.close()
        return pool

    assert asyncio.run(scenario()).stats["failures"] == 1


def test_pad_to_aspect_limits_wide_crops():
    wide = Image.new("L", (5814, 20), 0)
    padded = pad_to_aspect(wide, 20.0)
    assert padded.size == (5814, 291)
    assert padded.getpixel((0, 0)) == 255 and padded.getpixel((0, 145)) == 0  # content centred on white
    normal = Image.new("RGB", (400, 100))
    assert pad_to_aspect(normal, 20.0) is normal


def test_token_cap_scales_with_region_size(tmp_path):
    pool = make_pool(tmp_path, FakeCompletions())
    assert pool.token_cap("figure", 2000, 2000) == 512  # descriptions are short
    assert pool.token_cap("text", 21, 296) == 256  # tiny sliver: floor
    assert pool.token_cap("text", 900, 1200) == 3240  # about 540 words of room at 6 tokens each
    assert pool.token_cap("text", 6000, 9000) == 8192  # never above the configured maximum
