"""
Stage 4a: text from chandra-ocr-2 on a remote OpenAI-compatible (vLLM) endpoint.

The endpoint allows only a couple of concurrent sessions, so all requests go through a
priority queue served by exactly `chandra_concurrency` worker tasks. Priorities are
(page sequence, reading order), so the earliest page finishes first and can be written
out and cleaned up while later pages wait their turn.

Chandra answers with HTML blocks (<div data-bbox=".." data-label="Text">...) even when
asked for plain text; html_to_text turns that into plain lines for alignment and export.
"""

import asyncio
import base64
import difflib
import hashlib
import io
import itertools
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from newspaper_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "ul", "ol", "section", "header", "footer", "caption"}


IMAGE_LABELS = ("Image", "Figure", "Picture")


def html_to_text(content: str, drop_labels: Tuple[str, ...] = ()) -> Tuple[str, List[str], List[str]]:
    """
    Convert chandra's HTML output to plain text.

    Tables become tab separated rows, <br> and block elements become line breaks, and
    images contribute their alt text. Plain text responses pass through unchanged.

    Args:
        content: Raw model output
        drop_labels: Block labels to leave out of the text. Inside text regions chandra's
            "Image" blocks are descriptions of an illustration, not transcriptions.

    Returns:
        (plain text, chandra block labels such as "Text", "Table", "Section-Header",
        plain text of each dropped block)
    """
    if not content:
        return "", [], []
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", stripped).strip()
    if "<" not in stripped or not re.search(r"</?[a-zA-Z][^>]*>", stripped):
        return stripped, [], []

    from bs4 import BeautifulSoup, NavigableString, Tag

    soup = BeautifulSoup(stripped, "html.parser")
    labels = [tag.get("data-label") for tag in soup.find_all(attrs={"data-label": True})]
    dropped = []
    for tag in soup.find_all(attrs={"data-label": True}):
        if tag.get("data-label") in drop_labels:
            alt = " ".join(img.get("alt", "") for img in tag.find_all("img"))
            dropped.append(" ".join(f"{alt} {tag.get_text(' ', strip=True)}".split()))
            tag.decompose()

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [" ".join(cell.get_text(" ", strip=True).split()) for cell in tr.find_all(["td", "th"])]
            rows.append("\t".join(cells))
        table.replace_with(NavigableString("\n" + "\n".join(rows) + "\n"))
    for img in soup.find_all("img"):
        img.replace_with(NavigableString(f"\n{img.get('alt', '')}\n" if img.get("alt") else ""))
    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for tag in soup.find_all(BLOCK_TAGS):
        if isinstance(tag, Tag):
            tag.insert_before(NavigableString("\n"))
            tag.insert_after(NavigableString("\n"))

    lines = []
    for line in soup.get_text().split("\n"):
        if "\t" in line:
            cleaned = "\t".join(" ".join(cell.split()) for cell in line.split("\t"))
        else:
            cleaned = " ".join(line.split())
        if cleaned.strip():
            lines.append(cleaned)
    return "\n".join(lines), [label for label in labels if label], dropped


def knit_texts(first: str, second: str, min_overlap: int = 12) -> str:
    """
    Join the transcriptions of two vertically overlapping segments.

    Looks for the longest common substring between the end of the first text and the
    start of the second (the overlap region) and joins them there. Falls back to a plain
    newline join when no convincing overlap is found.
    """
    if not first:
        return second
    if not second:
        return first
    tail_start = int(len(first) * 0.4)
    head_end = int(len(second) * 0.6) + 1
    matcher = difflib.SequenceMatcher(None, first, second, autojunk=False)
    match = matcher.find_longest_match(tail_start, len(first), 0, head_end)
    if match.size < min_overlap:
        return first.rstrip() + "\n" + second.lstrip()
    return first[: match.a] + second[match.b :]


def pad_to_aspect(image, max_aspect: float):
    """Pad a very wide image with white top and bottom so width / height <= max_aspect."""
    from PIL import Image

    width, height = image.size
    if height <= 0 or width / height <= max_aspect:
        return image
    new_height = int(-(-width // max_aspect))
    fill = 255 if image.mode in ("L", "1") else (255,) * len(image.getbands())
    padded = Image.new(image.mode, (width, new_height), fill)
    padded.paste(image, (0, (new_height - height) // 2))
    return padded


def split_offsets(height: int, width: int, max_ratio: float, overlap_fraction: float, max_segments: int = 40) -> List[Tuple[int, int]]:
    """
    Vertical (y_start, y_end) segments for a tall crop, each at most max_ratio * width high,
    overlapping by overlap_fraction. A crop within the ratio is one segment.
    """
    segment_height = max(1, int(width * max_ratio))
    if height <= segment_height:
        return [(0, height)]
    step = max(1, int(segment_height * (1 - overlap_fraction)))
    offsets: List[Tuple[int, int]] = []
    y = 0
    while len(offsets) < max_segments:
        if y + segment_height >= height:
            offsets.append((max(0, height - segment_height), height))
            break
        offsets.append((y, y + segment_height))
        y += step
    return offsets


@dataclass(order=True)
class _Job:
    priority: Tuple
    seq: int
    image_png: bytes = field(compare=False)
    prompt: str = field(compare=False)
    cache_key: str = field(compare=False)
    future: asyncio.Future = field(compare=False)
    max_tokens: int = field(compare=False, default=8192)
    temperature: float = field(compare=False, default=0.0)


class ChandraPool:
    """
    Fixed-size pool of async workers talking to the chandra endpoint, with an on-disk response cache.

    Usage:
        pool = ChandraPool(config); await pool.start(); ...; await pool.close()
    """

    def __init__(self, config: PipelineConfig, client: Any = None) -> None:
        self.config = config
        self.client = client
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.queue: "asyncio.PriorityQueue[_Job]" = asyncio.PriorityQueue()
        self.workers: List[asyncio.Task] = []
        self._seq = itertools.count()
        self.stats = {"requests": 0, "cache_hits": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def start(self) -> None:
        if self.client is None:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                base_url=self.config.chandra_api_base,
                api_key=self.config.chandra_api_key or "none",
                timeout=self.config.chandra_timeout,
                max_retries=0,  # retries are handled with backoff in _request
            )
        self.workers = [asyncio.create_task(self._worker(i), name=f"chandra-{i}") for i in range(self.config.chandra_concurrency)]

    async def close(self) -> None:
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        close = getattr(self.client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    @property
    def pending(self) -> int:
        return self.queue.qsize()

    def _cache_key(self, image_png: bytes, prompt: str, max_tokens: int, temperature: float) -> str:
        digest = hashlib.sha256()
        digest.update(f"{self.config.chandra_model}|{max_tokens}|{temperature}|{prompt}|".encode())
        digest.update(image_png)
        return digest.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def submit(
        self, image_png: bytes, prompt: str, priority: Tuple, max_tokens: Optional[int] = None, temperature: Optional[float] = None
    ) -> "asyncio.Future[Dict]":
        """Queue one image; resolves to {content, finish_reason, prompt_tokens, completion_tokens, cached}."""
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        max_tokens = max_tokens or self.config.chandra_max_tokens
        temperature = self.config.chandra_temperature if temperature is None else temperature
        key = self._cache_key(image_png, prompt, max_tokens, temperature)
        cache_path = self._cache_path(key)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                cached["cached"] = True
                self.stats["cache_hits"] += 1
                future.set_result(cached)
                return future
            except (OSError, json.JSONDecodeError):
                logger.warning(f"Ignoring unreadable cache entry {cache_path}")
        self.queue.put_nowait(_Job(priority, next(self._seq), image_png, prompt, key, future, max_tokens, temperature))
        return future

    async def _worker(self, index: int) -> None:
        while True:
            job = await self.queue.get()
            try:
                if job.future.cancelled():
                    continue
                result = await self._request(job.image_png, job.prompt, job.max_tokens, job.temperature)
                path = self._cache_path(job.cache_key)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(result))
                tmp.replace(path)
                result["cached"] = False
                if not job.future.done():
                    job.future.set_result(result)
            except asyncio.CancelledError:
                if not job.future.done():
                    job.future.cancel()
                raise
            except Exception as exc:  # surfaced to the region that asked
                self.stats["failures"] += 1
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self.queue.task_done()

    async def _request(self, image_png: bytes, prompt: str, max_tokens: int, temperature: float) -> Dict:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
        from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

        def retryable(exc: BaseException) -> bool:
            if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
                return True
            return isinstance(exc, APIStatusError) and exc.status_code >= 500

        image_b64 = base64.b64encode(image_png).decode("ascii")
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.config.chandra_retries),
            wait=wait_exponential(multiplier=2, min=self.config.chandra_retry_min_wait, max=self.config.chandra_retry_max_wait),
            retry=retry_if_exception(retryable),
            reraise=True,
        ):
            with attempt:
                self.stats["requests"] += 1
                response = await self.client.chat.completions.create(
                    model=self.config.chandra_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                            ],
                        }
                    ],
                )
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.stats["prompt_tokens"] += prompt_tokens
        self.stats["completion_tokens"] += completion_tokens
        return {
            "content": response.choices[0].message.content or "",
            "finish_reason": response.choices[0].finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def token_cap(self, region_class: str, width: int, height: int) -> int:
        """
        max_tokens for one request, from how much text the crop could physically hold.

        With greedy decoding chandra occasionally falls into a repetition loop (JSON
        coordinates for illustrations, counting, rows of dots) that runs to the token limit;
        on the sample page nine such loops produced over half of all output tokens. A real
        transcription needs a few tokens per word, so capping at tokens_per_word x the
        number of words that fit leaves ample room while cutting loops short.
        """
        if region_class == "figure":
            return self.config.chandra_figure_max_tokens
        words = (width * height) / self.config.chandra_min_word_area
        return int(min(self.config.chandra_max_tokens, max(256, self.config.chandra_tokens_per_word * words)))

    async def ocr_region(self, crop_path: str, region_class: str, priority: Tuple, attempt: int = 0) -> Dict:
        """
        Transcribe one region, splitting tall crops into overlapping segments.

        Args:
            crop_path: Region crop on disk
            region_class: Layout class (text, title, figure, table); selects the prompt
            priority: Queue priority, lower runs first
            attempt: 0 for the first try. A retry (after output that wasn't a transcription) uses
                segments half as tall per attempt and a little sampling temperature, which breaks
                the repetition loops greedy decoding can fall into

        Returns:
            Dict with text, html, labels, image_descriptions, finish_reasons, prompt/completion tokens,
            segments and cached (True if every segment came from the cache)
        """
        from PIL import Image

        prompts = self.config.prompts
        prompt = prompts.get(region_class, prompts["text"])
        with Image.open(crop_path) as image:
            image.load()
            width, height = image.size
            if region_class == "figure":
                offsets = [(0, height)]
            else:
                offsets = split_offsets(height, width, self.config.split_max_ratio / 2**attempt, self.config.split_overlap_fraction)
            segments, caps = [], []
            for y1, y2 in offsets:
                caps.append(self.token_cap(region_class, width, y2 - y1))
                buffer = io.BytesIO()
                pad_to_aspect(image.crop((0, y1, width, y2)), self.config.max_aspect_ratio).save(buffer, format="PNG")
                segments.append(buffer.getvalue())

        temperature = self.config.chandra_retry_temperature if attempt else None
        futures = [self.submit(png, prompt, (*priority, i), cap, temperature) for i, (png, cap) in enumerate(zip(segments, caps))]
        results = await asyncio.gather(*futures)

        drop = () if region_class == "figure" else IMAGE_LABELS
        text, labels, descriptions = "", [], []
        for result in results:
            segment_text, segment_labels, segment_dropped = html_to_text(result["content"], drop)
            text = knit_texts(text, segment_text)
            labels.extend(segment_labels)
            descriptions.extend(segment_dropped)
        return {
            "text": text,
            "html": "\n".join(r["content"] for r in results),
            "labels": labels,
            "image_descriptions": descriptions,
            "finish_reasons": [r["finish_reason"] for r in results],
            "prompt_tokens": sum(r["prompt_tokens"] for r in results),
            "completion_tokens": sum(r["completion_tokens"] for r in results),
            "segments": len(results),
            "cached": all(r.get("cached") for r in results),
        }
