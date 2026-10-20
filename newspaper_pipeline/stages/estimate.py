"""
Stage 5b: estimate word boxes for chandra tokens that Tesseract's words didn't match.

Alignment leaves some chandra tokens without a Tesseract box: Tesseract misread them beyond
fuzzy matching, split or merged them, or missed them. Their position is constrained by the
matched words around them, so each run of unmatched tokens inside a chandra line is placed
from its neighbours:

- gap:   between two matched words on the same printed line
- start: before the first matched word of a line
- end:   after the last matched word of a line
- break: between matched words that sit on two different printed lines

If Tesseract left exactly as many unmatched boxes in that spot as there are tokens, those
boxes are used ("misread" words). Otherwise the words are cut from the ink of the region
crop: the row is described by the matched words near the run (top/bottom percentiles,
character width, word gap), its ink columns are cut at the widest gaps, and each word is
trimmed to its ink. A run is skipped whenever the evidence doesn't fit: Tesseract saw a
different number of words there, the ink doesn't split into the right number of words, or
a word's width doesn't fit its characters. Whole lines with no matched word are skipped.

Hold-out test on the 6 sample pages not used for tuning: matched words were hidden (singly
and in runs of 2-3) and re-estimated, counting an estimate correct at IoU >= 0.5 with
Tesseract's box. With the hidden words' Tesseract boxes removed, 98.65% of 44,065 estimates
were correct (79% of hidden words estimated); with them left in as unmatched boxes, 99.34%.
Against exact, confident references every case scored 97.9% or more.
"""

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]
Span = Tuple[int, int]

_NARROW = set(".,;:'\"!|-()")


def text_width(text: str) -> float:
    """Width of a token in average characters: capitals and wide letters count more, punctuation less."""
    return sum(0.4 if c in _NARROW else 1.35 if c.isupper() or c in "mwMW" else 1.0 for c in text) or 1.0


class Ink:
    """
    Dark pixels of a greyscale region crop (Otsu), as connected pieces.

    Rows are read through band(), which keeps only pieces of letter size for that row: dots,
    dashes, leaders and rules are much shorter than a line, and brackets, vertical rules and
    drop capitals much taller. None of them make a word on their own.
    """

    def __init__(self, gray: np.ndarray, min_area: int = 8):
        _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, self.labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        self.heights = stats[:, cv2.CC_STAT_HEIGHT]
        self.keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        self.keep[0] = False  # background
        self.height, self.width = gray.shape

    def band(self, y1: int, y2: int, x1: int, x2: int, line_height: float) -> np.ndarray:
        """Ink in [y1, y2) x [x1, x2) from pieces between 0.3 and 2.2 line heights tall."""
        labels = self.labels[max(0, y1):y2, max(0, x1):x2]
        heights = self.heights[labels]
        letters: np.ndarray = self.keep[labels] & (heights >= 0.3 * line_height) & (heights <= 2.2 * line_height)
        return letters


def _runs(flags: np.ndarray) -> List[Span]:
    """[start, end) runs of True."""
    padded = np.concatenate(([False], flags, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])]


class Row:
    """Geometry of one printed line near a run, from the matched words on its Tesseract line."""

    def __init__(self, boxes: List[Box], texts: List[str], near_x: Optional[float] = None, nearest: int = 8):
        if near_x is not None and len(boxes) > nearest:
            # Lines drift slightly over their length: describe the row near the run
            order = sorted(range(len(boxes)), key=lambda i: abs((boxes[i][0] + boxes[i][2]) / 2 - near_x))[:nearest]
            boxes, texts = [boxes[i] for i in order], [texts[i] for i in order]
        # Percentiles, not min/max: one bad Tesseract box can span two lines
        self.y1 = int(np.percentile([b[1] for b in boxes], 25))
        self.y2 = int(np.percentile([b[3] for b in boxes], 75))
        self.h = float(np.median([b[3] - b[1] for b in boxes]))
        self.char_w = float(np.median([(b[2] - b[0]) / text_width(t) for b, t in zip(boxes, texts)]))
        ordered = sorted(boxes)
        gaps = [b[0] - a[2] for a, b in zip(ordered, ordered[1:]) if b[0] > a[2]]
        self.gap = float(np.median(gaps)) if gaps else 0.35 * self.h

    @property
    def split(self) -> float:
        """Gaps at least this wide separate words."""
        return max(0.12 * self.h, 0.45 * self.gap)

    @property
    def stop(self) -> float:
        """A gap this wide ends the line (column gutter or the end of the text)."""
        return max(2.5 * self.gap, 1.2 * self.h)


Anchor = Tuple[Box, Row]


def _column_runs(ink: Ink, row: Row, x1: int, x2: int) -> List[Span]:
    x1, x2 = max(0, int(x1)), min(ink.width, int(x2))
    if x2 <= x1:
        return []
    band = ink.band(row.y1, row.y2, x1, x2, row.h)
    return [(x1 + a, x1 + b) for a, b in _runs(band.any(axis=0))]


def _merge(runs: List[Span], gap: float) -> List[Span]:
    words: List[List[int]] = []
    for a, b in runs:
        if words and a - words[-1][1] < gap:
            words[-1][1] = b
        else:
            words.append([a, b])
    return [(a, b) for a, b in words]


def _scan(ink: Ink, row: Row, start: int, direction: int) -> List[Span]:
    """Words on the row from x=start outward (+1 right, -1 left), nearest first, up to a stop gap."""
    if direction > 0:
        words = _merge(_column_runs(ink, row, start, ink.width), row.split)
    else:
        words = _merge(_column_runs(ink, row, 0, start), row.split)[::-1]
    found: List[Span] = []
    edge = start
    for a, b in words:
        if ((a - edge) if direction > 0 else (edge - b)) > row.stop:
            break
        found.append((a, b))
        edge = b if direction > 0 else a
    return found


def _split_gap(ink: Ink, row: Row, x1: int, x2: int, n: int) -> Optional[List[Span]]:
    """n words between x1 and x2: the ink there, cut at its n-1 widest gaps."""
    runs = _column_runs(ink, row, x1, x2)
    if not runs:
        return None
    gaps = sorted(((runs[i + 1][0] - runs[i][1], i) for i in range(len(runs) - 1)), reverse=True)
    if len(gaps) < n - 1:
        return None
    if n > 1 and gaps[n - 2][0] < row.split * 0.6:
        return None  # would have to cut inside a word
    spans, start = [], 0
    for i in sorted(i for _, i in gaps[: n - 1]) + [len(runs) - 1]:
        spans.append((runs[start][0], runs[i][1]))
        start = i + 1
    return spans


def _outward(ink: Ink, row: Row, start: int, direction: int, texts: Sequence[str]) -> Optional[List[Span]]:
    """
    n words next to an anchor, going outward from x=start (+1 right, -1 left).

    Takes the ink runs out to the first stop gap and keeps as many of them as gives the width
    closest to the expected width (characters x the row's character width), never ending
    inside a word; then cuts that span at its n-1 widest gaps.
    """
    if direction > 0:
        runs = _column_runs(ink, row, start, ink.width)
    else:
        runs = _column_runs(ink, row, 0, start)[::-1]
    chain: List[Span] = []
    edge = start
    for a, b in runs:
        if ((a - edge) if direction > 0 else (edge - b)) > row.stop:
            break
        chain.append((a, b))
        edge = b if direction > 0 else a
    if not chain:
        return None
    n = len(texts)
    expected = sum(text_width(t) for t in texts) * row.char_w + (n - 1) * row.gap
    best: Optional[Tuple[float, int, int]] = None
    for m in range(1, len(chain) + 1):
        if m < len(chain):
            after = (chain[m][0] - chain[m - 1][1]) if direction > 0 else (chain[m - 1][0] - chain[m][1])
            if after < max(row.split * 0.6, 0.6 * row.gap):
                continue  # would end inside a word
        lo, hi = min(chain[0][0], chain[m - 1][0]), max(chain[0][1], chain[m - 1][1])
        error = abs((hi - lo) - expected)
        if best is None or error < best[0]:
            best = (error, lo, hi)
    if best is None:
        return None
    return _split_gap(ink, row, best[1], best[2], n)


def _vertical(ink: Ink, row: Row, a: int, b: int) -> Optional[Span]:
    """Top and bottom of the ink in columns [a, b) near the row band."""
    extra = int(round(0.15 * row.h))
    top, bottom = max(0, row.y1 - extra), min(ink.height, row.y2 + extra)
    rows = ink.band(top, bottom, a, b, row.h).any(axis=1)
    keep = [(s + top, e + top) for s, e in _runs(rows) if s + top < row.y2 and e + top > row.y1]
    if not keep:
        return None
    return keep[0][0], keep[-1][1]


def _boxes(ink: Ink, row: Row, spans: List[Span]) -> Optional[List[Box]]:
    out = []
    for a, b in spans:
        v = _vertical(ink, row, a, b)
        if v is None:
            return None
        out.append((int(a), int(v[0]), int(b), int(v[1])))
    return out


def _plausible(spans: Sequence[Span], texts: Sequence[str], row: Row) -> bool:
    # Outside this range, hold-out estimates were wrong far more often (30-70% vs 98-99% inside)
    return all(0.55 <= (b - a) / max(text_width(t) * row.char_w, 1.0) <= 1.4 for (a, b), t in zip(spans, texts))


def _on_row(boxes: Sequence[Box], row: Row) -> bool:
    """Each box sits on the row and is at most 1.5 lines tall (Tesseract sometimes boxes two lines as one word)."""
    return all(b[3] - b[1] <= 1.5 * row.h and row.y1 <= (b[1] + b[3]) / 2 <= row.y2 for b in boxes)


def _box(word: Dict) -> Box:
    return (word["x1"], word["y1"], word["x2"], word["y2"])


def _leftovers_in(words: List[Dict], leftover: Sequence[int], row: Row, x1: float, x2: float) -> List[int]:
    found = [
        i
        for i in leftover
        if x1 <= (words[i]["x1"] + words[i]["x2"]) / 2 <= x2 and row.y1 <= (words[i]["y1"] + words[i]["y2"]) / 2 <= row.y2
    ]
    return sorted(found, key=lambda i: words[i]["x1"])


Estimate = Tuple[Box, str, Optional[int]]  # box, which anchor's row it is on ("left"/"right"), reused Tesseract word


def estimate_run(
    texts: List[str],
    left: Optional[Anchor],
    right: Optional[Anchor],
    ink: Ink,
    words: List[Dict],
    leftover: Sequence[int],
    margin: Optional[float] = None,
) -> Tuple[Optional[List[Estimate]], str]:
    """
    Boxes for one run of consecutive unmatched tokens in a chandra line.

    Args:
        texts: The run's chandra tokens
        left, right: (box, Row) of the matched token before / after the run, if any
        ink: Ink of the region crop
        words: All Tesseract words of the region
        leftover: Indices of Tesseract words that matched nothing
        margin: Left edge of the column (where matched lines start); a line start can't begin left of it

    Returns:
        (one estimate per token, or None to skip the run; the case: gap, start, end or break)
    """
    n = len(texts)
    for neighbour in (left, right):
        if neighbour is not None and not neighbour[1].y1 <= (neighbour[0][1] + neighbour[0][3]) / 2 <= neighbour[1].y2:
            return None, "inconsistent"  # the anchor doesn't sit on its own row
    if left and right:
        (lb, lrow), (rb, rrow) = left, right
        if min(lb[3], rb[3]) - max(lb[1], rb[1]) > 0.5 * min(lb[3] - lb[1], rb[3] - rb[1]) and rb[0] > lb[2]:
            found = _leftovers_in(words, leftover, lrow, lb[2], rb[0])
            if found:
                found_spans = [(words[i]["x1"], words[i]["x2"]) for i in found]
                if len(found) == n and _plausible(found_spans, texts, lrow) and _on_row([_box(words[i]) for i in found], lrow):
                    return [(_box(words[i]), "left", i) for i in found], "gap"
                return None, "gap"  # Tesseract saw something else here: don't guess
            spans = _split_gap(ink, lrow, lb[2] + 1, rb[0], n)
            boxes = _boxes(ink, lrow, spans) if spans and _plausible(spans, texts, lrow) else None
            if not boxes or not _on_row(boxes, lrow):
                return None, "gap"
            return [(b, "left", None) for b in boxes], "gap"
        tail = _scan(ink, lrow, lb[2] + 1, +1)
        head = _scan(ink, rrow, rb[0], -1)[::-1]
        if len(tail) + len(head) != n:
            return None, "break"
        if not _plausible(tail, texts[: len(tail)], lrow) or not _plausible(head, texts[len(tail):], rrow):
            return None, "break"
        tail_boxes, head_boxes = _boxes(ink, lrow, tail), _boxes(ink, rrow, head)
        if tail_boxes is None or head_boxes is None or not _on_row(tail_boxes, lrow) or not _on_row(head_boxes, rrow):
            return None, "break"
        both: List[Estimate] = [(b, "left", None) for b in tail_boxes]
        both += [(b, "right", None) for b in head_boxes]
        return both, "break"

    if left is not None:
        (anchor, row), direction, side, case = left, +1, "left", "end"
    else:
        assert right is not None
        (anchor, row), direction, side, case = right, -1, "right", "start"
    edge = anchor[2] if direction > 0 else anchor[0]
    found = _leftovers_in(words, leftover, row, edge, ink.width) if direction > 0 else _leftovers_in(words, leftover, row, 0, edge)[::-1]
    chain: List[int] = []
    for i in found:
        if ((words[i]["x1"] - edge) if direction > 0 else (edge - words[i]["x2"])) > row.stop:
            break
        chain.append(i)
        edge = words[i]["x2"] if direction > 0 else words[i]["x1"]
    if chain:
        if len(chain) >= n:
            chain = chain[:n] if direction > 0 else chain[:n][::-1]
            if _plausible([(words[i]["x1"], words[i]["x2"]) for i in chain], texts, row) and _on_row([_box(words[i]) for i in chain], row):
                return [(_box(words[i]), side, i) for i in chain], case
        return None, case  # Tesseract saw something else here: don't guess
    spans = _outward(ink, row, anchor[2] + 1 if direction > 0 else anchor[0], direction, texts)
    if spans is None or len(spans) != n or not _plausible(spans, texts, row):
        return None, case
    if direction < 0 and margin is not None and spans[0][0] < margin - 0.6 * row.h:
        return None, case  # starts left of the column: a bracket, rule or the column next door
    boxes = _boxes(ink, row, spans)
    if not boxes or not _on_row(boxes, row):
        return None, case
    return [(b, side, None) for b in boxes], case


def estimate_unmatched(
    tokens: List[str],
    c_segments: List[List[int]],
    words: List[Dict],
    pairs: Sequence[Tuple[int, int, float]],
    ink: Ink,
    ignore: Optional[Set[int]] = None,
) -> Dict[int, Tuple[Box, int, Optional[int]]]:
    """
    Estimate boxes for every unmatched token that has a matched neighbour in its chandra line.

    Args:
        tokens: chandra tokens of the region
        c_segments: Token indices per chandra line (or table cell)
        words: Tesseract words of the region
        pairs: (Tesseract index, chandra index, similarity) from the alignment
        ink: Ink of the region crop
        ignore: Tesseract words not to reuse (already used by repair_matches, or rejected by it)

    Returns:
        chandra index -> (box, Tesseract word whose line the token is on, reused Tesseract word or None)
    """
    matched = {cj: ti for ti, cj, _ in pairs}
    used = set(matched.values()) | (ignore or set())
    leftover = [i for i in range(len(words)) if i not in used]
    by_line: Dict[int, List[int]] = {}
    for cj, ti in matched.items():
        by_line.setdefault(words[ti]["line"], []).append(cj)

    def anchor(cj: int) -> Anchor:
        ti = matched[cj]
        same = by_line[words[ti]["line"]]
        row = Row([_box(words[matched[c]]) for c in same], [tokens[c] for c in same], near_x=(words[ti]["x1"] + words[ti]["x2"]) / 2)
        return _box(words[ti]), row

    # Left edge of the column: where lines whose first token matched start
    starts = [words[matched[seg[0]]]["x1"] for seg in c_segments if seg and seg[0] in matched]
    margin = float(np.percentile(starts, 10)) if len(starts) >= 3 else None

    estimates: Dict[int, Tuple[Box, int, Optional[int]]] = {}
    for seg in c_segments:
        if not any(j in matched for j in seg):
            continue
        p = 0
        while p < len(seg):
            if seg[p] in matched:
                p += 1
                continue
            q = p
            while q < len(seg) and seg[q] not in matched:
                q += 1
            left_j = seg[p - 1] if p > 0 else None
            right_j = seg[q] if q < len(seg) else None
            run = seg[p:q]
            result, _ = estimate_run(
                [tokens[j] for j in run],
                anchor(left_j) if left_j is not None else None,
                anchor(right_j) if right_j is not None else None,
                ink,
                words,
                leftover,
                margin,
            )
            if result:
                for j, (box, side, reused) in zip(run, result):
                    neighbour = left_j if side == "left" else right_j
                    estimates[j] = (box, matched[neighbour], reused)  # type: ignore[index]
            p = q
    return estimates


HYPHENS = ("-", "\u00ac", "\u2010", "\u2011")


class Repairs:
    """What repair_matches changed: see there."""

    def __init__(self) -> None:
        self.pairs: List[Tuple[int, int, float]] = []
        self.parts: Dict[int, List[Box]] = {}  # chandra index -> boxes of a word printed on two lines
        self.split: Dict[int, Tuple[Box, int]] = {}  # chandra index -> (part of a fused Tesseract box, that box)
        self.ignore: Set[int] = set()  # Tesseract words that are now taken or were rejected


def repair_matches(
    tokens: List[str],
    c_segments: List[List[int]],
    words: List[Dict],
    pairs: Sequence[Tuple[int, int, float]],
    ink: Ink,
    normalise: Callable[[str], str],
) -> Repairs:
    """
    Fix matched words whose Tesseract box doesn't cover the word.

    - Hyphenated across lines: chandra joins "cap-" + "tain" into "captain", which matches only
      one half. The other half (an unmatched Tesseract word ending the previous line, or
      starting the next) is added, and the word gets both boxes.
    - Fused: Tesseract boxed two words as one ("wellknown"). The box is cut at the ink gap (or
      by character widths when there is none) and each word gets its part.
    - Implausible width: a box far narrower or wider than the word's characters (Tesseract boxed
      only "," of "B," or "31.-The" for "The") is dropped, so the word goes to estimation.
    - Blank: a box that is under 3% ink holds no word; it is dropped too.

    Args:
        normalise: Token normalisation used by the alignment
    """
    from rapidfuzz import fuzz

    out = Repairs()
    matched = {cj: ti for ti, cj, _ in pairs}
    used = set(matched.values())
    lines: Dict[int, List[int]] = {}
    for i, word in enumerate(words):
        lines.setdefault(word["line"], []).append(i)
    order = sorted(lines)
    first = {line: min(ix, key=lambda i: words[i]["x1"]) for line, ix in lines.items()}
    last = {line: max(ix, key=lambda i: words[i]["x2"]) for line, ix in lines.items()}
    by_line: Dict[int, List[int]] = {}
    for cj, ti in matched.items():
        by_line.setdefault(words[ti]["line"], []).append(cj)
    segment_of = {j: n for n, seg in enumerate(c_segments) for j in seg}

    def row(ti: int) -> Optional[Row]:
        same = by_line.get(words[ti]["line"], [])
        if len(same) < 3:
            return None
        return Row([_box(words[matched[c]]) for c in same], [tokens[c] for c in same], near_x=(words[ti]["x1"] + words[ti]["x2"]) / 2)

    def neighbour_line(line: int, step: int) -> Optional[int]:
        k = order.index(line) + step
        return order[k] if 0 <= k < len(order) else None

    demoted: Set[int] = set()
    for ti, cj, similarity in pairs:
        t, w = normalise(tokens[cj]), normalise(words[ti]["text"])
        line = words[ti]["line"]
        if len(t) >= len(w) + 2 and _has_letters(t):
            # The matched Tesseract word is one half of a word hyphenated across lines
            other: Optional[int] = None
            if first[line] == ti and (prev := neighbour_line(line, -1)) is not None:
                u = last[prev]
                if u not in used and words[u]["text"].rstrip().endswith(HYPHENS):
                    head = normalise(words[u]["text"].rstrip().rstrip("".join(HYPHENS)))
                    # The layout (hyphen ending one line, tail starting the next) is strong evidence on its
                    # own: a misread head ("Theoasp-" for "cap-") only has to have about the right length
                    fits = fuzz.ratio(head + w, t) >= 85 or (t.endswith(w) and abs(len(head) + len(w) - len(t)) <= 3)
                    if head and abs(len(head) + len(w) - len(t)) <= 3 and fits:
                        other, boxes = u, [_box(words[u]), _box(words[ti])]
            if other is None and last[line] == ti and words[ti]["text"].rstrip().endswith(HYPHENS) and (nxt := neighbour_line(line, +1)) is not None:
                u = first[nxt]
                tail = normalise(words[u]["text"])
                fits = fuzz.ratio(w + tail, t) >= 85 or (t.startswith(w) and abs(len(w) + len(tail) - len(t)) <= 3)
                if u not in used and tail and abs(len(w) + len(tail) - len(t)) <= 3 and fits:
                    other, boxes = u, [_box(words[ti]), _box(words[u])]
            if other is not None:
                out.parts[cj] = boxes
                used.add(other)
                out.pairs.append((ti, cj, similarity))
                continue
            starts_line, ends_line = first[line] == ti and t.endswith(w), last[line] == ti and t.startswith(w)
            if w and (starts_line or ends_line):
                demoted.add(ti)  # half of a hyphenated word whose other half wasn't found: the box covers too little
                continue
        # One Tesseract box holding this token and unmatched neighbours in the same chandra line
        fused = None
        seg = c_segments[segment_of[cj]]
        k = seg.index(cj)
        for a, b in ((k - 1, k + 1), (k, k + 2), (k - 2, k + 1), (k - 1, k + 2), (k, k + 3)):
            group = seg[max(a, 0):b] if a >= 0 and b <= len(seg) else []
            if len(group) < 2 or any(j != cj and (j in matched or not _has_letters(normalise(tokens[j]))) for j in group):
                continue
            joined = "".join(normalise(tokens[j]) for j in group)
            # the box text is explained better by the words together than by this word alone ("ina" = "in a")
            if len(w) > len(t) and fuzz.ratio(w, joined) >= 85 and fuzz.ratio(w, joined) > fuzz.ratio(w, t):
                fused = group
                break
        if fused:
            box = _box(words[ti])
            r = row(ti)
            texts = [tokens[j] for j in fused]
            spans = None
            if r is not None:
                spans = _split_gap(ink, r, box[0], box[2], len(fused))
                if spans is not None and not _plausible(spans, texts, r):
                    spans = None
            if spans is None:  # no clear gap: share the box by character widths
                widths = [text_width(t_) for t_ in texts]
                edges = np.concatenate(([0.0], np.cumsum(widths))) / sum(widths) * (box[2] - box[0]) + box[0]
                spans = [(int(round(edges[i])), int(round(edges[i + 1]))) for i in range(len(fused))]
            for j, (x1, x2) in zip(fused, spans):
                out.split[j] = ((x1, box[1], x2, box[3]), ti)
            out.pairs.append((ti, cj, similarity))
            continue
        # A box on (almost) blank paper: printed words are 15-30% ink
        x1, y1, x2, y2 = _box(words[ti])
        if x2 > x1 and y2 > y1 and ink.band(y1, y2, x1, x2, max(y2 - y1, 1)).mean() < 0.03:
            demoted.add(ti)
            continue
        # A box that can't hold the word's characters
        r = row(ti)
        if r is not None and _has_letters(t):
            tw = text_width(tokens[cj])
            ratio = (words[ti]["x2"] - words[ti]["x1"]) / max(tw * r.char_w, 1.0)
            if (tw >= 3 and not 0.5 <= ratio <= 2.0) or (tw >= 1.7 and ratio < 0.35):
                demoted.add(ti)
                continue
        out.pairs.append((ti, cj, similarity))
    out.ignore = (used - {ti for ti, _, _ in out.pairs}) | demoted
    return out


def _has_letters(text: str) -> bool:
    return any(c.isalnum() for c in text)
