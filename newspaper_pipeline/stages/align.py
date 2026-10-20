"""
Stage 5: align chandra's transcription onto Tesseract's word boxes.

Chandra reads historical type far better than Tesseract, but only Tesseract says where
each word is. Both token streams are normalised and aligned monotonically:

1. Exact matches (difflib on normalised tokens) give anchors.
2. Gaps between anchors are aligned with a fuzzy Needleman-Wunsch pass scored by
   rapidfuzz similarity, so OCR misspellings ("Hotcl" vs "Hotel") still pair up.
3. Runs left unmatched (typically whole columns the engines read in a different order)
   are paired by shared vocabulary and aligned against each other.

Before that, Tesseract lines are split into cells and paired with chandra lines by text
similarity in any order (align_region), which handles tables: chandra reads them column by
column, Tesseract row by row.

Matched words carry chandra's text with Tesseract's box. Given the region crop, unmatched
chandra tokens next to matched ones get an estimated box (stages/estimate.py, match
"estimated"). The rest are attached to the line of their nearest matched neighbour so no text
is lost at line level; unmatched Tesseract words (often bleed-through noise) are dropped
unless an estimate reuses them.
"""

import difflib
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from newspaper_pipeline.stages.estimate import Box, Ink, _box, estimate_unmatched, repair_matches
from newspaper_pipeline.transforms import union_box

MAX_FUZZY_CELLS = 250_000  # cap on the NW matrix for one gap; larger gaps keep exact anchors only
_NON_ALNUM = re.compile(r"[^\w]+", re.UNICODE)


def normalise_token(token: str) -> str:
    """Lowercase, fold compatibility forms (long s, ligatures) and strip punctuation; pure punctuation keeps its symbols."""
    folded = unicodedata.normalize("NFKC", token).replace("ſ", "s").lower()
    core = _NON_ALNUM.sub("", folded).replace("_", "")
    return core or folded


def _fuzzy_pairs(left: Sequence[str], right: Sequence[str], min_similarity: float) -> List[Tuple[int, int, float]]:
    """Weighted monotonic alignment of two short token lists; returns (i, j, similarity) pairs."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cdist

    n, m = len(left), len(right)
    if n == 0 or m == 0 or n * m > MAX_FUZZY_CELLS:
        return []
    sim = cdist(list(left), list(right), scorer=fuzz.ratio, dtype=np.float32) / 100.0
    gain = sim - min_similarity  # only pairs above the threshold add value
    score = np.zeros((n + 1, m + 1), dtype=np.float32)
    for i in range(1, n + 1):
        row_prev, row = score[i - 1], score[i]
        g = gain[i - 1]
        for j in range(1, m + 1):
            diag = row_prev[j - 1] + g[j - 1]
            up = row_prev[j]
            left_ = row[j - 1]
            row[j] = diag if diag >= up and diag >= left_ else (up if up >= left_ else left_)

    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if score[i, j] == score[i - 1, j - 1] + gain[i - 1, j - 1] and gain[i - 1, j - 1] >= 0:
            pairs.append((i - 1, j - 1, float(sim[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif score[i, j] == score[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _monotonic_pairs(tess_tokens: Sequence[str], chandra_tokens: Sequence[str], min_similarity: float) -> List[Tuple[int, int, float]]:
    matcher = difflib.SequenceMatcher(None, list(tess_tokens), list(chandra_tokens), autojunk=False)
    pairs: List[Tuple[int, int, float]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend((i1 + k, j1 + k, 1.0) for k in range(i2 - i1))
        elif tag == "replace":
            pairs.extend((i1 + a, j1 + b, s) for a, b, s in _fuzzy_pairs(tess_tokens[i1:i2], chandra_tokens[j1:j2], min_similarity))
    return pairs


def _unmatched_runs(length: int, matched: set, min_run: int) -> List[Tuple[int, int]]:
    """Contiguous [start, end) runs of unmatched indices at least min_run long."""
    runs, start = [], None
    for k in range(length + 1):
        free = k < length and k not in matched
        if free and start is None:
            start = k
        elif not free and start is not None:
            if k - start >= min_run:
                runs.append((start, k))
            start = None
    return runs


def align_tokens(
    tess_tokens: Sequence[str],
    chandra_tokens: Sequence[str],
    min_similarity: float = 0.6,
    min_run: int = 3,
    min_shared: int = 3,
    max_rounds: int = 50,
) -> List[Tuple[int, int, float]]:
    """
    Align normalised Tesseract tokens to normalised chandra tokens.

    A monotonic pass handles the common case. Regions spanning several columns are often
    read in a different block order by the two engines, which a single monotonic pass can
    only follow for one ordering, so the remaining unmatched runs are then paired up (by
    shared vocabulary) and aligned monotonically against each other, repeatedly.

    Returns:
        (tesseract index, chandra index, similarity) for each matched pair, sorted by chandra index
    """
    pairs = _monotonic_pairs(tess_tokens, chandra_tokens, min_similarity)
    for _ in range(max_rounds):
        t_runs = _unmatched_runs(len(tess_tokens), {i for i, _, _ in pairs}, min_run)
        c_runs = _unmatched_runs(len(chandra_tokens), {j for _, j, _ in pairs}, min_run)
        best, best_shared = None, min_shared - 1
        for ts, te in t_runs:
            t_vocab = Counter(tess_tokens[ts:te])
            for cs, ce in c_runs:
                shared = sum((t_vocab & Counter(chandra_tokens[cs:ce])).values())
                if shared > best_shared:
                    best, best_shared = ((ts, te), (cs, ce)), shared
        if best is None:
            break
        (ts, te), (cs, ce) = best
        new = _monotonic_pairs(tess_tokens[ts:te], chandra_tokens[cs:ce], min_similarity)
        if not new:
            break
        pairs.extend((ts + a, cs + b, s) for a, b, s in new)
    return sorted(pairs, key=lambda pair: pair[1])


def tesseract_segments(words: List[Dict], gap_factor: float = 2.5) -> List[List[int]]:
    """
    Split Tesseract lines into cells: runs of words without a large horizontal gap.

    Tesseract reads a table row by row, so one of its "lines" can hold several cells
    ("Metchosin..... | Monday, 1 P.M. | Tuesday, 12 Noon."). A gap wider than gap_factor x
    the line's median word height separates cells. In ordinary prose no gap is that wide,
    so segments are just lines.

    Returns:
        Lists of word indices, in Tesseract line order and left to right within a line
    """
    by_line: Dict[int, List[int]] = {}
    for index, word in enumerate(words):
        by_line.setdefault(word["line"], []).append(index)
    segments: List[List[int]] = []
    for line in sorted(by_line):
        indices = sorted(by_line[line], key=lambda i: words[i]["x1"])
        height = float(np.median([words[i]["y2"] - words[i]["y1"] for i in indices]))
        current = [indices[0]]
        for prev, cur in zip(indices, indices[1:]):
            if words[cur]["x1"] - words[prev]["x2"] > gap_factor * max(height, 1.0):
                segments.append(current)
                current = [cur]
            else:
                current.append(cur)
        segments.append(current)
    return segments


def chandra_segments(text: str) -> Tuple[List[str], List[List[int]]]:
    """Tokens of the chandra text and their grouping into lines (and tab-separated table cells)."""
    tokens: List[str] = []
    segments: List[List[int]] = []
    for line in text.split("\n"):
        for cell in line.split("\t"):
            cell_tokens = cell.split()
            if cell_tokens:
                segments.append(list(range(len(tokens), len(tokens) + len(cell_tokens))))
                tokens.extend(cell_tokens)
    return tokens, segments


def _has_text(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


_LEADER_CHARS = re.compile(r"[.,:;'\"`\-_~|eEsSoO0]")


def is_leader_junk(token: str) -> bool:
    """Tesseract's reading of dot leaders ("Seseeeee", "eee", "......"): nothing left once leader-like characters go."""
    return not _LEADER_CHARS.sub("", token)


def _segment_box(words: List[Dict], indices: Sequence[int]) -> Tuple[float, float, float, float]:
    return union_box((words[i]["x1"], words[i]["y1"], words[i]["x2"], words[i]["y2"]) for i in indices)


def _segment_pairs(
    t_norm: Sequence[str],
    c_norm: Sequence[str],
    t_segments: List[List[int]],
    c_segments: List[List[int]],
    words: List[Dict],
    min_similarity: float,
    min_segment_score: float = 75.0,
    ambiguity_margin: float = 5.0,
    context_bonus: float = 15.0,
) -> List[Tuple[int, int, float]]:
    """
    Pair Tesseract cells with chandra lines by text similarity, in any order, and align the
    words inside each pair.

    - Unambiguous pairs go first: a pair whose score clearly beats every alternative for
      that Tesseract cell. Tables repeat cells ("of each month" under both ARRIVE and
      CLOSE), and those identical cells are only decided once their neighbours are placed:
      a chandra line whose neighbouring line sits in the same table column, directly above
      or below, gets a bonus.
    - Both sides stay available while they have unmatched tokens: a chandra line can span
      several Tesseract cells, and dot leaders can fuse two cells into one Tesseract segment.
    - Scores ignore punctuation-only tokens and Tesseract's readings of dot leaders.
    """
    from rapidfuzz import fuzz

    t_left = [[i for i in seg] for seg in t_segments]
    c_left = [[j for j in seg] for seg in c_segments]
    t_boxes = [_segment_box(words, seg) for seg in t_segments]
    paired_t_of_c: Dict[int, int] = {}
    blocked: set = set()

    def text_of_t(ti: int) -> str:
        return " ".join(t_norm[i] for i in t_left[ti] if _has_text(t_norm[i]) and not is_leader_junk(words[i]["text"]))

    def text_of_c(ci: int) -> str:
        return " ".join(c_norm[j] for j in c_left[ci] if _has_text(c_norm[j]))

    def text_score(ti: int, ci: int) -> float:
        if (ti, ci) in blocked:
            return 0.0
        a, b = text_of_t(ti), text_of_c(ci)
        if not a or not b:
            return 0.0
        if min(len(a), len(b)) < 8:
            return fuzz.ratio(a, b)
        return fuzz.partial_ratio(a, b) * (0.75 + 0.25 * min(len(a), len(b)) / max(len(a), len(b)))

    def context(ti: int, ci: int) -> float:
        """Bonus when the previous/next chandra line is placed directly above/below this Tesseract cell."""
        x1, y1, x2, y2 = t_boxes[ti]
        bonus = 0.0
        for neighbour, below in ((ci - 1, True), (ci + 1, False)):
            tj = paired_t_of_c.get(neighbour)
            if tj is None or tj == ti:
                continue
            nx1, ny1, nx2, ny2 = t_boxes[tj]
            x_overlap = min(x2, nx2) - max(x1, nx1)
            if x_overlap > 0.3 * min(x2 - x1, nx2 - nx1) and ((below and y1 >= ny1) or (not below and y1 <= ny1)):
                bonus = context_bonus
        return bonus

    scores = np.array([[text_score(ti, ci) for ci in range(len(c_segments))] for ti in range(len(t_segments))], dtype=float)
    pairs: List[Tuple[int, int, float]] = []
    while scores.size and scores.max() >= min_segment_score:
        # Prefer the best pair among Tesseract cells whose best chandra line is unambiguous
        best = None
        for ti in range(len(t_segments)):
            row = scores[ti]
            if row.max() < min_segment_score:
                continue
            order = np.argsort(row)[::-1]
            top, second = row[order[0]], (row[order[1]] if len(order) > 1 else 0.0)
            if top - second >= ambiguity_margin and (best is None or top > best[0]):
                best = (top, ti, int(order[0]))
        if best is None:
            # Only ambiguous pairs left: decide them by position relative to placed neighbours
            candidates = [
                (scores[ti, ci] + context(ti, ci), ti, ci)
                for ti in range(len(t_segments))
                for ci in range(len(c_segments))
                if scores[ti, ci] >= min_segment_score
            ]
            best = max(candidates)
        _, ti, ci = best
        local = _monotonic_pairs([t_norm[i] for i in t_left[ti]], [c_norm[j] for j in c_left[ci]], min_similarity)
        if not local:
            blocked.add((ti, ci))
            scores[ti, ci] = 0.0
            continue
        new_t = {t_left[ti][a] for a, _, _ in local}
        new_c = {c_left[ci][b] for _, b, _ in local}
        pairs.extend((t_left[ti][a], c_left[ci][b], similarity) for a, b, similarity in local)
        paired_t_of_c.setdefault(ci, ti)
        t_left[ti] = [i for i in t_left[ti] if i not in new_t]
        c_left[ci] = [j for j in c_left[ci] if j not in new_c]
        for other in range(len(c_segments)):
            scores[ti, other] = text_score(ti, other)
        for other in range(len(t_segments)):
            scores[other, ci] = text_score(other, ci)
    return pairs


def align_region(
    chandra_text: str, words: List[Dict], min_similarity: float = 0.6, noise_conf: float = 20.0, crop_path: Optional[str] = None
) -> Dict:
    """
    Combine chandra text and Tesseract words for one region.

    Tesseract lines are split into cells and paired with chandra lines by similarity in any
    order (so a table chandra reads column by column still lines up with Tesseract reading
    it row by row); words are aligned within each pair, and whatever is left over is aligned
    globally with align_tokens.

    Args:
        chandra_text: Plain text from chandra
        words: Tesseract words (text, conf, line, x1..y2 in crop pixels)
        min_similarity: Minimum rapidfuzz ratio (0-1) for a fuzzy match
        noise_conf: Unmatched Tesseract words below this confidence don't extend line boxes
        crop_path: The region crop Tesseract read; when given, unmatched tokens next to matched
            ones get estimated boxes

    Returns:
        Dict with:
            words: words with a box, in chandra order {text, chandra_index, tesseract_text, conf, line,
                similarity, match, x1..y2}; match is "exact", "fuzzy" or "estimated" (tesseract_text
                and conf are set when the estimate reused an unmatched Tesseract box, else "" / None)
            lines: {line, text, x1..y2, n_words} for every line (or table cell) that received chandra text
            score: fraction of chandra tokens with letters or digits placed on a Tesseract word
                (matched only, estimates not counted)
            n_chandra_tokens, n_tesseract_words, n_matched, n_estimated
    """
    chandra_tokens, c_segments = chandra_segments(chandra_text)
    result: Dict = {
        "words": [],
        "lines": [],
        "score": 0.0,
        "n_chandra_tokens": len(chandra_tokens),
        "n_tesseract_words": len(words),
        "n_matched": 0,
        "n_estimated": 0,
    }
    if not chandra_tokens or not words:
        return result

    t_segments = tesseract_segments(words)
    segment_of = {i: n for n, seg in enumerate(t_segments) for i in seg}
    t_norm = [normalise_token(w["text"]) for w in words]
    c_norm = [normalise_token(t) for t in chandra_tokens]

    pairs = _segment_pairs(t_norm, c_norm, t_segments, c_segments, words, min_similarity)
    done_t, done_c = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
    t_rest = [i for seg in t_segments for i in seg if i not in done_t]
    c_rest = [j for j in range(len(chandra_tokens)) if j not in done_c]
    if t_rest and c_rest:
        leftover = align_tokens([t_norm[i] for i in t_rest], [c_norm[j] for j in c_rest], min_similarity)
        pairs.extend((t_rest[a], c_rest[b], similarity) for a, b, similarity in leftover)
    pairs.sort(key=lambda pair: pair[1])

    # With the crop, fix matched boxes that don't cover their word (see estimate.repair_matches)
    gray = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE) if crop_path is not None else None
    ink = Ink(gray) if gray is not None else None
    parts: Dict[int, List[Box]] = {}
    split: Dict[int, Tuple[Box, int]] = {}
    ignore: Set[int] = set()
    if ink is not None and pairs:
        repairs = repair_matches(chandra_tokens, c_segments, words, pairs, ink, normalise_token)
        pairs, parts, split, ignore = repairs.pairs, repairs.parts, repairs.split, repairs.ignore

    def entry(cj: int, box: Box, line: int, match: str, tesseract_text: str, conf: Optional[float], similarity: float) -> Dict:
        return {
            "text": chandra_tokens[cj],
            "chandra_index": cj,
            "tesseract_text": tesseract_text,
            "conf": conf,
            "line": line,
            "similarity": similarity,
            "match": match,
            "x1": box[0],
            "y1": box[1],
            "x2": box[2],
            "y2": box[3],
        }

    token_line: List[int] = [-1] * len(chandra_tokens)
    for ti, cj, similarity in pairs:
        word = words[ti]
        token_line[cj] = segment_of[ti]
        box = parts[cj][0] if cj in parts else split[cj][0] if cj in split else _box(word)
        result["words"].append(
            entry(cj, box, segment_of[ti], "exact" if similarity == 1.0 else "fuzzy", word["text"], word["conf"], round(similarity, 3))
        )
        if cj in parts:
            result["words"][-1]["parts"] = parts[cj]  # printed on two lines: head, then tail
    result["n_matched"] = len(pairs)
    # Score over tokens with letters or digits: braces and dot leaders carry no position
    textual = [j for j, t in enumerate(chandra_tokens) if _has_text(t)]
    matched_c = {cj for _, cj, _ in pairs}
    result["score"] = round(sum(1 for j in textual if j in matched_c) / max(len(textual), 1), 4)
    if not pairs:
        return result

    estimated_boxes: Dict[int, Box] = {}
    for cj, (box, ti) in split.items():
        if cj not in matched_c:  # the other words of a fused Tesseract box
            token_line[cj] = segment_of[ti]
            estimated_boxes[cj] = box
            result["words"].append(entry(cj, box, segment_of[ti], "estimated", words[ti]["text"], words[ti]["conf"], 0.0))
    if ink is not None:
        for cj, (box, line_word, reused) in estimate_unmatched(chandra_tokens, c_segments, words, pairs, ink, ignore).items():
            if cj in estimated_boxes or not _has_text(chandra_tokens[cj]):
                continue  # braces, rules and leaders: the hold-out test only covered real words
            token_line[cj] = segment_of[line_word]
            estimated_boxes[cj] = box
            reused_text, reused_conf = (words[reused]["text"], words[reused]["conf"]) if reused is not None else ("", None)
            result["words"].append(entry(cj, box, segment_of[line_word], "estimated", reused_text, reused_conf, 0.0))
    result["words"].sort(key=lambda w: w["chandra_index"])
    result["n_estimated"] = len(estimated_boxes)

    # A line annotation is one chandra line (or table cell) within one Tesseract segment.
    # Unmatched chandra tokens join the line of the previous matched token in the same chandra
    # line (or the next one); chandra lines with no match at all stay region-only.
    c_segment_of = {j: n for n, seg in enumerate(c_segments) for j in seg}
    token_key: List[Optional[Tuple[int, int]]] = [
        (token_line[j], c_segment_of[j]) if token_line[j] >= 0 else None for j in range(len(chandra_tokens))
    ]
    for seg in c_segments:
        known = [token_key[j] for j in seg if token_key[j] is not None]
        if not known:
            continue
        last = known[0]
        for j in seg:
            if token_key[j] is None:
                token_key[j] = last
            else:
                last = token_key[j]

    matched_by_key: Dict[Tuple[int, int], List[int]] = {}
    for ti, cj, _ in pairs:
        matched_by_key.setdefault((segment_of[ti], c_segment_of[cj]), []).append(ti)
    shared_segments = {t for t, _ in matched_by_key if sum(1 for k in matched_by_key if k[0] == t) > 1}

    line_tokens: Dict[Tuple[int, int], List[str]] = {}
    estimated_by_key: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
    for j, (token, key) in enumerate(zip(chandra_tokens, token_key)):
        if key is not None:
            line_tokens.setdefault(key, []).append(token)
            if j in estimated_boxes:
                estimated_by_key.setdefault(key, []).append(estimated_boxes[j])
    matched_tess = {ti for ti, _, _ in pairs}
    for number, (key, tokens) in enumerate(sorted(line_tokens.items())):
        t_seg, _ = key
        if t_seg in shared_segments:
            # Several chandra lines share this Tesseract segment (cells fused by dot leaders):
            # box only the words matched to this chandra line
            members = matched_by_key.get(key, [])
        else:
            members = [i for i in t_segments[t_seg] if i in matched_tess or words[i]["conf"] >= noise_conf]
        if not members:
            continue
        boxes = [(words[i]["x1"], words[i]["y1"], words[i]["x2"], words[i]["y2"]) for i in members] + estimated_by_key.get(key, [])
        x1, y1, x2, y2 = union_box(boxes)
        result["lines"].append(
            {"line": number, "text": " ".join(tokens), "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "n_words": len(tokens)}
        )
    return result
