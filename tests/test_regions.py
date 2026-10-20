import cv2
import numpy as np
import pandas as pd

from newspaper_pipeline.stages.regions import (
    INK_SCALE,
    build_regions,
    ink_mask,
    recover_missed_text,
    remove_nested,
    split_overlaps,
    xy_cut_order,
)


def box(cls, x1, y1, x2, y2, conf=0.9):
    return {"class": cls, "confidence": conf, "bbox": (x1, y1, x2, y2), "recovered": False}


def text_lines(page, x1, y1, x2, y2, line=34):
    """Draw rows of word-like bars, like a column of body text."""
    rng = np.random.default_rng(x1 + y1)
    for y in range(y1 + 6, y2 - 14, line):
        x = x1 + 4
        while x < x2 - 60:
            w = int(rng.integers(25, 90))
            cv2.rectangle(page, (x, y), (min(x + w, x2 - 4), y + 14), 20, -1)
            x += w + int(rng.integers(12, 20))
    return page


def overlap_area(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def test_remove_nested_drops_contained_and_duplicate_boxes():
    boxes = [
        box("text", 0, 0, 1000, 1000),
        box("title", 100, 100, 600, 200),  # inside the text box
        box("text", 2000, 0, 3000, 500, conf=0.4),
        box("text", 2005, 5, 3000, 500, conf=0.8),  # near duplicate, more confident
        box("figure", 200, 400, 500, 700),  # figures are kept even inside text
    ]
    kept = remove_nested(boxes)
    assert [(b["class"], b["confidence"]) for b in kept] == [("text", 0.9), ("text", 0.8), ("figure", 0.9)]


def test_split_overlaps_cuts_in_the_whitespace_between_lines():
    page = np.full((1200, 1000), 240, np.uint8)
    text_lines(page, 0, 0, 1000, 400)
    cv2.rectangle(page, (0, 430), (1000, 440), 240, -1)  # make sure there is a clear gap
    text_lines(page, 0, 460, 1000, 900)
    ink = ink_mask(page)
    # YOLO-style boxes overlapping across the gap between two paragraphs
    upper, lower = box("text", 0, 0, 1000, 520), box("text", 0, 380, 1000, 900)
    a, b = split_overlaps([upper, lower], ink)
    assert overlap_area(a["bbox"], b["bbox"]) == 0
    cut = a["bbox"][3]
    assert 395 <= cut <= 470  # between the paragraphs, not through a line of text


def test_split_overlaps_side_by_side_boxes_cut_vertically():
    ink = np.zeros((300, 300), np.uint8)
    left, right = box("text", 0, 0, 620, 1000), box("text", 580, 0, 1200, 1000)
    a, b = split_overlaps([left, right], ink)
    assert overlap_area(a["bbox"], b["bbox"]) == 0 and a["bbox"][2] == b["bbox"][0]


def test_figures_may_overlap_text():
    ink = np.zeros((300, 300), np.uint8)
    text, figure = box("text", 0, 0, 1000, 1000), box("figure", 500, 500, 900, 900)
    a, b = split_overlaps([text, figure], ink)
    assert a["bbox"] == (0, 0, 1000, 1000) and b["bbox"] == (500, 500, 900, 900)


def test_xy_cut_reads_masthead_then_columns_with_spanning_heading():
    #  [ masthead                         ]
    #  [col1 a]  [ heading over 2 and 3   ]
    #  [col1 b]  [col2 a]  [col3 a]
    #            [col2 b]  [col3 b]
    boxes = [
        box("text", 1000, 900, 1900, 1500),  # col2 b
        box("title", 0, 0, 3000, 300),  # masthead
        box("text", 0, 400, 900, 1000),  # col1 a
        box("title", 1000, 400, 2900, 500),  # heading over 2-3
        box("text", 2000, 550, 2900, 880),  # col3 a
        box("text", 0, 1050, 900, 1500),  # col1 b
        box("text", 1000, 550, 1900, 880),  # col2 a
        box("text", 2000, 900, 2900, 1500),  # col3 b
    ]
    order = [i for i, _, _ in xy_cut_order(boxes, x_tolerance=10, y_tolerance=2)]
    assert order == [1, 2, 5, 3, 6, 0, 4, 7]


def test_recover_missed_text_finds_uncovered_text_but_not_rules():
    page = np.full((2000, 3000), 240, np.uint8)
    text_lines(page, 100, 100, 1100, 900)  # detected by "YOLO"
    text_lines(page, 1500, 1200, 2500, 1500)  # missed by "YOLO"
    cv2.rectangle(page, (100, 1800), (2900, 1812), 15, -1)  # a printed rule
    cv2.line(page, (1300, 100), (1300, 1700), 15, 6)  # a column separator
    ink = ink_mask(page)
    found = recover_missed_text([box("text", 100, 100, 1100, 900)], ink, (3000, 2000))
    assert len(found) == 1
    x1, y1, x2, y2 = found[0]["bbox"]
    assert x1 <= 1510 and y1 <= 1210 and x2 >= 2440 and y2 >= 1450
    assert x2 - x1 < 1200 and y2 - y1 < 450  # just the missed block
    assert found[0]["recovered"]


def test_build_regions_end_to_end():
    page = np.full((2400, 3000), 240, np.uint8)
    for x in (100, 1100, 2100):
        text_lines(page, x, 400, x + 850, 2300)
    text_lines(page, 100, 100, 2900, 300, line=30)  # a full width heading YOLO misses
    detections = pd.DataFrame(
        [
            {"class": "plain text", "confidence": 0.9, "x1": 90, "y1": 390, "x2": 960, "y2": 1400},
            {"class": "plain text", "confidence": 0.9, "x1": 90, "y1": 1350, "x2": 960, "y2": 2310},  # overlaps the one above
            {"class": "plain text", "confidence": 0.9, "x1": 1090, "y1": 390, "x2": 1960, "y2": 2310},
            {"class": "plain text", "confidence": 0.9, "x1": 2090, "y1": 390, "x2": 2960, "y2": 2310},
            {"class": "abandon", "confidence": 0.5, "x1": 2800, "y1": 2350, "x2": 2900, "y2": 2390},
            {"class": "figure_caption", "confidence": 0.5, "x1": 1200, "y1": 1000, "x2": 1500, "y2": 1100},  # nested
        ]
    )
    regions = build_regions(detections, page)
    boxes = regions[["x1", "y1", "x2", "y2"]].to_numpy()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert overlap_area(boxes[i], boxes[j]) <= 2 * 2 * 3000  # at most rounding slivers
    assert regions["recovered"].sum() == 1
    first = regions.iloc[0]
    assert first["recovered"] and first["y1"] < 150 and first["x2"] - first["x1"] > 2500  # heading read first
    # then column 1 (two boxes), column 2, column 3
    assert list(regions["x1"].iloc[1:]) == sorted(regions["x1"].iloc[1:])
    assert set(regions["class"]) == {"text"}
    assert regions["box_page_id"].is_unique


def test_recovered_boxes_give_way_to_detections():
    page = np.full((800, 1600), 240, np.uint8)
    text_lines(page, 300, 100, 1200, 700)
    ink = ink_mask(page)
    detected = box("text", 300, 100, 1200, 700)
    recovered = {**box("text", 50, 150, 360, 650), "recovered": True}  # e.g. an illustration beside the text
    a, b = split_overlaps([detected, recovered], ink)
    assert a["bbox"] == (300, 100, 1200, 700)  # the detection keeps its first letters
    assert b["bbox"][2] == 300


def test_only_figures_with_their_own_text_are_transcribed():
    import pandas as pd

    from newspaper_pipeline.stages.regions import figures_with_own_text

    def region(bpid, cls, order, x1, y1, x2, y2):
        return {"box_page_id": bpid, "class": cls, "reading_order": order, "x1": x1, "y1": y1, "x2": x2, "y2": y2}

    regions = pd.DataFrame(
        [
            region("lone", "figure", 1, 0, 0, 100, 100),  # an advert nothing else covers
            region("dup_text", "text", 2, 200, 0, 300, 100),
            region("dup_fig", "figure", 3, 200, 0, 300, 100),  # same box as a text region
            region("ad", "figure", 4, 0, 200, 300, 400),
            region("heading", "title", 5, 20, 210, 280, 250),  # detected inside the advert
            region("table_a", "figure", 6, 400, 0, 500, 100),
            region("table_b", "figure", 7, 402, 2, 498, 98),  # the same table detected twice
        ]
    )
    assert figures_with_own_text(regions) == {"lone", "table_a"}
