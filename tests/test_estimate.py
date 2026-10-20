import cv2
import numpy as np

from newspaper_pipeline.stages.align import align_region

# One printed line of four words, 10 px per character, 15 px word gaps
LAYOUT = {"alpha": (10, 60), "beta": (75, 115), "gamma": (130, 180), "delta": (195, 245)}
TOP, BOTTOM = 20, 40


def crop(tmp_path, names=tuple(LAYOUT)):
    image = np.full((60, 300), 255, np.uint8)
    for name in names:
        x1, x2 = LAYOUT[name]
        cv2.rectangle(image, (x1, TOP), (x2 - 1, BOTTOM - 1), 0, -1)
    path = tmp_path / "crop.png"
    cv2.imwrite(str(path), image)
    return str(path)


def tess(name, text=None):
    x1, x2 = LAYOUT[name]
    return {"text": text or name, "conf": 90.0, "line": 0, "block": 0, "x1": x1, "y1": TOP, "x2": x2, "y2": BOTTOM}


def boxes(result):
    return {w["text"]: (w["match"], (w["x1"], w["y1"], w["x2"], w["y2"])) for w in result["words"]}


def test_word_tesseract_missed_is_cut_from_the_ink_between_its_neighbours(tmp_path):
    words = [tess("alpha"), tess("gamma"), tess("delta")]
    result = align_region("alpha beta gamma delta", words, crop_path=crop(tmp_path))
    assert boxes(result)["beta"] == ("estimated", (75, TOP, 115, BOTTOM))
    assert [w["text"] for w in result["words"]] == ["alpha", "beta", "gamma", "delta"]  # chandra order
    assert result["n_estimated"] == 1 and result["n_matched"] == 3
    assert result["score"] == 0.75  # the score counts measured matches only
    estimated = next(w for w in result["words"] if w["match"] == "estimated")
    assert estimated["tesseract_text"] == "" and estimated["conf"] is None


def test_word_at_the_end_of_a_line_extends_the_line_box(tmp_path):
    words = [tess("alpha"), tess("beta"), tess("gamma")]
    result = align_region("alpha beta gamma delta", words, crop_path=crop(tmp_path))
    assert boxes(result)["delta"] == ("estimated", (195, TOP, 245, BOTTOM))
    assert result["lines"][0]["x2"] == 245


def test_misread_tesseract_box_is_reused(tmp_path):
    words = [tess("alpha"), tess("beta", text="xq#z"), tess("gamma"), tess("delta")]
    result = align_region("alpha beta gamma delta", words, crop_path=crop(tmp_path))
    beta = next(w for w in result["words"] if w["text"] == "beta")
    assert beta["match"] == "estimated" and beta["tesseract_text"] == "xq#z"
    assert (beta["x1"], beta["x2"]) == (75, 115)


def test_no_box_without_ink_or_when_the_width_does_not_fit(tmp_path):
    words = [tess("alpha"), tess("gamma"), tess("delta")]
    # nothing printed where the missing word should be
    blank = align_region("alpha beta gamma delta", words, crop_path=crop(tmp_path, ("alpha", "gamma", "delta")))
    assert "beta" not in boxes(blank) and blank["n_estimated"] == 0
    # a 15-character word can't be the 40 px of ink there
    long = align_region("alpha extraordinarily gamma delta", words, crop_path=crop(tmp_path))
    assert "extraordinarily" not in boxes(long)
    # unmatched text still reaches the line annotations
    assert " ".join(line["text"] for line in long["lines"]) == "alpha extraordinarily gamma delta"


def test_no_estimates_without_the_crop():
    result = align_region("alpha beta gamma delta", [tess("alpha"), tess("gamma"), tess("delta")])
    assert result["n_estimated"] == 0 and "beta" not in boxes(result)


def two_line_crop(tmp_path, bars):
    image = np.full((80, 300), 255, np.uint8)
    for x1, y1, x2, y2 in bars:
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), 0, -1)
    path = tmp_path / "crop2.png"
    cv2.imwrite(str(path), image)
    return str(path)


def box_word(text, line, x1, x2, y1):
    return {"text": text, "conf": 90.0, "line": line, "block": 0, "x1": x1, "y1": y1, "x2": x2, "y2": y1 + 20}


def test_word_hyphenated_across_lines_gets_both_parts(tmp_path):
    # "the old cap-" / "tain said so": chandra joins the two halves into "captain"
    line0 = [box_word("the", 0, 10, 40, 10), box_word("old", 0, 55, 85, 10), box_word("cap-", 0, 100, 140, 10)]
    line1 = [box_word("tain", 1, 10, 50, 40), box_word("said", 1, 65, 105, 40), box_word("so", 1, 120, 140, 40)]
    words = line0 + line1
    crop = two_line_crop(tmp_path, [(w["x1"], w["y1"], w["x2"], w["y2"]) for w in words])
    result = align_region("the old captain\nsaid so", words, crop_path=crop)
    captain = next(w for w in result["words"] if w["text"] == "captain")
    assert captain["parts"] == [(100, 10, 140, 30), (10, 40, 50, 60)]
    assert (captain["x1"], captain["y1"], captain["x2"], captain["y2"]) == (100, 10, 140, 30)  # the head


def test_fused_tesseract_box_is_split_between_the_words(tmp_path):
    # Tesseract boxed "well known" as one word "wellknown"
    words = [box_word("a", 0, 10, 20, 20), box_word("wellknown", 0, 35, 130, 20), box_word("firm", 0, 145, 185, 20)]
    bars = [(10, 20, 20, 40), (35, 20, 75, 40), (85, 20, 130, 40), (145, 20, 185, 40)]
    result = align_region("a well known firm", words, crop_path=two_line_crop(tmp_path, bars))
    boxes = {w["text"]: (w["x1"], w["x2"]) for w in result["words"]}
    assert boxes["well"] == (35, 75) and boxes["known"] == (85, 130)


def test_box_too_narrow_for_its_word_is_dropped(tmp_path):
    # Tesseract boxed only the comma of "Victoria," (5 px for 9 characters)
    words = [box_word("Esquimalt", 0, 10, 100, 20), box_word("and", 0, 115, 145, 20), box_word("Victoria,", 0, 245, 250, 20), box_word("road", 0, 160, 200, 20)]
    result = align_region("Esquimalt and road Victoria,", words, crop_path=two_line_crop(tmp_path, []))
    assert "Victoria," not in {w["text"] for w in result["words"]}


def test_fused_box_with_a_one_letter_word_is_split(tmp_path):
    # Tesseract read "in a" as one word "ina"
    words = [box_word("up", 0, 10, 30, 20), box_word("ina", 0, 45, 85, 20), box_word("minute", 0, 100, 160, 20)]
    bars = [(10, 20, 30, 40), (45, 20, 65, 40), (75, 20, 85, 40), (100, 20, 160, 40)]
    result = align_region("up in a minute", words, crop_path=two_line_crop(tmp_path, bars))
    boxes = {w["text"]: (w["x1"], w["x2"]) for w in result["words"]}
    assert boxes["in"] == (45, 65) and boxes["a"] == (75, 85)


def test_box_on_blank_paper_is_dropped(tmp_path):
    words = [box_word("up", 0, 10, 30, 20), box_word("A", 0, 45, 60, 20), box_word("minute", 0, 100, 160, 20)]
    bars = [(10, 20, 30, 40), (100, 20, 160, 40)]  # nothing printed where Tesseract put "A"
    result = align_region("up A minute", words, crop_path=two_line_crop(tmp_path, bars))
    assert "A" not in {w["text"] for w in result["words"]}
