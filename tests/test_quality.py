from newspaper_pipeline.stages.quality import assess_region, is_layout_json


def assess(text, cls="text", raw=None, w=1000, h=300, n_tess=None, score=0.9):
    tokens = len(text.split())
    return assess_region(cls, text, raw if raw is not None else text, w, h, 3500.0, tokens, tokens if n_tess is None else n_tess, score)


def test_normal_transcription_passes():
    assert assess("THE UNDERSIGNED HAVING PURCHASED the Colonial Hotel and Driard House") is None


def test_layout_json_is_flagged():
    raw = '[{"label": "Section-Header", "bbox": "126 137 891 182"}, {"label": "Text", "bbox": "0 194 27 421"}]'
    assert is_layout_json(raw)
    assert assess(raw) == "layout JSON instead of text"
    assert not is_layout_json("[1874] Victoria")


def test_commentary_is_flagged_except_for_figures():
    text = "No newspaper title text is visible as the image is heavily cropped on the left"
    assert "commentary" in assess(text, cls="title")
    assert assess("A woodcut of a ship; the image shows three masts", cls="figure") is None


def test_more_words_than_the_region_can_hold():
    # 693 tokens for a 5814x20 rule line on the sample page
    assert "holds about" in assess("word " * 693, w=5814, h=20, n_tess=24, score=0.01)


def test_text_unrelated_to_what_tesseract_saw():
    reason = assess("lorem " * 80, w=3000, h=900, n_tess=12, score=0.05)
    assert reason and "alignment 0.05" in reason


def test_hitting_the_token_limit_is_flagged():
    reason = assess_region("figure", "", '[{"x": 10, "y": 10}, {"x": 10, "y": 10}', 456, 162, 3500.0, 0, 0, 0.0, ["length"])
    assert reason == "output hit its token limit (repetition loop)"
    assert assess_region("text", "fine text here", "fine text here", 900, 300, 3500.0, 3, 3, 1.0, ["stop"]) is None
