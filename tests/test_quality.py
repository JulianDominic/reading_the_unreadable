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


def test_cut_off_column_at_the_page_edge_is_withheld():
    from newspaper_pipeline.stages.quality import assess_region

    fragments = "to e, in le een, ed all he he ce ed he vn an vn on be he he in- no"
    prose = "Frederick Percy Sadler, a fish salesman, employed at Billingsgate, and living at Forest Glen"
    args = (60, 1700, 900.0, 20, 18, 0.5)  # 60 px wide with words about 30 px: two words wide
    assert assess_region("text", fragments, fragments, *args, at_page_edge=True) == "cut-off text at the page edge"
    assert assess_region("text", fragments, fragments, *args) is None  # the same text away from the edge is kept
    # a real column of prose at the edge is at least a few words wide
    assert assess_region("text", prose, prose, 400, 1700, 900.0, 20, 18, 0.5, at_page_edge=True) is None
    # a strip under 2.5 word sizes wide is withheld whatever it holds (numbers have no letters to judge)
    numbers = "0 1 16 11 1946 1946 1946 1946 4 5 6 7 8 9 10 11 12"
    assert assess_region("text", numbers, numbers, 42, 633, 900.0, 17, 12, 0.0, at_page_edge=True) == "cut-off text at the page edge"
    # a real column at the edge of a full page is many words wide, even with an advert's short tokens
    advert = "J. E. PHILLIPS, 78 View St. W T F No. 1 Pass. Daily"
    assert assess_region("text", advert, advert, 700, 400, 900.0, 12, 12, 0.9, at_page_edge=True) is None


def test_victorian_prose_is_not_mistaken_for_model_commentary():
    from newspaper_pipeline.stages.quality import transcription_problem

    for prose in (
        "The Coroner: What is the cause of death?—Dr. Matcham: Starvation. I cannot conceive there is any foundation.",
        "The complainant, who is twenty-five years of age, appears to be of German extraction.",
        "The movements of the British North Pacific squadron appears to be rather uncertain.",
    ):
        assert transcription_problem("text", prose, prose) is None
    description = "Royal coat of arms of the United Kingdom, with a lion and a unicorn.\nThe image shows the arms."
    assert transcription_problem("title", description, description) == "model commentary instead of a transcription"
    assert transcription_problem("text", "I cannot read this text, it is heavily cropped.", "") is not None
