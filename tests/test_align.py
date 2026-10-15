from newspaper_pipeline.stages.align import align_region, align_tokens, normalise_token


def word(text, line, x, conf=90.0):
    return {"text": text, "conf": conf, "line": line, "block": 0, "x1": x, "y1": line * 20, "x2": x + 40, "y2": line * 20 + 15}


def test_normalise_token():
    assert normalise_token("Hotel,") == "hotel"
    assert normalise_token("ſtreet") == "street"
    assert normalise_token("ﬁne") == "fine"
    assert normalise_token("—") == "—"  # pure punctuation keeps its symbol


def test_exact_and_fuzzy_matches_carry_chandra_text():
    words = [word("Colonial", 0, 0), word("Hotcl", 0, 50), word("and", 0, 100), word("Driard", 1, 0), word("Housc.", 1, 50)]
    result = align_region("Colonial Hotel and\nDriard House.", words)
    texts = [(w["text"], w["tesseract_text"], w["match"]) for w in result["words"]]
    assert texts == [
        ("Colonial", "Colonial", "exact"),
        ("Hotel", "Hotcl", "fuzzy"),
        ("and", "and", "exact"),
        ("Driard", "Driard", "exact"),
        ("House.", "Housc.", "fuzzy"),
    ]
    assert result["score"] == 1.0
    assert [line["text"] for line in result["lines"]] == ["Colonial Hotel and", "Driard House."]


def test_noise_words_are_dropped_and_unmatched_chandra_text_kept_on_line():
    # Tesseract picked up bleed-through garbage and missed the word "first"
    words = [word("in", 0, 0), word("~~#", 0, 50, conf=5), word("class", 0, 100), word("style", 0, 150)]
    result = align_region("in first class style", words)
    assert [w["text"] for w in result["words"]] == ["in", "class", "style"]
    assert result["lines"][0]["text"] == "in first class style"
    assert result["score"] == 0.75
    # the low confidence noise word does not stretch the line box
    assert result["lines"][0]["x1"] == 0 and result["lines"][0]["x2"] == 190


def test_alignment_is_monotonic_with_repeated_words():
    tess = ["the", "cat", "and", "the", "dog"]
    chandra = ["the", "cat", "and", "the", "dog"]
    assert [(i, j) for i, j, _ in align_tokens(tess, chandra)] == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


def test_dissimilar_words_are_not_forced_together():
    result = align_region("Liquors", [word("xqz", 0, 0)])
    assert result["words"] == [] and result["lines"] == [] and result["score"] == 0.0


def test_empty_inputs():
    assert align_region("", [word("a", 0, 0)])["n_matched"] == 0
    assert align_region("some text", [])["n_chandra_tokens"] == 2


def test_columns_read_in_different_order_are_still_aligned():
    # Tesseract reads column B before column A; chandra reads A then B
    col_a = "the daily british colonist published every day".split()
    col_b = "insurance company of canada pays in case of death".split()
    words = [word(t, 0, 10 * k) for k, t in enumerate(col_b)] + [word(t, 1, 10 * k) for k, t in enumerate(col_a)]
    result = align_region(" ".join(col_a + col_b), words)
    assert result["score"] == 1.0
    # each chandra token sits on the Tesseract word with the same text
    assert all(w["text"] == w["tesseract_text"] for w in result["words"])
    assert [line["text"] for line in result["lines"]] == [" ".join(col_b), " ".join(col_a)]


def cell(text, line, x, width=None):
    width = width or 22 * len(text)
    return {"text": text, "conf": 90.0, "line": line, "block": 0, "x1": x, "y1": line * 40, "x2": x + width, "y2": line * 40 + 25}


def test_table_read_column_wise_against_row_wise_tesseract():
    """Chandra lists each column top to bottom; Tesseract reads each row across all columns.
    Identical cells ("of each month") must still land in their own column."""
    rows = [
        ("Canada, Great Britain", "About the 10th", "About the 12th"),
        ("Europe,", "of each month", "of each month"),
        ("Esquimalt", "Daily at 10:30", "Daily at 11"),
        ("Metchosin", "Monday, 1 P.M.", "Tuesday, 12 Noon."),
    ]
    words = []
    for line, row in enumerate(rows):
        for column, text in enumerate(row):
            x = 20 + column * 700
            for token in text.split():
                words.append(cell(token, line, x))
                x += 22 * len(token) + 15
    chandra = "\n".join(row[c] for c in range(3) for row in rows)  # column by column
    result = align_region(chandra, words)
    assert result["score"] == 1.0
    tokens = chandra.split()
    for w in result["words"]:
        expected_column = next(c for c in range(3) for row in rows if w["chandra_index"] in range(*_span(tokens, chandra, row[c], c, rows)))
        assert w["x1"] // 700 == expected_column, (w["text"], w["x1"])
    # one line annotation per table cell, not per table row
    assert len(result["lines"]) == 12


def _span(tokens, chandra, text, column, rows):
    """Token index range of a given cell in the column-wise chandra text."""
    start = 0
    for c in range(3):
        for row in rows:
            n = len(row[c].split())
            if c == column and row[column] == text:
                return start, start + n
            start += n
    raise ValueError(text)


def test_dot_leader_junk_does_not_block_pairing():
    words = [cell("Metchosin.....", 0, 20), {**cell("Seseeeee", 0, 330), "conf": 0.0}, cell("Monday,", 0, 720), cell("1", 0, 900)]
    result = align_region("Metchosin.....\nMonday, 1", words)
    assert sorted(w["text"] for w in result["words"]) == ["1", "Metchosin.....", "Monday,"]
