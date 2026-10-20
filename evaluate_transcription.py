"""
Score pipeline transcriptions against ground truth: character and word error rates.

Built for BLN600 (https://doi.org/10.15131/shef.data.25439023): 600 nineteenth-century
newspaper excerpts with human transcriptions. Unzip it into data/BLN600 (password BLN600,
AES-encrypted, so use 7-Zip or pyzipper), run the pipeline on some of its images, then score:

    python run_pipeline.py --input data/BLN600/Images --output output/bln600
    python evaluate_transcription.py output/bln600

Each page's text is its published regions in reading order: not withheld ones, and figures
only when they were transcribed rather than described.
Both sides are normalised the same way: Unicode NFKC, long s and curly quotes folded, words
hyphenated at a line end joined, whitespace collapsed. --dashes also ignores how dashes are
spaced or doubled ("GAME. -Lawn" vs "GAME.--Lawn"), a typesetting convention rather than a
reading error. CER micro = total edit distance / total ground-truth characters.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

FOLD = str.maketrans({"ſ": "s", "‘": "'", "’": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-", "¬": "-", " ": " "})


def normalise(text: str, dashes: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text).translate(FOLD)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    if dashes:
        text = re.sub(r"\s*-+\s*", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def published_text(page_dir: Path) -> str:
    regions = pd.read_parquet(page_dir / "regions.parquet").sort_values("reading_order")
    # Figures count when they were transcribed (ocr_class "text"), not when they were described
    read_as = regions["ocr_class"] if "ocr_class" in regions else regions["class"]
    keep = regions[(read_as != "figure") & (regions["suspect"].fillna("") == "") & regions["text"].fillna("").ne("")]
    return "\n".join(keep["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="pipeline --output folder")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/BLN600/Ground Truth"), help="folder of <page_id>.txt")
    parser.add_argument("--dashes", action="store_true", help="ignore dash spacing and doubling")
    args = parser.parse_args()

    rows = []
    for page_dir in sorted(p for p in args.output.iterdir() if (p / "regions.parquet").exists()):
        truth_file = args.ground_truth / f"{page_dir.name}.txt"
        if not truth_file.exists():
            continue
        truth = normalise(truth_file.read_text(encoding="utf-8", errors="replace"), args.dashes)
        text = normalise(published_text(page_dir), args.dashes)
        rows.append(
            {
                "page_id": page_dir.name,
                "chars": len(truth),
                "char_errors": Levenshtein.distance(text, truth),
                "words": len(truth.split()),
                "word_errors": Levenshtein.distance(text.split(), truth.split()),
            }
        )
    if not rows:
        raise SystemExit(f"No pages in {args.output} have ground truth in {args.ground_truth}")
    scores = pd.DataFrame(rows)
    scores["cer"] = scores.char_errors / scores.chars
    scores["wer"] = scores.word_errors / scores.words
    cer = scores.char_errors.sum() / scores.chars.sum()
    wer = scores.word_errors.sum() / scores.words.sum()
    print(f"{len(scores)} pages, {scores.chars.sum()} ground-truth characters")
    print(f"CER {cer:.4f} (character accuracy {1 - cer:.2%}), median page {scores.cer.median():.4f}")
    print(f"WER {wer:.4f} (word accuracy {1 - wer:.2%}), median page {scores.wer.median():.4f}")
    print("Worst pages:")
    print(scores.sort_values("cer", ascending=False).head(5)[["page_id", "chars", "cer", "wer"]].to_string(index=False))


if __name__ == "__main__":
    main()
