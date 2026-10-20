# Reading the Unreadable

OCR pipeline for historical newspaper images using vision-language models.

## Setup

```bash
brew install tesseract imagemagick            # Ubuntu: apt install tesseract-ocr imagemagick
uv sync --inexact --extra pipeline --extra dev
```

Create `.env` for the chandra-ocr-2 (vLLM) endpoint:

```bash
OPENAI_API_BASE=your_api_endpoint
OPENAI_API_KEY=your_api_key
```

Put scans (JPEG2000, TIFF, PNG or JPG) in `input_images/`.

## IIIF pipeline (`run_pipeline.py`)

Turns scans into [IIIF Presentation 3.0](https://iiif.io/api/presentation/3.0/) manifests
with block, line and word annotations:

convert → pre-process (border mask, deskew, denoise) → DocLayout-YOLO layout → region
building → chandra (text) + Tesseract (word boxes) → align → repair and estimate word boxes →
quality check (with chandra retries) → IIIF → cleanup

```bash
python run_pipeline.py                  # input_images/ -> output/iiif/
python run_pipeline.py --help           # all options
```

- Each stage has its own workers and pages move on independently. chandra runs at most 2
  requests at once.
- Regions don't overlap, text YOLO missed is recovered, and reading order follows the columns.
  `--region-builder legacy` uses the paper's original post-processing instead.
- chandra's text is aligned onto Tesseract's word boxes, and tables are matched cell by cell.
- Tesseract boxes that don't fit their word are repaired: a word hyphenated across lines gets
  both parts (`cont_x/y/w/h` in `words.parquet`, a two-rectangle SVG selector in IIIF), a box
  holding two words is split at the ink gap, and boxes far too narrow, too wide or on blank
  paper are dropped.
- Words Tesseract missed or misread get an estimated box from their matched neighbours and the
  ink in the crop (`match` = `estimated` in `words.parquet`, tagged "estimated position" in the
  IIIF word annotations). In hold-out tests 98.2–99.3% of estimates overlap the real box
  (IoU ≥ 0.5); see [Accuracy](#accuracy). A word is skipped when the evidence doesn't fit.
- Output that isn't a transcription (repetition loops, model commentary, layout JSON) is sent
  to chandra again, up to twice, in smaller segments with a little sampling temperature. What
  still fails, plus far too many words and cut-off text from a neighbouring column or page at
  the scan's edge, is flagged in `regions.parquet` and left out of the IIIF annotations.
- Re-runs skip finished pages (`_SUCCESS` marker) and reuse cached chandra answers.

Output per page in `output/iiif/<page_id>/`:

| File | Contents |
|---|---|
| `manifest.json` | One canvas in original-scan coordinates, served by Cantaloupe |
| `annotations/{blocks,lines,words}.json` | `supplementing` annotations with `textGranularity` |
| `{regions,lines,words}.parquet` | Flat exports |
| `page.txt` | Text in reading order |
| `overlay.jpg` | Boxes drawn on the scan, for checking |
| `report.json` | Scores, estimated word count, timings, flagged regions |

## Accuracy

Measured on 2026-10-19.

| What | Result | How |
|---|---|---|
| Characters | **99.0%** (CER 1.00%) | 150 held-out BLN600 excerpts, strict; 99.1% ignoring dash spacing. Gale's commercial OCR: 93.6% |
| Words | 96.7% | Same pages, strict; 97.7% ignoring dash spacing, 98.6% ignoring punctuation |
| Word boxes | **99.6%** (896 of 900) | Random words from both sets, checked by eye: box on the right word, IoU ≥ 0.5 |
| Estimated boxes | 98.2–99.3% | Hold-out: matched words hidden and re-placed (about 2.5% of words are estimated) |

Most remaining word errors are chandra misreadings and spelling it modernises
("Serjeant" → "Sergeant"); some are ground-truth slips. A prompt asking for exact
transcription made no measurable difference.

To score a run against BLN600 ([download](https://doi.org/10.15131/shef.data.25439023),
unzip into `data/BLN600` with password `BLN600`; it is AES-encrypted, so use 7-Zip or `pyzipper`):

```bash
python run_pipeline.py --input data/BLN600/Images --output output/bln600
python evaluate_transcription.py output/bln600            # add --dashes to ignore dash spacing
```

## Viewer (Mirador + Cantaloupe)

```bash
docker compose up --build -d            # Cantaloupe on :8182, manifests + Mirador on :8000
open http://localhost:8000
docker compose down
```

- The viewer shows word boxes by default. Add `?layers=blocks,lines,words` for every layer or
  `?manifest=<page_id>` for a single page.
- Ctrl+F / ⌘F searches the open page's OCR text as you type. Every match is highlighted and
  the view jumps to the current one, which is also selected in the annotation sidebar (its
  line or block if words are hidden). Enter / Shift+Enter step through matches, Esc closes the
  bar. Matching ignores case and accents (toggles for match case and whole words), works on
  phrases, and joins words hyphenated at a line end (both parts are highlighted).
- Estimated word boxes show an "estimated position" tag in the sidebar.
- `docker/iiif-viewer/mirador-fixes.js` patches a Mirador 4.3.0 bug (unbalanced canvas
  `save()` for hidden annotations) that made panning slower the longer the viewer was open.
  Recheck it when upgrading Mirador.
- `compose.yaml` mounts `./input_images` and `./output/iiif`; edit those paths if your files
  are elsewhere.
- On your own servers, point Cantaloupe at the scan folder and serve the output folder with
  CORS. Then pass `--image-base-url` and `--manifest-base-url` to the pipeline.
- Tiled pyramidal TIFF/JP2 makes Cantaloupe much faster, e.g.
  `vips tiffsave in.tif out.tif --tile --pyramid --compression jpeg`.

## Tests

```bash
python -m pytest                        # unit tests + offline end-to-end run
python -m pytest -m "not integration"   # unit tests only
```

## Original pipeline (`run_sync_pipeline.py`)

The synchronous pipeline from the paper takes about 8 minutes per page. It needs extra packages:

```bash
uv pip install -r requirements.txt && uv add mistralai bs4
export MAGICK_HOME=$(brew --prefix imagemagick)   # macOS
python run_sync_pipeline.py             # -> data/periodical_dataframes/post_processed/
python view_ocr_results.py              # plain text from chandra's HTML
python visualize_regions.py             # draw detected regions
```

To change the model, edit `model=` in `run_sync_pipeline.py`. To re-run OCR, delete
`data/cache/custom_newspaper_ocr_cache.parquet`.

## Citation

[Reading the unreadable: Creating a dataset of 19th century English newspapers using image-to-text language models](https://doi.org/10.1093/llc/fqaf151)
