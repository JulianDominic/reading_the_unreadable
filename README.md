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
building → chandra (text) + Tesseract (word boxes) → align → quality check → IIIF → cleanup

```bash
python run_pipeline.py                  # input_images/ -> output/iiif/
python run_pipeline.py --help           # all options
```

- Each stage has its own workers and pages move on independently. chandra runs at most 2
  requests at once.
- Regions don't overlap, text YOLO missed is recovered, and reading order follows the columns.
  `--region-builder legacy` uses the paper's original post-processing instead.
- chandra's text is aligned onto Tesseract's word boxes, and tables are matched cell by cell.
- Output that isn't a transcription (repetition loops, image descriptions, far too many
  words) is flagged in `regions.parquet` and left out of the IIIF annotations.
- Re-runs skip finished pages (`_SUCCESS` marker) and reuse cached chandra answers.

Output per page in `output/iiif/<page_id>/`:

| File | Contents |
|---|---|
| `manifest.json` | One canvas in original-scan coordinates, served by Cantaloupe |
| `annotations/{blocks,lines,words}.json` | `supplementing` annotations with `textGranularity` |
| `{regions,lines,words}.parquet` | Flat exports |
| `page.txt` | Text in reading order |
| `overlay.jpg` | Boxes drawn on the scan, for checking |
| `report.json` | Scores, timings, flagged regions |

## Viewer (Mirador + Cantaloupe)

```bash
docker compose up --build -d            # Cantaloupe on :8182, manifests + Mirador on :8000
open http://localhost:8000
docker compose down
```

- The viewer shows word boxes by default. Add `?layers=blocks,lines,words` for every layer or
  `?manifest=<page_id>` for a single page.
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
