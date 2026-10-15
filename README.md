# Reading the Unreadable

OCR pipeline for historical newspaper images using vision-language models.

## Quick Start

### 1. Installation

Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt install imagemagick

# macOS
brew install imagemagick
export MAGICK_HOME=$(brew --prefix imagemagick)
```

Install Python packages:
```bash
uv pip install -r requirements.txt
uv add mistralai bs4
uv pip install -e .
```

### 2. Environment Setup

Create a `.env` file:

```bash
OPENAI_API_BASE=your_api_endpoint
OPENAI_API_KEY=your_api_key
```

### 3. Add Images

Place newspaper images (PNG format) in the `input_images/` folder.

### 4. Run Pipeline (~8 minutes per page)

```bash
python run_sync_pipeline.py
```

This will:
- Detect layout regions using DocLayout-YOLO
- Extract text using vision-language models
- Save results to `data/periodical_dataframes/post_processed/custom_newspaper.parquet`

### 5. View Results

Extract plain text from the HTML (`chandra-ocr-2`):
```bash
python view_ocr_results.py
```

Visualize detected regions, this creates `detected_regions.py`:
```bash
python visualize_regions.py
```

## Concurrent IIIF Pipeline (`run_pipeline.py`)

A faster pipeline that turns archive scans (JPEG2000, TIFF, PNG, JPG) into
[IIIF Presentation 3.0](https://iiif.io/api/presentation/3.0/) manifests with block, line
and word annotations, ready for Mirador with Cantaloupe as the image server.

For every page: convert → pre-process (border mask, deskew, background flattening and
denoise) → DocLayout-YOLO layout → region building → per region `chandra-ocr-2` (text) and
Tesseract (word boxes) in parallel → align chandra's text onto Tesseract's boxes → IIIF +
exports → cleanup.

Region building (`newspaper_pipeline/stages/regions.py`) turns YOLO's raw boxes into
non-overlapping regions: nested duplicates are dropped, partial overlaps are split along
the emptiest row or column, text YOLO missed (mastheads, datelines, headings) is recovered
from uncovered ink, and reading order comes from a recursive XY-cut that reads columns top
to bottom, including under headings that span several columns. On the 1874 sample page the
original post-processing (`function_modules.bbox_functions`, still used by
`run_sync_pipeline.py` and available here with `--region-builder legacy`) covered 80% of the
region area two or more times, so most text was transcribed twice; the new builder covers it
once, leaves less ink outside any region (2% vs 10%) and sends half as many pixels to chandra.

chandra is called with `temperature=0` (at the server default the same crop returned 70,
71, 119 and 70 words over four requests) and a per-request token cap based on how much
text the crop can hold, which stops the occasional greedy repetition loop early. Output
that hits the cap, describes the image instead of transcribing it, or is far longer than
the region can hold is kept in `regions.parquet` (`suspect` column) but left out of the
IIIF annotations.
Each stage has its own pool of workers and every page moves on as soon as it is ready.
chandra is limited to 2 concurrent requests and is the bottleneck; everything else runs
while it works. Code lives in `newspaper_pipeline/`.

### Setup

```bash
brew install tesseract                    # language data (eng.traineddata) for tesserocr
uv sync --inexact --extra pipeline        # --inexact keeps packages installed from requirements.txt
```

`.env` needs the same `OPENAI_API_BASE` / `OPENAI_API_KEY` as above. Set `TESSDATA_PREFIX`
if your `tessdata` folder is not in a Homebrew location.

### Run

```bash
python run_pipeline.py                     # input_images/ -> output/iiif/
python run_pipeline.py --input /scans --image-base-url https://images.example.org/iiif/3 \
    --manifest-base-url https://data.example.org/iiif
python run_pipeline.py --help              # all options
```

Re-running skips pages that finished (`_SUCCESS` marker) and serves chandra answers from
`output/cache/chandra`, so an interrupted run resumes cheaply. Pages with a failed region
are still written but not marked finished, so the next run retries them.

### Output (`output/iiif/<page_id>/`)

| File | Contents |
|---|---|
| `manifest.json` | One canvas sized to the original scan, painted by the Cantaloupe image service |
| `annotations/{blocks,lines,words}.json` | `supplementing` annotations with `textGranularity` |
| `regions.parquet`, `lines.parquet`, `words.parquet` | Flat exports (text, chandra HTML, alignment scores, boxes) |
| `page.txt` | Region text in reading order |
| `overlay.jpg` | Regions and word boxes drawn on the original scan, for checking |
| `report.json` | Deskew angle, timings per stage, problems |

`output/iiif/collection.json` lists every page manifest.

All coordinates are original-scan pixels: pre-processing records its crop and rotation
as an affine transform and every box is mapped back through it.

### Mirador + Cantaloupe

- Point Cantaloupe's `FilesystemSource` at the input folder. Image identifiers are the
  scan's path relative to that folder, URL-encoded (`a/b.jp2` → `a%2Fb.jp2`); if you set
  Cantaloupe's `slash_substitute`, pass the same value with `--slash-substitute`.
- Serve the output folder over HTTP at `--manifest-base-url` (with CORS enabled).
- Mirador 4 shows annotations of every motivation by default. Mirador 3 filters out
  `supplementing`, so on Mirador 3 add it to the config:

```js
Mirador.viewer({
  id: "viewer",
  annotations: { filteredMotivations: ["oa:commenting", "oa:tagging", "sc:painting", "commenting", "tagging", "supplementing"] },
  windows: [{ manifestId: "http://localhost:8000/<page_id>/manifest.json" }],
});
```

For large TIFFs, converting them once to tiled pyramidal TIFF (e.g. `vips tiffsave in.tif
out.tif --tile --pyramid --compression jpeg`) makes Cantaloupe much faster.

### Tests

```bash
uv sync --inexact --extra pipeline --extra dev
python -m pytest                           # unit tests + an offline end-to-end run on the sample page
python -m pytest -m "not integration"      # unit tests only
```

## Changing Models

Edit `run_sync_pipeline.py` line 195 to use a different model:
```python
model="openai/your-model-name",
```

Clear the cache to re-run OCR:
```bash
rm data/cache/custom_newspaper_ocr_cache.parquet
```

## Citation

If you use this project, please cite:
[Reading the unreadable: Creating a dataset of 19th century English newspapers using image-to-text language models](https://doi.org/10.1093/llc/fqaf151)
