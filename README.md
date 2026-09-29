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
