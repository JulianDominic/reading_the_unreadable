# Improved Pipeline

The goal of this pipeline is to improve on current pipeline found in @run_sync_pipeline.py

1. Given an input of Newspaper Pages, for each page, convert them to `png`
2. For each `png` Newspaper Page, do pre-processing on the page. This comes in terms of Image manipulation (contrast, masking, etc)
3. For each processed Newspaper Page, run the layout tool (DocLayout-YOLO)
4. For each region identified in the layout, run the OCR tools on it. `chandra-ocr-2` for extracting the text, and `tesseract` for getting the word-level coordinates. Bottleneck: `chandra-ocr-2` can only have 2 concurrent sessions.
5. When all regions of a particular page has been processed, fit the data into the IIIF (International Image Interoperability Framework) standard
6. Cleanup any temp and/or non-output files.

For each step, use a group of workers to do it, don't need to process each entity one-by-one, can be done concurrently. Once it has finished a step, it can move onto the next step without needing to wait for the rest of its neighbouring entities.

System constraints:

- Able to run on a MacBook Pro with M5Pro Chip, 24GB Unified Memory
