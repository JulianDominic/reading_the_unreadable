"""
Concurrent newspaper OCR pipeline producing IIIF Presentation 3.0 manifests.

Stages: convert -> preprocess -> layout (DocLayout-YOLO) -> OCR (chandra-ocr-2 + Tesseract)
-> alignment -> IIIF -> cleanup. Each page moves through the stages independently.
See run_pipeline.py for the command line entry point.
"""
