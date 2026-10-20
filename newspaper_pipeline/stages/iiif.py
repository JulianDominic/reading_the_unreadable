"""
Stage 6: IIIF Presentation 3.0 output.

Per page the output folder gets:
    manifest.json               one Canvas painted with the Cantaloupe image service
    annotations/blocks.json     one annotation per layout region (textGranularity "block")
    annotations/lines.json      one per aligned text line ("line")
    annotations/words.json      one per aligned word ("word")

Annotation pages are referenced from canvas.annotations rather than embedded, because a
newspaper page can carry tens of thousands of word annotations. All targets use the
original scan's pixel space, which is the canvas size Cantaloupe reports.

Mirador 4 displays these annotations out of the box; Mirador 3 needs "supplementing"
added to `annotations.filteredMotivations` (see README).
"""

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from urllib.parse import quote

PRESENTATION_CONTEXT = "http://iiif.io/api/presentation/3/context.json"
TEXT_GRANULARITY_CONTEXT = "http://iiif.io/api/extension/text-granularity/context.json"
CONTEXT = [TEXT_GRANULARITY_CONTEXT, PRESENTATION_CONTEXT]
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "iiif_3_0.json"


def image_identifier(source_relpath: str, slash_substitute: Optional[str] = None) -> str:
    """
    Cantaloupe identifier for a scan path relative to its FilesystemSource base folder.

    With slash_substitute set (Cantaloupe's slash_substitute option) "/" is replaced by it;
    otherwise the path is fully percent-encoded, including "/".
    """
    path = source_relpath.replace("\\", "/")
    if slash_substitute:
        path = path.replace("/", slash_substitute)
    return quote(path, safe="")


ESTIMATED_TAG = {"type": "TextualBody", "value": "estimated position", "purpose": "tagging"}


def _text_annotation(annotation_id: str, target: Union[str, Dict], text: str, granularity: str, language: str, extra_bodies: Iterable[Dict] = ()) -> Dict:
    body = {"type": "TextualBody", "value": text, "format": "text/plain", "language": language}
    bodies = [body, *extra_bodies]
    return {
        "id": annotation_id,
        "type": "Annotation",
        "motivation": "supplementing",
        "textGranularity": granularity,
        "body": bodies if len(bodies) > 1 else body,
        "target": target,
    }


def _parts_target(canvas_id: str, boxes: List[Dict]) -> Dict:
    """
    Target for a word printed in parts (hyphenated across lines): an SVG selector with one
    rectangle per part, plus the first part as an xywh fragment for viewers without SVG.
    """
    path = " ".join(f"M{b['x']},{b['y']} h{b['w']} v{b['h']} h{-b['w']} Z" for b in boxes)
    first = boxes[0]
    return {
        "type": "SpecificResource",
        "source": {"id": canvas_id, "type": "Canvas"},
        "selector": [
            {"type": "FragmentSelector", "conformsTo": "http://www.w3.org/TR/media-frags/", "value": f"xywh={first['x']},{first['y']},{first['w']},{first['h']}"},
            {"type": "SvgSelector", "value": f'<svg xmlns="http://www.w3.org/2000/svg"><path d="{path}" fill="none" stroke="#00BFFF"/></svg>'},
        ],
    }


def _xywh(canvas_id: str, box: Dict) -> str:
    return f"{canvas_id}#xywh={box['x']},{box['y']},{box['w']},{box['h']}"


def build_page_documents(page: Dict, regions: List[Dict], config) -> Dict[str, Dict]:
    """
    Build the manifest and annotation pages for one page.

    Args:
        page: page_id, source_relpath, width, height (original scan) and optional label/metadata
        regions: Per-region results with original-scan boxes. Each region has
            box_page_id, class, reading_order, text, box {x, y, w, h},
            lines [{text, box}], words [{text, box}]
        config: PipelineConfig (base URLs, slash substitute, language)

    Returns:
        {"manifest.json": ..., "annotations/blocks.json": ..., "annotations/lines.json": ..., "annotations/words.json": ...}
    """
    page_id = page["page_id"]
    page_base = f"{config.manifest_base_url.rstrip('/')}/{quote(page_id)}"
    canvas_id = f"{page_base}/canvas/p1"
    identifier = image_identifier(page["source_relpath"], config.slash_substitute)
    service_id = f"{config.image_base_url.rstrip('/')}/{identifier}"
    language = config.language

    blocks, lines, words = [], [], []
    for region in sorted(regions, key=lambda r: r["reading_order"]):
        if region.get("suspect"):
            continue  # probably not a transcription (see stages/quality.py)
        rid = region["box_page_id"]
        tag = {"type": "TextualBody", "value": region["class"], "purpose": "tagging"}
        if region.get("text"):
            blocks.append(
                _text_annotation(f"{page_base}/annotation/block/{rid}", _xywh(canvas_id, region["box"]), region["text"], "block", language, [tag])
            )
        for n, line in enumerate(region.get("lines", [])):
            lines.append(_text_annotation(f"{page_base}/annotation/line/{rid}-{n}", _xywh(canvas_id, line["box"]), line["text"], "line", language))
        for n, word in enumerate(region.get("words", [])):
            # Estimated boxes are tagged so viewers and users can tell them from measured ones
            tags = [ESTIMATED_TAG] if word.get("match") == "estimated" else []
            target = _parts_target(canvas_id, word["part_boxes"]) if word.get("part_boxes") else _xywh(canvas_id, word["box"])
            words.append(_text_annotation(f"{page_base}/annotation/word/{rid}-{n}", target, word["text"], "word", language, tags))

    annotation_files = {"blocks": blocks, "lines": lines, "words": words}
    documents: Dict[str, Dict] = {}
    references = []
    for name, items in annotation_files.items():
        page_url = f"{page_base}/annotations/{name}.json"
        documents[f"annotations/{name}.json"] = {
            "@context": CONTEXT,
            "id": page_url,
            "type": "AnnotationPage",
            "label": {"en": [f"{name.capitalize()} ({len(items)})"]},
            "items": items,
        }
        references.append({"id": page_url, "type": "AnnotationPage", "label": {"en": [name.capitalize()]}})

    width, height = int(page["width"]), int(page["height"])
    metadata = [
        {"label": {"en": ["Source file"]}, "value": {"none": [page["source_relpath"]]}},
        {"label": {"en": ["Regions"]}, "value": {"none": [str(len(regions))]}},
        {"label": {"en": ["Generated"]}, "value": {"none": [datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")]}},
        {"label": {"en": ["OCR"]}, "value": {"none": [f"{config.chandra_model} text aligned to Tesseract word boxes; layout by DocLayout-YOLO"]}},
    ]
    for key, value in (page.get("metadata") or {}).items():
        metadata.append({"label": {"en": [str(key)]}, "value": {"none": [str(value)]}})

    documents["manifest.json"] = {
        "@context": CONTEXT,
        "id": f"{page_base}/manifest.json",
        "type": "Manifest",
        "label": {"none": [page.get("label", page_id)]},
        "metadata": metadata,
        "items": [
            {
                "id": canvas_id,
                "type": "Canvas",
                "label": {"none": [page.get("label", page_id)]},
                "width": width,
                "height": height,
                "items": [
                    {
                        "id": f"{canvas_id}/page/painting",
                        "type": "AnnotationPage",
                        "items": [
                            {
                                "id": f"{canvas_id}/annotation/painting",
                                "type": "Annotation",
                                "motivation": "painting",
                                "body": {
                                    "id": f"{service_id}/full/max/0/default.jpg",
                                    "type": "Image",
                                    "format": "image/jpeg",
                                    "width": width,
                                    "height": height,
                                    "service": [{"id": service_id, "type": "ImageService3", "profile": "level2"}],
                                },
                                "target": canvas_id,
                            }
                        ],
                    }
                ],
                "annotations": references,
            }
        ],
    }
    return documents


def build_collection(manifests: List[Dict], config, label: str = "Newspaper pages") -> Dict:
    """IIIF Collection listing every page manifest, so Mirador can browse the whole run."""
    base = config.manifest_base_url.rstrip("/")
    return {
        "@context": PRESENTATION_CONTEXT,
        "id": f"{base}/collection.json",
        "type": "Collection",
        "label": {"en": [label]},
        "items": [{"id": m["id"], "type": "Manifest", "label": m["label"]} for m in manifests],
    }


@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft7Validator

    return Draft7Validator(json.loads(SCHEMA_PATH.read_text()))


def validate(document: Dict) -> List[str]:
    """
    Validate a manifest, collection or annotation page against the IIIF Presentation 3 schema
    (vendored from github.com/IIIF/presentation-validator).

    Returns:
        Human readable error messages; empty when valid
    """
    errors = sorted(_validator().iter_errors(document), key=lambda e: list(e.absolute_path))
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:300]}" for e in errors]


def write_documents(page_dir: Path, documents: Dict[str, Dict]) -> None:
    """Write documents atomically (write to .tmp, then rename) so a crash never leaves half a manifest."""
    for relative, document in documents.items():
        path = Path(page_dir) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
        tmp.replace(path)
