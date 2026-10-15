import numpy as np
from PIL import Image

from newspaper_pipeline.config import PipelineConfig
from newspaper_pipeline.orchestrator import discover_sources, make_page_id
from newspaper_pipeline.stages.convert import convert_page
from newspaper_pipeline.stages.iiif import build_collection, build_page_documents, image_identifier, validate

CONFIG = PipelineConfig(image_base_url="https://images.example.org/iiif/3/", manifest_base_url="https://data.example.org/iiif")


def sample_documents():
    box = {"x": 100, "y": 200, "w": 300, "h": 40}
    regions = [
        {
            "box_page_id": "B0C1R1",
            "class": "text",
            "reading_order": 1,
            "text": "Colonial Hotel and Driard House.",
            "box": box,
            "lines": [{"text": "Colonial Hotel and", "box": box}],
            "words": [{"text": "Colonial", "box": {"x": 100, "y": 200, "w": 90, "h": 40}}],
        },
        {"box_page_id": "B0C1R2", "class": "figure", "reading_order": 2, "text": "", "box": box, "lines": [], "words": []},
    ]
    page = {"page_id": "colonist_1874-09-20_p1", "source_relpath": "1874/09/p1.jp2", "width": 6117, "height": 8875}
    return build_page_documents(page, regions, CONFIG)


def test_documents_validate_against_iiif_schema():
    for name, document in sample_documents().items():
        assert validate(document) == [], name


def test_manifest_points_at_cantaloupe_service():
    manifest = sample_documents()["manifest.json"]
    canvas = manifest["items"][0]
    body = canvas["items"][0]["items"][0]["body"]
    assert (canvas["width"], canvas["height"]) == (6117, 8875)
    assert body["service"][0] == {"id": "https://images.example.org/iiif/3/1874%2F09%2Fp1.jp2", "type": "ImageService3", "profile": "level2"}
    assert body["id"].endswith("/full/max/0/default.jpg")
    assert [ref["id"] for ref in canvas["annotations"]] == [
        f"https://data.example.org/iiif/colonist_1874-09-20_p1/annotations/{n}.json" for n in ("blocks", "lines", "words")
    ]


def test_annotations_target_canvas_regions():
    docs = sample_documents()
    canvas_id = docs["manifest.json"]["items"][0]["id"]
    word = docs["annotations/words.json"]["items"][0]
    assert word["target"] == f"{canvas_id}#xywh=100,200,90,40"
    assert word["motivation"] == "supplementing" and word["textGranularity"] == "word"
    assert word["body"]["value"] == "Colonial"
    block_bodies = docs["annotations/blocks.json"]["items"][0]["body"]
    assert {"type": "TextualBody", "value": "text", "purpose": "tagging"} in block_bodies
    assert len(docs["annotations/blocks.json"]["items"]) == 1  # figure without text has no block


def test_validator_rejects_broken_manifest():
    manifest = dict(sample_documents()["manifest.json"])
    del manifest["items"]
    assert validate(manifest)


def test_collection_validates():
    manifest = sample_documents()["manifest.json"]
    collection = build_collection([manifest], CONFIG)
    assert validate(collection) == []
    assert collection["items"][0]["id"] == manifest["id"]


def test_image_identifier_slash_substitute():
    assert image_identifier("a b/c.jp2") == "a%20b%2Fc.jp2"
    assert image_identifier("a/b/c.tif", slash_substitute="~") == "a~b~c.tif"


def test_page_ids_and_discovery(tmp_path):
    assert make_page_id(__import__("pathlib").Path("1874/Sept 20 (p1).jp2")) == "1874__Sept_20_p1"
    (tmp_path / "x").mkdir()
    for name in ("x/a.tif", "x/a.jp2", "b.png", ".hidden.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    pages = discover_sources(tmp_path)
    assert [p.source_relpath for p in pages] == ["b.png", "x/a.jp2", "x/a.tif"]
    assert len({p.page_id for p in pages}) == 3


def test_convert_decodes_jp2_and_16bit_tiff(tmp_path):
    gradient = (np.linspace(0, 255, 300 * 200).reshape(200, 300)).astype(np.uint8)
    Image.fromarray(gradient).convert("RGB").save(tmp_path / "scan.jp2")
    Image.fromarray(gradient.astype(np.uint16) * 257).save(tmp_path / "scan16.tif")

    jp2 = convert_page(str(tmp_path / "scan.jp2"), "scan.jp2", "scan", str(tmp_path / "work"))
    assert jp2["converted"] and (jp2["width"], jp2["height"]) == (300, 200)
    tif = convert_page(str(tmp_path / "scan16.tif"), "scan16.tif", "scan16", str(tmp_path / "work"))
    out = np.asarray(Image.open(tif["png_path"]))
    assert out.dtype == np.uint8 and abs(int(out.max()) - 255) <= 1 and out.min() == 0
