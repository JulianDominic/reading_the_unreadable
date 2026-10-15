import cv2
import numpy as np
import pytest

from newspaper_pipeline.stages.preprocess import estimate_skew, find_paper, preprocess_page
from newspaper_pipeline.transforms import PageTransform, crop_matrix, rotation_matrix


@pytest.mark.parametrize("angle", [-3.0, -0.7, 0.0, 1.3, 4.5])
def test_rotation_matrix_matches_opencv(angle):
    w, h = 640, 900
    matrix, new_w, new_h = rotation_matrix(angle, w, h)
    expected = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    expected[0, 2] += new_w / 2 - w / 2
    expected[1, 2] += new_h / 2 - h / 2
    np.testing.assert_allclose(matrix, expected, atol=1e-9)
    # Every corner of the source lands inside the expanded canvas
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    mapped = corners @ matrix[:, :2].T + matrix[:, 2]
    assert mapped.min() >= -1e-6 and mapped[:, 0].max() <= new_w + 1e-6 and mapped[:, 1].max() <= new_h + 1e-6


def test_composed_transform_round_trip():
    transform = PageTransform.identity(5000, 7000).then(crop_matrix(120, 340), (4700, 6500))
    matrix, nw, nh = rotation_matrix(1.7, 4700, 6500)
    transform = transform.then(matrix, (nw, nh))
    rng = np.random.default_rng(1)
    points = rng.uniform([0, 0], [5000, 7000], size=(200, 2))
    np.testing.assert_allclose(transform.points_to_original(transform.points_to_processed(points)), points, atol=1e-6)


def test_json_round_trip(tmp_path):
    transform = PageTransform.identity(100, 200).then(crop_matrix(5, 7), (90, 180))
    transform.angle, transform.content_box = 0.5, (5, 7, 95, 187)
    transform.save(tmp_path / "t.json")
    loaded = PageTransform.load(tmp_path / "t.json")
    np.testing.assert_allclose(loaded.matrix, transform.matrix)
    assert loaded.original_size == (100, 200) and loaded.angle == 0.5 and tuple(loaded.content_box) == (5, 7, 95, 187)


def test_box_to_original_clamps_and_offsets():
    transform = PageTransform.identity(1000, 1000)
    assert transform.box_to_original((10, 20, 30, 40), offset=(100, 200)) == (110, 220, 20, 20)
    x, y, w, h = transform.box_to_original((-50, -50, 2000, 2000))
    assert (x, y, w, h) == (0, 0, 1000, 1000)


def synthetic_page(width=1600, height=2200, angle=0.0, border=0, dots=()):
    """Text-like rows of dark bars on light paper, optionally rotated and placed on a dark scanner bed."""
    page = np.full((height, width), 235, np.uint8)
    rng = np.random.default_rng(0)
    for y in range(120, height - 120, 34):
        x = 100
        while x < width - 160:
            word = int(rng.integers(30, 120))
            cv2.rectangle(page, (x, y), (x + word, y + 14), 30, -1)
            x += word + int(rng.integers(14, 26))
    for (x, y) in dots:
        cv2.circle(page, (x, y), 6, 0, -1)
    if angle:
        m = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        page = cv2.warpAffine(page, m, (width, height), borderValue=235)
    if border:
        bed = np.full((height + 2 * border, width + 2 * border), 25, np.uint8)
        bed[border:border + height, border:border + width] = page
        page = bed
    return page


@pytest.mark.parametrize("angle", [-2.0, 1.2])
def test_estimate_skew_recovers_rotation(angle):
    page = synthetic_page(angle=angle)
    assert estimate_skew(page) == pytest.approx(-angle, abs=0.1)


def test_find_paper_ignores_borderless_page():
    assert find_paper(synthetic_page()) is None


def test_find_paper_detects_scanner_bed():
    page = synthetic_page(border=150)
    box, mask = find_paper(page)
    # the box is pulled slightly inside the paper so no scanner bed survives
    assert 150 <= box[0] <= 165 and 150 <= box[1] <= 165
    assert 150 + 1600 - 15 <= box[2] <= 150 + 1600 and 150 + 2200 - 15 <= box[3] <= 150 + 2200
    assert mask[10, 10] == 0 and mask[1000, 800] == 255


def test_preprocess_maps_points_back_to_original(tmp_path):
    """A dot found on the processed page must map back onto the same dot in the original scan."""
    dots_on_paper = [(300, 400), (1300, 1900), (800, 1100)]
    border = 120
    # Rotation is applied around the paper centre, so recompute where the dots land in the original image
    m = cv2.getRotationMatrix2D((1600 / 2, 2200 / 2), 1.5, 1.0)
    original_dots = [tuple(np.array([x, y, 1.0]) @ m.T + border) for x, y in dots_on_paper]
    page = synthetic_page(angle=1.5, border=border, dots=dots_on_paper)
    src = tmp_path / "page.png"
    cv2.imwrite(str(src), page)

    result = preprocess_page("p", str(src), str(tmp_path / "work"), enhance_page=False)
    assert result["angle"] == pytest.approx(-1.5, abs=0.1)
    assert result["content_box"][0] > 0  # the scanner bed was cropped away

    processed = cv2.imread(result["processed_path"], cv2.IMREAD_GRAYSCALE)
    transform = PageTransform.load(result["transform_path"])
    for ox, oy in original_dots:
        px, py = transform.points_to_processed(np.array([[ox, oy]]))[0]
        # The dot is the darkest thing near the predicted spot on the processed page
        window = processed[int(py) - 12:int(py) + 13, int(px) - 12:int(px) + 13]
        _, _, _, (wx, wy) = cv2.minMaxLoc(cv2.GaussianBlur(255 - window, (9, 9), 0))
        found = np.array([[int(px) - 12 + wx, int(py) - 12 + wy]], dtype=float)
        back = transform.points_to_original(found)[0]
        assert np.hypot(back[0] - ox, back[1] - oy) < 3.0
