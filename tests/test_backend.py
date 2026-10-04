from io import BytesIO
from html.parser import HTMLParser

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.app import app
from backend.analysis_service import HairAnalysisService
from backend.recommendations import COLOR_PALETTE, recommend_colors, recommend_haircuts


client = TestClient(app)


def test_segmentation_preprocessing_matches_training_normalization():
    image_array = np.zeros((16, 16, 3), dtype=np.uint8)
    image_array[:, :, 0] = 255
    image_array[:, :, 1] = 128
    tensor = HairAnalysisService.prepare_segmentation_tensor(Image.fromarray(image_array))

    expected = np.array(
        [
            (1.0 - 0.485) / 0.229,
            (128 / 255 - 0.456) / 0.224,
            (0.0 - 0.406) / 0.225,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(tensor.shape, (1, 3, 224, 224))
    np.testing.assert_allclose(tensor[0, :, 0, 0].numpy(), expected, rtol=1e-5)


def test_virtual_shades_change_brightness_and_color():
    image = np.full((16, 16, 3), [80, 65, 55], dtype=np.uint8)
    mask = np.ones((16, 16), dtype=np.uint8)
    caramel = HairAnalysisService.apply_hair_color(image, mask, (185, 121, 67))
    platinum = HairAnalysisService.apply_hair_color(image, mask, (231, 225, 214))

    assert not np.array_equal(caramel, platinum)
    assert platinum.mean() > caramel.mean() + 5


def test_red_shade_does_not_shift_blue_hair_toward_green():
    blue_hair = np.full((16, 16, 3), [24, 31, 48], dtype=np.uint8)
    mask = np.ones((16, 16), dtype=np.uint8)
    auburn = HairAnalysisService.apply_hair_color(blue_hair, mask, (138, 51, 36))

    center_pixel = auburn[8, 8]
    assert center_pixel[0] > center_pixel[1]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["project"] == "AI Hair Intelligence"


def test_haircut_and_color_recommendations_match_supported_palette():
    curly_cuts = recommend_haircuts("Curly Hair")
    straight_cuts = recommend_haircuts("Straight Hair")
    curly_colors = recommend_colors("Curly Hair", "Medium Brown")
    straight_colors = recommend_colors("Straight Hair", "Medium Brown")

    assert curly_cuts != straight_cuts
    assert {item["name"] for item in curly_colors} != {item["name"] for item in straight_colors}
    assert all(item["name"] in COLOR_PALETTE for item in curly_colors + straight_colors)
    assert len(COLOR_PALETTE) >= 40


def test_overview_exposes_shared_color_palette():
    response = client.get("/api/v1/overview")

    assert response.status_code == 200
    palette = response.json()["color_palette"]
    assert len(palette) == len(COLOR_PALETTE)
    assert {item["name"] for item in palette} >= {"Mushroom Brown", "Champagne Blonde", "Blue Black"}


def test_frontend_files_are_served():
    page = client.get("/")
    stylesheet = client.get("/styles.css")
    script = client.get("/app.js")

    assert page.status_code == 200
    assert 'id="analysis-form"' in page.text
    assert all(f'id="page-{page_name}"' in page.text for page_name in ["type", "mapping", "color", "complete"])
    assert 'id="region"' in page.text
    assert 'id="target-color"' in page.text
    assert 'id="source-name"' in page.text
    assert 'id="condition-shortlist"' in page.text
    assert 'id="haircut-recommendations"' in page.text
    assert 'id="color-recommendations"' in page.text
    assert 'id="camera-toggle"' in page.text
    assert 'id="camera-video"' in page.text
    assert 'id="capture-photo"' in page.text
    assert "MODEL'S TOP MATCH" in page.text

    class PageSectionParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.section_stack = []
            self.parents = {}

        def handle_starttag(self, tag, attrs):
            if tag == "section":
                section_id = dict(attrs).get("id")
                if section_id and section_id.startswith("page-"):
                    self.parents[section_id] = next(
                        (parent for parent in reversed(self.section_stack) if parent),
                        None,
                    )
                self.section_stack.append(section_id)

        def handle_endtag(self, tag):
            if tag == "section" and self.section_stack:
                self.section_stack.pop()

    parser = PageSectionParser()
    parser.feed(page.text)
    assert set(parser.parents) == {"page-type", "page-mapping", "page-color", "page-complete"}
    assert set(parser.parents.values()) == {"results"}

    assert stylesheet.status_code == 200
    assert "--forest" in stylesheet.text
    assert script.status_code == 200
    assert 'fetch(`/api/v1/analyze?${parameters}`' in script.text
    assert "new URLSearchParams({ region: regionSelect.value })" in script.text
    assert "?${parameters}" in script.text
    assert "navigator.mediaDevices.getUserMedia" in script.text


def test_analyze_endpoint_accepts_upload():
    image_array = np.zeros((120, 160, 3), dtype=np.uint8)
    image_array[:, :, 0] = 220
    image_array[:, :, 1] = 180
    image_array[:, :, 2] = 150

    buffer = BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
        params={"region": "Ends", "target_color": "Caramel"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "hair_type" in payload
    assert "disease" in payload
    assert "color" in payload
    assert "quality" in payload
    assert "hair_mask" in payload
    assert "region_masks" in payload
    assert isinstance(payload["hair_mask"], list)
    assert set(payload["region_masks"].keys()) >= {"Scalp", "Middle", "Ends"}
    assert payload["current_region"] == "Ends"
    assert payload["target_color"] == "Caramel"
    assert payload["preview_image"]
    assert set(payload["region_colors"]) >= {"Whole Hair", "Scalp", "Middle", "Ends"}
    assert payload["haircut_recommendations"]
    assert payload["recommended_colors"]
    assert all(item["name"] in COLOR_PALETTE for item in payload["recommended_colors"])
