# © 2026 Андрій Кобізь
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from src.config import ROOT, load_config
from src.inference.predictor import Predictor

CKPT = ROOT / load_config()["model"]["checkpoint_path"]
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="trained checkpoint not available")


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_model_loads(predictor):
    assert len(predictor.classes) == 10


def test_prediction_has_expected_fields(predictor):
    result = predictor.predict(Image.new("RGB", (256, 256), (40, 120, 40)), top_k=3)
    assert set(result) == {"predicted_class", "confidence", "top_predictions"}
    assert result["predicted_class"] in predictor.classes
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_predictions"]) == 3
    assert result["top_predictions"][0]["class"] == result["predicted_class"]


def test_accepts_bytes_and_known_image(predictor):
    split = json.load(open(ROOT / load_config()["data"]["split_file"]))
    rel, label = split["test"][0]
    path = ROOT / load_config()["data"]["dataset_path"] / rel
    from_path = predictor.predict(path)
    from_bytes = predictor.predict(path.read_bytes())
    assert from_path == from_bytes
    assert from_path["predicted_class"] == predictor.classes[label]


def test_invalid_image_rejected(predictor):
    with pytest.raises(ValueError):
        predictor.predict(b"this is not an image")


def test_missing_checkpoint():
    with pytest.raises(FileNotFoundError):
        Predictor("models/checkpoints/does_not_exist.pt")


def test_gradcam_map_shape_and_range(predictor):
    from src.visualization.gradcam import compute_gradcam

    cam, idx = compute_gradcam(predictor, Image.new("RGB", (90, 70), (40, 120, 40)))
    assert cam.shape == (70, 90)  # same height/width as the input image
    assert 0.0 <= cam.min() and cam.max() <= 1.0
    assert 0 <= idx < len(predictor.classes)
    assert len(predictor.model.layer4._forward_hooks) == 0  # hook was removed


def test_exif_orientation_applied_to_bytes(predictor):
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    Image.new("RGB", (60, 40), (30, 140, 30)).save(buf, format="JPEG", exif=exif)
    assert predictor._to_image(buf.getvalue()).size == (40, 60)
