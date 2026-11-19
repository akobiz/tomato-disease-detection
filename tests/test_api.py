# © 2026 Андрій Кобізь
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.config import ROOT, load_config

CFG = load_config()
needs_model = pytest.mark.skipif(not (ROOT / CFG["model"]["checkpoint_path"]).exists(),
                                 reason="trained checkpoint not available")


def temp_config(tmp_path):
    # Справжній конфіг, але база і uploads у тимчасовій папці.
    cfg = load_config()
    cfg["api"] = {**cfg["api"], "db_path": str(tmp_path / "test.db"), "uploads_dir": str(tmp_path / "uploads")}
    return cfg


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    cfg = temp_config(tmp_path_factory.mktemp("api"))
    mp.setattr("src.api.main.load_config", lambda: cfg)
    with TestClient(app) as c:  # "with" runs startup (model loading, DB creation)
        yield c
    mp.undo()


def png_bytes(size=(64, 64), color=(30, 140, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@needs_model
def test_predict_valid_image(client):
    r = client.post("/predict", files={"file": ("leaf.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"predicted_class", "confidence", "top_predictions"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_predictions"]) == CFG["api"]["top_k"]
    assert set(body["top_predictions"][0]) == {"class", "confidence"}


@needs_model
def test_predict_real_test_image(client):
    split = json.load(open(ROOT / CFG["data"]["split_file"]))
    rel, label = split["test"][0]
    path = ROOT / CFG["data"]["dataset_path"] / rel
    r = client.post("/predict", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["predicted_class"] == split["classes"][label]


@needs_model
def test_rejects_non_image_content_type(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


@needs_model
def test_rejects_corrupted_image(client):
    r = client.post("/predict", files={"file": ("fake.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 400


@needs_model
def test_rejects_empty_file(client):
    r = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


@needs_model
def test_rejects_too_large_file(client):
    big = b"0" * (CFG["api"]["max_upload_mb"] * 1024 * 1024 + 10)
    r = client.post("/predict", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413


def test_missing_file_field(client):
    assert client.post("/predict").status_code == 422


@needs_model
def test_prediction_saved_and_listed(client):
    before = len(client.get("/predictions").json())
    r = client.post("/predict", files={"file": ("my leaf.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    history = client.get("/predictions").json()
    assert len(history) == before + 1
    newest = history[0]
    assert newest["filename"] == "my leaf.png"
    assert newest["predicted_class"] == r.json()["predicted_class"]
    assert newest["confidence"] == r.json()["confidence"]
    img = client.get(newest["image_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


@needs_model
def test_rejected_upload_is_not_saved(client):
    before = len(client.get("/predictions").json())
    client.post("/predict", files={"file": ("fake.jpg", b"not really a jpeg", "image/jpeg")})
    assert len(client.get("/predictions").json()) == before


def test_predictions_limit_validation(client):
    assert client.get("/predictions?limit=0").status_code == 422
    assert client.get("/predictions?limit=1").status_code == 200


def test_uploads_rejects_bad_names(client):
    assert client.get("/uploads/..%2F..%2Fconfig%2Fconfig.yaml").status_code == 404
    assert client.get("/uploads/" + "0" * 32 + ".png").status_code == 404


def test_health_and_predict_without_model(monkeypatch, tmp_path):
    def no_model(*args, **kwargs):
        raise FileNotFoundError("checkpoint missing")

    cfg = temp_config(tmp_path)
    monkeypatch.setattr("src.api.main.load_config", lambda: cfg)
    monkeypatch.setattr("src.api.main.Predictor", no_model)
    saved_state = dict(app.state._state)  # lifespan overwrites the global app.state; restore it afterwards
    try:
        with TestClient(app) as c:
            assert c.get("/health").json() == {"status": "ok", "model_loaded": False}
            r = c.post("/predict", files={"file": ("leaf.png", png_bytes(), "image/png")})
            assert r.status_code == 503
            assert c.post("/gradcam", files={"file": ("leaf.png", png_bytes(), "image/png")}).status_code == 503
    finally:
        app.state._state.clear()
        app.state._state.update(saved_state)


def test_index_page_and_static_files(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "/static/app.js" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/../config/config.yaml").status_code == 404


@needs_model
def test_gradcam_returns_png_of_same_size(client):
    r = client.post("/gradcam", files={"file": ("leaf.png", png_bytes((80, 60)), "image/png")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (80, 60)


@needs_model
def test_gradcam_is_not_saved_and_rejects_bad_files(client):
    before = len(client.get("/predictions").json())
    client.post("/gradcam", files={"file": ("leaf.png", png_bytes(), "image/png")})
    assert len(client.get("/predictions").json()) == before
    assert client.post("/gradcam", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415
    assert client.post("/gradcam", files={"file": ("x.jpg", b"bad", "image/jpeg")}).status_code == 400


@needs_model
def test_predict_still_works_after_gradcam(client):
    """Grad-CAM не повинен лишати hook на моделі."""
    files = {"file": ("a.png", png_bytes(), "image/png")}
    first = client.post("/predict", files=files)
    assert client.post("/gradcam", files=files).status_code == 200
    second = client.post("/predict", files=files)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def jpeg_with_orientation(size=(60, 40), orientation=6):
    exif = Image.Exif()
    exif[0x0112] = orientation  # 6 = "rotate 90 degrees to display"
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 140, 30)).save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


@needs_model
def test_exif_orientation_is_applied(client):
    r = client.post("/gradcam", files={"file": ("phone.jpg", jpeg_with_orientation(), "image/jpeg")})
    assert r.status_code == 200
    assert Image.open(io.BytesIO(r.content)).size == (40, 60)  # width/height swapped by orientation 6


@needs_model
def test_delete_history_item_removes_row_and_file(client):
    client.post("/predict", files={"file": ("del me.png", png_bytes(), "image/png")})
    item = client.get("/predictions").json()[0]
    assert client.get(item["image_url"]).status_code == 200
    assert client.delete(f"/predictions/{item['id']}").status_code == 204
    assert all(x["id"] != item["id"] for x in client.get("/predictions").json())
    assert client.get(item["image_url"]).status_code == 404
    assert client.delete(f"/predictions/{item['id']}").status_code == 404


@needs_model
def test_csv_export_and_formula_neutralized(client):
    client.post("/predict", files={"file": ("=SUM(A1).png", png_bytes(), "image/png")})
    r = client.get("/predictions.csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    lines = r.text.lstrip("\ufeff").splitlines()
    assert lines[0] == "id,created_at_utc,filename,predicted_class,confidence"
    assert any("'=SUM(A1).png" in line for line in lines)
    assert not any(line.split(",")[2].startswith("=") for line in lines[1:])


@needs_model
def test_error_messages_are_ukrainian(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert "Використовуйте JPEG або PNG" in r.json()["detail"]
    r = client.post("/predict", files={"file": ("fake.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.json()["detail"].startswith("Не вдалося прочитати зображення")
    r = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert r.json()["detail"] == "Файл порожній."
