# © 2026 Андрій Кобізь
import sqlite3

from src.api.database import delete_prediction, get_recent, init_db, save_prediction


def test_init_creates_table(tmp_path):
    db = tmp_path / "h.db"
    init_db(db)
    init_db(db)  # calling twice must be safe
    with sqlite3.connect(db) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(predictions)")]
    assert cols == ["id", "filename", "predicted_class", "confidence", "created_at", "image_path"]


def test_save_and_read_history_newest_first(tmp_path):
    db = tmp_path / "h.db"
    init_db(db)
    first = save_prediction(db, "a.jpg", "Tomato___healthy", 0.9, "x.jpg")
    second = save_prediction(db, "b.jpg", "Tomato___Late_blight", 0.8)
    assert second == first + 1
    rows = get_recent(db)
    assert [r["filename"] for r in rows] == ["b.jpg", "a.jpg"]
    assert rows[1]["predicted_class"] == "Tomato___healthy"
    assert rows[1]["confidence"] == 0.9
    assert rows[1]["image_path"] == "x.jpg"
    assert rows[0]["image_path"] is None
    assert rows[0]["created_at"].endswith("+00:00")


def test_limit(tmp_path):
    db = tmp_path / "h.db"
    init_db(db)
    for i in range(5):
        save_prediction(db, f"{i}.jpg", "Tomato___healthy", 0.5)
    assert len(get_recent(db, limit=3)) == 3


def test_sql_injection_is_stored_as_text(tmp_path):
    db = tmp_path / "h.db"
    init_db(db)
    evil = "x'); DROP TABLE predictions;--"
    save_prediction(db, evil, "Tomato___healthy", 0.5)
    assert get_recent(db)[0]["filename"] == evil


def test_delete_prediction(tmp_path):
    db = tmp_path / "h.db"
    init_db(db)
    keep = save_prediction(db, "keep.jpg", "Tomato___healthy", 0.5, "k.jpg")
    drop = save_prediction(db, "drop.jpg", "Tomato___healthy", 0.5, "d.jpg")
    assert delete_prediction(db, drop) == "d.jpg"
    assert [r["id"] for r in get_recent(db)] == [keep]
    assert delete_prediction(db, drop) is None  # already gone
