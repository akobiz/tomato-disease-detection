# © 2026 Андрій Кобізь
#Історія передбачень у SQLite (тільки стандартна бібліотека
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    image_path TEXT
)
"""


def _connect(db_path):
    return sqlite3.connect(db_path)  # нове з'єднання на кожен виклик, так простіше і безпечно для потоків


def init_db(db_path):
    """Створює файл бази і таблицю, якщо їх ще нема."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(SCHEMA)


def save_prediction(db_path, filename, predicted_class, confidence, image_path=None):
#Додає запис і повертає його id. Параметри через ?, щоб не було SQL-ін'єкцій.
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            "INSERT INTO predictions (filename, predicted_class, confidence, created_at, image_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (filename, predicted_class, confidence, created_at, image_path))
        return cur.lastrowid


def get_recent(db_path, limit=20):
    #  Повертає записи від нових до старих (список словників).
    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, filename, predicted_class, confidence, created_at, image_path "
            "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def delete_prediction(db_path, prediction_id):
    #Видаляє запис. Повертає image_path (щоб видалити файл) або None, якщо такого id нема.
    with closing(_connect(db_path)) as conn, conn:
        row = conn.execute("SELECT image_path FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        return row[0] or ""
