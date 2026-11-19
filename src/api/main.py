# © 2026 Андрій Кобізь
import csv
import io
import logging
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from src.api.database import delete_prediction, get_recent, init_db, save_prediction
from src.api.schemas import HealthResponse, HistoryItem, PredictionResponse
from src.config import ROOT, load_config
from src.inference.predictor import Predictor
from src.visualization.gradcam import compute_gradcam, overlay_heatmap

logger = logging.getLogger("api")
ALLOWED_TYPES = {"image/jpeg", "image/png"}
EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png"}  # розширення беремо з вмісту файлу, а не з імені
SAVED_NAME = re.compile(r"^[0-9a-f]{32}\.(jpg|png)$")


@asynccontextmanager
async def lifespan(app):

    app.state.cfg = load_config()
    app.state.lock = threading.Lock()  # модель одна на всі запити, тому по черзі
    app.state.db_path = ROOT / app.state.cfg["api"]["db_path"]
    app.state.uploads_dir = ROOT / app.state.cfg["api"]["uploads_dir"]
    app.state.uploads_dir.mkdir(parents=True, exist_ok=True)
    init_db(app.state.db_path)
    app.state.predictor = None
    try:
        app.state.predictor = Predictor()
    except FileNotFoundError as e:
        logger.error("%s", e)  # сервер все одно стартує, /health покаже model_loaded=false
    yield


app = FastAPI(title="Crop disease detection", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")  # css і js


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.exception_handler(Exception)
async def unhandled_error(request, exc):
    logger.exception("Unhandled error on %s", request.url.path)  # повний трейсбек у лог
    return JSONResponse(status_code=500, content={"detail": "Внутрішня помилка сервера."})


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_loaded": app.state.predictor is not None}


def read_valid_image(file):
    #Перевірка файлу: тип, розмір, чи це справді JPEG/PNG. Повертає (байти, розширення
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Непідтримуваний тип файлу ({file.content_type}). Використовуйте JPEG або PNG.")

    api_cfg = app.state.cfg["api"]
    limit = api_cfg["max_upload_mb"] * 1024 * 1024
    data = file.file.read(limit + 1)  # читаємо на 1 байт більше ліміту, щоб побачити що файл завеликий
    if len(data) > limit:
        raise HTTPException(413, f"Файл більший за {api_cfg['max_upload_mb']} МБ.")
    if not data:
        raise HTTPException(400, "Файл порожній.")

    try:  
        with Image.open(io.BytesIO(data)) as im:
            extension = EXTENSIONS.get(im.format)
    except (UnidentifiedImageError, OSError):
        extension = None
    if extension is None:
        raise HTTPException(400, "Не вдалося прочитати зображення: файл пошкоджений або це не JPEG/PNG.")
    return data, extension


@app.post("/predict", response_model=PredictionResponse, response_model_by_alias=True)
def predict(file: UploadFile = File(...)):
    #Перевіряє файл, запускає модель, повертає клас, впевненість і топ-k.
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(503, "Модель не завантажена. Спочатку навчіть модель.")
    data, extension = read_valid_image(file)
    api_cfg = app.state.cfg["api"]

    try:
        with app.state.lock:
            result = predictor.predict(data, top_k=api_cfg["top_k"])
    except ValueError as e:
        raise HTTPException(400, str(e))

    # зберігаємо картинку під випадковим ім'ям (не ім'ям користувача) для історії
    saved_name = uuid.uuid4().hex + extension
    (app.state.uploads_dir / saved_name).write_bytes(data)
    save_prediction(app.state.db_path, Path(file.filename or "upload").name[:255],
                    result["predicted_class"], result["confidence"], saved_name)
    return result


@app.post("/gradcam", responses={200: {"content": {"image/png": {}}}})
def gradcam(file: UploadFile = File(...)):
    #PNG з картою Grad-CAM поверх зображення. В історію не записується.
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(503, "Модель не завантажена. Спочатку навчіть модель.")
    data, _ = read_valid_image(file)
    with Image.open(io.BytesIO(data)) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")  # та сама орієнтація, що й для моделі
    with app.state.lock:  # Grad-CAM тимчасово ставить hook на модель
        cam, _ = compute_gradcam(predictor, image)
    buffer = io.BytesIO()
    overlay_heatmap(image, cam).save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png")


@app.get("/predictions", response_model=list[HistoryItem])
def predictions(limit: int = Query(20, ge=1, le=100)):
    #Остання історія, нові записи першими.
    return [{**row, "image_url": f"/uploads/{row['image_path']}" if row["image_path"] else None}
            for row in get_recent(app.state.db_path, limit)]


@app.delete("/predictions/{prediction_id}", status_code=204)
def delete_history_item(prediction_id: int):
    #Видалення запису історії і його картинки.
    image_path = delete_prediction(app.state.db_path, prediction_id)
    if image_path is None:
        raise HTTPException(404, "Запис не знайдено.")
    if SAVED_NAME.match(image_path):  # видаляємо тільки файли з нашими іменами
        (app.state.uploads_dir / image_path).unlink(missing_ok=True)


def csv_safe(value):
    #Excel сприймає текст з = + - @ на початку як формулу, тому додаємо апостроф.
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


@app.get("/predictions.csv")
def predictions_csv():
    #Вс   я історія у форматі CSV."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "created_at_utc", "filename", "predicted_class", "confidence"])
    for row in reversed(get_recent(app.state.db_path, limit=100000)):  # старі першими
        writer.writerow([row["id"], row["created_at"], csv_safe(row["filename"]),
                         row["predicted_class"], row["confidence"]])
    return Response("\ufeff" + buffer.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="predictions.csv"'})


@app.get("/uploads/{name}")
def uploaded_image(name: str):
    """Віддає збережену картинку. Тільки імена, які створив сервер (32 hex + .jpg/.png)."""
    path = app.state.uploads_dir / name
    if not SAVED_NAME.match(name) or not path.is_file():
        raise HTTPException(404, "Зображення не знайдено.")
    return FileResponse(path)
