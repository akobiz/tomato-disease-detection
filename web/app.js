// © 2026 Андрій Кобізь
const MAX_MB = 10;          // як api.max_upload_mb на сервері
const MAX_SIDE = 1024;      // більші фото стискаємо в браузері
const LOW_CONFIDENCE = 0.8; // нижче - показуємо попередження (поріг виміряно на тесті PlantVillage)
const $ = (id) => document.getElementById(id);

// українські назви класів
const NAMES_UK = {
  "Bacterial_spot": "Бактеріальна плямистість",
  "Early_blight": "Альтернаріоз (рання плямистість)",
  "Late_blight": "Фітофтороз (пізня гниль)",
  "Leaf_Mold": "Кладоспоріоз (пліснявість листя)",
  "Septoria_leaf_spot": "Септоріоз",
  "Spider_mites Two-spotted_spider_mite": "Павутинний кліщ",
  "Target_Spot": "Цільова плямистість",
  "Tomato_Yellow_Leaf_Curl_Virus": "Вірус жовтого скручування листя",
  "Tomato_mosaic_virus": "Вірус мозаїки томата",
  "healthy": "Здоровий листок",
};

function englishName(name) {
  return name.replace("Tomato___", "").replace(/_/g, " ");
}

function ukName(name) {
  return NAMES_UK[name.replace("Tomato___", "")] || englishName(name);
}

function percent(value) {
  return (value * 100).toFixed(1) + "%";
}

function show(el, visible) {
  el.classList.toggle("hidden", !visible);
}

function showError(message) {
  $("error").textContent = message;
  show($("error"), Boolean(message));
}

const analyzeBtn = $("analyze-btn");
let currentFile = null;  // файл, який відправляємо на сервер (можливо стиснутий)
let objectUrls = [];     // URL прев'ю і карти, звільняємо при новому фото

function newObjectUrl(blob) {
  const url = URL.createObjectURL(blob);
  objectUrls.push(url);
  return url;
}

function releaseUrls() {
  objectUrls.forEach((u) => URL.revokeObjectURL(u));
  objectUrls = [];
}

// стискаємо великі фото (з телефона 3-8 МБ) до MAX_SIDE і перетворюємо в JPEG
// (наприклад HEIC з iPhone). Браузер при малюванні в canvas сам повертає фото за EXIF.
// Маленькі JPEG/PNG відправляємо як є, поворот зробить сервер
async function prepareImage(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    if (scale === 1 && ["image/jpeg", "image/png"].includes(file.type)) return file;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";  // інакше прозорий фон PNG стане чорним
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    if (!blob) throw new Error("toBlob failed");
    return new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" });
  } finally {
    URL.revokeObjectURL(url);
  }
}

function resetResult() {
  show($("result"), false);
  show($("gradcam-img"), false);
  show($("gradcam-wait"), true);
}

async function handleFile(file) {
  showError("");
  resetResult();
  releaseUrls();
  currentFile = null;
  analyzeBtn.disabled = true;
  show($("preview-box"), false);
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showError("Непідтримуваний тип файлу. Оберіть зображення (JPEG або PNG).");
    return;
  }
  try {
    currentFile = await prepareImage(file);
  } catch (err) {
    showError("Не вдалося прочитати зображення. Спробуйте JPEG або PNG.");
    return;
  }
  if (currentFile.size > MAX_MB * 1024 * 1024) {
    currentFile = null;
    showError(`Файл завеликий (максимум ${MAX_MB} МБ).`);
    return;
  }
  $("preview").src = newObjectUrl(currentFile);
  show($("preview-box"), true);
  analyzeBtn.disabled = false;
}

for (const id of ["file-input", "camera-input"]) {
  $(id).addEventListener("change", (e) => {
    handleFile(e.target.files[0]);
    e.target.value = "";  // щоб можна було вибрати той самий файл ще раз
  });
}

const dropzone = $("dropzone");
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("over"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("over");
  handleFile(e.dataTransfer.files[0]);
});

// fetch кидає TypeError, коли сервер недоступний
function errorText(err) {
  return err instanceof TypeError ? "немає зв'язку з сервером" : err.message;
}

async function apiError(response) {
  // у помилках API detail - текст, а у 422 - список
  const data = await response.json().catch(() => ({}));
  return typeof data.detail === "string" ? data.detail : "Некоректний запит.";
}

analyzeBtn.addEventListener("click", async () => {
  if (!currentFile) return;
  showError("");
  resetResult();
  analyzeBtn.disabled = true;
  show($("loading"), true);
  try {
    const body = new FormData();
    body.append("file", currentFile);
    const response = await fetch("/predict", { method: "POST", body });
    if (!response.ok) throw new Error(await apiError(response));
    renderResult(await response.json());
    loadHistory();
    loadGradcam(currentFile);  // у фоні, результат вже показаний
  } catch (err) {
    showError("Помилка: " + errorText(err));
  } finally {
    show($("loading"), false);
    analyzeBtn.disabled = false;
  }
});

async function loadGradcam(file) {
  try {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/gradcam", { method: "POST", body });
    if (!response.ok) throw new Error(await apiError(response));
    if (file !== currentFile) return;  // користувач вже вибрав інше фото
    $("gradcam-img").src = newObjectUrl(await response.blob());
    show($("gradcam-img"), true);
    show($("gradcam-wait"), false);
  } catch (err) {
    $("gradcam-wait").textContent = "Не вдалося побудувати карту: " + errorText(err);
  }
}

function renderResult(data) {
  $("result-class").textContent = ukName(data.predicted_class);
  $("result-confidence").textContent = percent(data.confidence);
  $("result-en").textContent = englishName(data.predicted_class);
  $("orig-img").src = $("preview").src;
  $("gradcam-wait").textContent = "Будуємо карту…";

  const low = data.confidence < LOW_CONFIDENCE;
  $("low-conf").textContent = low
    ? "Низька впевненість (менше 80 %). На тестових зображеннях PlantVillage серед таких відповідей вірними були лише близько 60 %; на інших фото надійність може бути ще нижчою."
    : "";
  show($("low-conf"), low);

  const list = $("top-list");
  list.replaceChildren();
  for (const item of data.top_predictions) {
    const li = document.createElement("li");
    li.textContent = `${ukName(item.class)} — ${percent(item.confidence)}`;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.style.width = `${item.confidence * 100}%`;
    bar.appendChild(fill);
    li.appendChild(bar);
    list.appendChild(li);
  }
  show($("result"), true);
}

async function deleteItem(id) {
  if (!confirm("Видалити цей запис з історії?")) return;
  try {
    const response = await fetch(`/predictions/${id}`, { method: "DELETE" });
    if (!response.ok && response.status !== 404) throw new Error("код відповіді " + response.status);
  } catch (err) {
    showError("Не вдалося видалити запис: " + errorText(err));
  }
  loadHistory();
}

async function loadHistory() {
  const body = $("history-body");
  try {
    const response = await fetch("/predictions?limit=20");
    if (!response.ok) throw new Error("код відповіді " + response.status);
    const rows = await response.json();
    body.replaceChildren();
    $("history-empty").textContent = "Історія порожня.";
    show($("history-empty"), rows.length === 0);
    for (const row of rows) {
      const tr = document.createElement("tr");
      const date = document.createElement("td");
      date.textContent = new Date(row.created_at).toLocaleString("uk-UA", { dateStyle: "short", timeStyle: "short" });  // в базі час UTC
      const imageCell = document.createElement("td");
      if (row.image_url) {
        const img = document.createElement("img");
        img.src = row.image_url;
        img.alt = "Завантажене зображення";
        imageCell.appendChild(img);
      }
      const cls = document.createElement("td");
      cls.textContent = ukName(row.predicted_class);
      cls.title = englishName(row.predicted_class);
      const confInline = document.createElement("div");  // на телефоні замість колонки впевненості
      confInline.className = "conf-inline";
      confInline.textContent = percent(row.confidence);
      cls.appendChild(confInline);
      const conf = document.createElement("td");
      conf.textContent = percent(row.confidence);
      const actions = document.createElement("td");
      const del = document.createElement("button");
      del.className = "del";
      del.textContent = "✕";
      del.title = "Видалити запис";
      del.setAttribute("aria-label", "Видалити запис");
      del.addEventListener("click", () => deleteItem(row.id));
      actions.appendChild(del);
      tr.append(date, imageCell, cls, conf, actions);
      body.appendChild(tr);
    }
  } catch (err) {
    $("history-empty").textContent = "Не вдалося завантажити історію: " + errorText(err);
    show($("history-empty"), true);
  }
}

loadHistory();
