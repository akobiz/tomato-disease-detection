# © 2026 Андрій Кобізь
import hashlib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def discover_samples(dataset_path, class_prefix):


    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}. See README (Dataset section)."
        )
    class_dirs = sorted(d for d in dataset_path.iterdir()
                        if d.is_dir() and d.name.startswith(class_prefix))
    if not class_dirs:
        raise ValueError(f"No class folders starting with '{class_prefix}' in {dataset_path}")
    classes = [d.name for d in class_dirs]
    samples = []  # (шлях відносно dataset_path, номер класу)
    seen = set()
    n_duplicates = 0
    for idx, d in enumerate(class_dirs):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            # в PlantVillage є однакові файли з різними іменами. Якщо копія потрапить і в train,
            # і в test, точність на тесті буде завищена, тому лишаємо одну
            digest = hashlib.md5(f.read_bytes()).hexdigest()
            if digest in seen:
                n_duplicates += 1
                continue
            seen.add(digest)
            samples.append((f"{d.name}/{f.name}", idx))
    print(f"Found {len(samples)} unique images; removed {n_duplicates} exact duplicates")
    return classes, samples


def make_split(samples, train_frac, val_frac, test_frac, seed):
  #
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1")
    labels = [y for _, y in samples]
    train, rest, _, rest_labels = train_test_split(
        samples, labels, train_size=train_frac, stratify=labels, random_state=seed)
    val_share = val_frac / (val_frac + test_frac)
    val, test = train_test_split(
        rest, train_size=val_share, stratify=rest_labels, random_state=seed)
    return {"train": train, "val": val, "test": test}


def build_and_save_split(cfg):
    d = cfg["data"]
    classes, samples = discover_samples(ROOT / d["dataset_path"], d["class_prefix"])
    split = make_split(samples, d["train_frac"], d["val_frac"], d["test_frac"], cfg["seed"])
    out = ROOT / d["split_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"seed": cfg["seed"], "classes": classes, **split}, f)
    return out


def load_split(cfg):
    path = ROOT / cfg["data"]["split_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. See README")
    with open(path) as f:
        return json.load(f)
