# © 2026 Андрій Кобізь
# Оці нка збереженого чекпойнта на тестовій вибірці.
import torch

from src.config import ROOT, get_device
from src.data.dataset import get_dataloaders
from src.models.resnet import build_model
from src.training.metrics import compute_metrics


def load_checkpoint(path, device):
    path = ROOT / path if not str(path).startswith("/") else path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. See README")
    return torch.load(path, map_location=device)


def evaluate_checkpoint(ckpt_path):
    """Завантажує модель з train.py і рахує метрики на тестовій вибірці.

    Конфіг береться з чекпойнта, тобто ті самі налаштування, що при навчанні.
    """
    cfg_device = "auto"
    device = get_device(cfg_device)
    ckpt = load_checkpoint(ckpt_path, device)
    cfg, classes = ckpt["config"], ckpt["classes"]
    model = build_model(len(classes), pretrained=False).to(device)  # ваги беремо з чекпойнта
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loaders, _ = get_dataloaders(cfg)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loaders["test"]:
            y_pred += model(x.to(device)).argmax(1).cpu().tolist()
            y_true += y.tolist()
    result = compute_metrics(y_true, y_pred, classes)
    result.update({"classes": classes, "checkpoint_epoch": ckpt["epoch"],
                   "checkpoint_val_acc": ckpt["val_acc"], "device": str(device)})
    return result
