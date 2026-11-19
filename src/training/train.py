# © 2026 Андрій Кобізь
 


import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.config import ROOT, get_device, set_seed
from src.data.dataset import get_dataloaders
from src.models.resnet import build_model


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    """Один прохід по loader. З optimizer - навчання, без нього - валідація.

    Повертає (середній loss на зображення, точність).
    """
    training = optimizer is not None
    model.train(training)  # в режимі train оновлюється batchnorm
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):  # на валідації градієнти не потрібні
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)
    return total_loss / seen, correct / seen


def make_optimizer(model, tcfg):
    params = [p for p in model.parameters() if p.requires_grad]
    name = tcfg["optimizer"].lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=tcfg["learning_rate"])
    if name == "adamw":
        return torch.optim.AdamW(params, lr=tcfg["learning_rate"])
    raise ValueError(f"Unknown optimizer '{tcfg['optimizer']}' (use adam or adamw)")


def train(cfg, run_name, max_batches=None):
    """Навчання, збереження чекпойнта з найкращою точністю на валідації, історія.

    Тестова вибірка тут не використовується, тільки для фінальної оцінки,
    щоб точність на тесті була чесною.
    """
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    loaders, classes = get_dataloaders(cfg)
    cfg["data"]["num_classes"] = len(classes)
    tcfg = cfg["train"]
    model = build_model(len(classes), cfg["model"]["pretrained"],
                        cfg["model"].get("freeze_backbone", False)).to(device)
    criterion = nn.CrossEntropyLoss()  # стандартна функція втрат для класифікації на багато класів
    optimizer = make_optimizer(model, tcfg)

    ckpt_path = ROOT / Path(cfg["model"]["checkpoint_path"]).parent / f"{run_name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_seconds": []}
    best_acc = -1.0
    print(f"Run '{run_name}' on {device}: {len(classes)} classes, augmentation="
          f"{cfg['data']['augmentation']}, epochs={tcfg['epochs']}, lr={tcfg['learning_rate']}")

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer, max_batches)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device, None, max_batches)
        secs = time.time() - t0
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, secs)):
            history[k].append(v)
        saved = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "config": cfg, "epoch": epoch, "val_acc": va_acc}, ckpt_path)
            saved = " *saved*"
        print(f"epoch {epoch:2d}/{tcfg['epochs']}  train_loss {tr_loss:.4f}  train_acc {tr_acc:.4f}  "
              f"val_loss {va_loss:.4f}  val_acc {va_acc:.4f}  {secs:.0f}s{saved}")

    metrics_dir = ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    hist_path = metrics_dir / f"{run_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump({"run_name": run_name, "classes": classes, "config": cfg,
                   "device": str(device), "best_val_acc": best_acc, "history": history}, f, indent=2)
    print(f"Best val_acc {best_acc:.4f}. Checkpoint: {ckpt_path}\nHistory: {hist_path}")
    return history
