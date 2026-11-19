# © 2026 Андрій Кобізь
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    # Читаю config/config.yaml, якщо його нема то тоді config.example.yaml.
    if path is None:
        path = ROOT / "config" / "config.yaml"
        if not path.exists():
            path = ROOT / "config" / "config.example.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    # Фіксую всі генератори випадкових чисел, щоб результати повторювались.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name="auto"):
    # CUDA -> MPS (Apple Silicon) -> CPU, якщо в конфігу не вказано інше.
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
