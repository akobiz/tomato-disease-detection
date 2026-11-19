# © 2026 Андрій Кобізь
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import ROOT
from src.data.preprocessing import get_eval_transform, get_train_transform
from src.data.split_dataset import load_split


class LeafDataset(Dataset):

    def __init__(self, samples, root, transform):
        self.samples = samples
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rel_path, label = self.samples[i]
        image = Image.open(self.root / rel_path).convert("RGB")
        return self.transform(image), label


def get_dataloaders(cfg):
 
    split = load_split(cfg)
    root = ROOT / cfg["data"]["dataset_path"]
    size = cfg["data"]["image_size"]
    tcfg = cfg["train"]
    transforms_ = {
        "train": get_train_transform(size, cfg["data"]["augmentation"]),
        "val": get_eval_transform(size),
        "test": get_eval_transform(size),
    }
    generator = torch.Generator().manual_seed(cfg["seed"])  # однакове перемішування при кожному запуску
    loaders = {}
    for name in ("train", "val", "test"):
        ds = LeafDataset(split[name], root, transforms_[name])
        loaders[name] = DataLoader(
            ds, batch_size=tcfg["batch_size"], shuffle=(name == "train"),
            num_workers=tcfg["num_workers"], generator=generator if name == "train" else None)
    return loaders, split["classes"]
