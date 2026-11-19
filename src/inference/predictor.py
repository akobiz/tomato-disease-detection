# © 2026 Андрій Кобізь
#Клас для передбачення: на вхід зображення, на вихід клас і впевненість
import io
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import ROOT, get_device, load_config
from src.data.preprocessing import get_eval_transform
from src.models.resnet import build_model


class Predictor:
  

    def __init__(self, checkpoint_path=None, device="auto"):
        if checkpoint_path is None:
            checkpoint_path = load_config()["model"]["checkpoint_path"]
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. See README")
        self.device = get_device(device)
        ckpt = torch.load(path, map_location=self.device)
        self.classes = ckpt["classes"]
        self.model = build_model(len(self.classes), pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()  # режим передбачення, batchnorm не змінюється
        self.transform = get_eval_transform(ckpt["config"]["data"]["image_size"])

    @staticmethod
    def _to_image(image):
        """Приймає PIL, шлях або байти. Якщо не читається - ValueError."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        try:
            source = io.BytesIO(image) if isinstance(image, (bytes, bytearray)) else image
            with Image.open(source) as im:
                # телефон зберігає вертикальне фото боком + прапорець EXIF, повертаємо як треба
                return ImageOps.exif_transpose(im).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError("Не вдалося прочитати зображення: файл пошкоджений або це не зображення.") from e

    def predict(self, image, top_k=3):

        x = self.transform(self._to_image(image)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        values, indices = probs.topk(min(top_k, len(self.classes)))
        top = [{"class": self.classes[i], "confidence": round(v, 4)}
               for v, i in zip(values.tolist(), indices.tolist())]
        return {"predicted_class": top[0]["class"], "confidence": top[0]["confidence"],
                "top_predictions": top}
