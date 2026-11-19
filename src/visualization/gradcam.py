# © 2026 Андрій Кобізь
# Grad-CAM: показує ділянки зображення, які вплинули на рішення моделі. ВАЖЛИВО!! МОДЕЛЬ НЕ ТОЧНА, ДАНИЙ ФУНКЦІОНАЛ ДОЗВОЛЯЄ
# ВІЗУАЛІЗУВАТИ РЕЗУЛЬТАТИ, АЛЕ НЕ ГАРАНТУЄ ТОЧНОСТІ. НЕ ВИКОРИСТОВУВАТИ ДЛЯ ДІАГНОСТИКИ.
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def compute_gradcam(predictor, image, class_index=None):
    image = predictor._to_image(image)
    x = predictor.transform(image).unsqueeze(0).to(predictor.device)
    stored = {}
    handle = predictor.model.layer4.register_forward_hook(lambda m, i, o: stored.update(act=o))
    try:
        with torch.enable_grad():
            logits = predictor.model(x)
            if class_index is None:
                class_index = int(logits.argmax(1))
            grads = torch.autograd.grad(logits[0, class_index], stored["act"])[0]
    finally:
        handle.remove()  # hook обов'язково прибираємо з моделі
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * stored["act"].detach()).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.size[::-1], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = (cam / cam.max()).cpu().numpy() if cam.max() > 0 else cam.cpu().numpy()
    return cam, class_index


def overlay_heatmap(image, cam, alpha=0.45):
    """Накладає теплову карту (червоне - важливе) на зображення, повертає PIL."""
    image = image.convert("RGB")
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    return Image.blend(image, Image.fromarray(heat), alpha)
