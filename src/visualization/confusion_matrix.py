# © 2026 Андрій Кобізь
#s
# Малюнок матриці помилок.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.labels_uk import uk_class, uk_run, use_decimal_comma


def plot_confusion_matrix(cm, class_names, run_name, out_path):
    """Рядки - справжній клас, стовпці - передбачений. Числа - кількість зображень,
    колір - частка від справжнього класу, щоб малі класи теж було видно.
    Клітинки поза діагоналлю показують, які хвороби модель плутає.
    """
    cm = np.array(cm)
    share = cm / cm.sum(axis=1, keepdims=True)
    short = [uk_class(c) for c in class_names]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(short))); ax.set_xticklabels(short, rotation=60, ha="right")
    ax.set_yticks(range(len(short))); ax.set_yticklabels(short)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("передбачений клас"); ax.set_ylabel("справжній клас")
    ax.set_title(f"Матриця помилок, тестова вибірка ({uk_run(run_name)})")
    cb = fig.colorbar(im, ax=ax, label="частка від справжнього класу")
    use_decimal_comma(cb.ax.yaxis)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
