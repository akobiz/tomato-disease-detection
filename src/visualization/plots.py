# © 2026 Андрій Кобізь
#Графіки навчання (loss і точність по епохах). 
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization.labels_uk import uk_run, use_decimal_comma


def plot_training_curves(history, run_name, out_dir):
    """Зберігає графіки loss і точності для train і val.

    По них видно перенавчання: train loss падає, а val loss вже не покращується.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], marker="o", label="навчальна вибірка")
    ax.plot(epochs, history["val_loss"], marker="o", label="валідаційна вибірка")
    ax.set_xlabel("епоха"); ax.set_ylabel("втрати"); ax.set_title(f"Втрати по епохах ({uk_run(run_name)})")
    use_decimal_comma(ax.yaxis)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / f"{run_name}_loss.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["val_acc"], marker="o", label="валідаційна вибірка")
    ax.plot(epochs, history["train_acc"], marker="o", linestyle="--", label="навчальна вибірка")
    ax.set_xlabel("епоха"); ax.set_ylabel("точність"); ax.set_title(f"Точність по епохах ({uk_run(run_name)})")
    use_decimal_comma(ax.yaxis)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / f"{run_name}_accuracy.png", dpi=150); plt.close(fig)
