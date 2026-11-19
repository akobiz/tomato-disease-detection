# © 2026 Андрій Кобізь
#   Метрики класифікації (scikit-learn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, class_names):
    """Accuracy, precision, recall, F1 (macro і weighted), по класах, матриця помилок.

    Precision - з передбачених як клас X скільки справді X.
    Recall - зі справжніх X скільки знайдено.
    F1 - середнє гармонічне precision і recall.
    Macro - просте середнє по класах, видно слабкі малі класи.
    Weighted - середнє з вагою за розміром класу, близьке до accuracy.
    """
    labels = list(range(len(class_names)))
    result = {"accuracy": accuracy_score(y_true, y_pred), "n_samples": len(y_true)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0)
        result[avg] = {"precision": p, "recall": r, "f1": f}
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    result["per_class"] = {
        name: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(support[i])}
        for i, name in enumerate(class_names)
    }
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return result
