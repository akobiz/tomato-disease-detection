# © 2026 Андрій Кобізь
# Українські підписи для графіків (методичка: підписи мовою роботи).
from matplotlib.ticker import FuncFormatter

CLASS_NAMES_UK = {
    "Bacterial_spot": "Бактеріальна плямистість",
    "Early_blight": "Альтернаріоз",
    "Late_blight": "Фітофтороз",
    "Leaf_Mold": "Кладоспоріоз",
    "Septoria_leaf_spot": "Септоріоз",
    "Spider_mites Two-spotted_spider_mite": "Павутинний кліщ",
    "Target_Spot": "Цільова плямистість",
    "Tomato_Yellow_Leaf_Curl_Virus": "Вірус жовтого скручування листя",
    "Tomato_mosaic_virus": "Вірус мозаїки томата",
    "healthy": "Здоровий листок",
}
RUN_NAMES_UK = {"baseline": "базова модель", "augmented": "з аугментацією", "repro": "повторний запуск"}
SPLIT_NAMES_UK = {"train": "навчальна", "val": "валідаційна", "test": "тестова"}


def uk_class(name):
    """Українська назва класу за назвою папки датасету."""
    key = name.replace("Tomato___", "")
    return CLASS_NAMES_UK.get(key, key.replace("_", " "))


def uk_run(run_name):
    return RUN_NAMES_UK.get(run_name, run_name)


def comma(value, _pos=None):
    """Число з десятковою комою замість крапки."""
    return f"{value:g}".replace(".", ",")


def use_decimal_comma(axis):
    axis.set_major_formatter(FuncFormatter(comma))
