# © 2026 Андрій Кобізь
from torchvision import transforms

# середнє і std ImageNet. ResNet18 вчилась на так нормалізованих картинках,
# тому наші треба нормалізувати так cамщо
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform(image_size=224):
    """Обробка для валідації, тесту і передбачення (без випадковості)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_transform(image_size=224, augment=False):


    if not augment:
        return get_eval_transform(image_size)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
