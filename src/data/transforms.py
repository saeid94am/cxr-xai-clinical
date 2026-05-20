import torchvision.transforms as T

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def train_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])


def val_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])
