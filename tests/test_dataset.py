from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets import load_and_split_dataset, FaceDataset


DATASET_DIR = Path("/home/alimaskin/Загрузки")

dataset_files_dir = DATASET_DIR / "celebA_train_500"
split_file = dataset_files_dir / "celebA_train_split.txt"
anno_file = dataset_files_dir / "celebA_anno.txt"

datasets_split = load_and_split_dataset(dataset_files_dir, split_file, anno_file)

datasets_split["train"] = datasets_split["train"][:100]

# Определяем аугментации
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)
])

transforms = transforms.Compose([
    # transforms.Resize((RESCALE_SIZE, RESCALE_SIZE)),
    transforms.ToTensor(),
])

# Берем параметры нормализации от ImageNet. т.к. будем использовать модели,
# предобученные на этом датасете
normalization = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

train_dataset = FaceDataset(datasets_split['train'],
                            transforms,
                            augmentations,
                            norm_params=normalization,
                            balance_classes=True,
                            load_to_device="cpu")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

print("Размерность train_dataset.images:", train_dataset.images.shape)
image, label = train_dataset.__getitem__(0)  # Получаем первый элемент
print("Размерность изображения, возвращаемого __getitem__:", image.shape)
for images, labels in train_loader:
    print("Размерность батча, возвращаемого DataLoader:", images.shape)
    break  # Останавливаемся после первого батча

print("ok!")