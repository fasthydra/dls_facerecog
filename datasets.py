import random
import gc
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def load_and_split_dataset(data_dir: str, split_file: str, anno_file: str) -> dict:
    """
        Загружает и разделяет датасет на подмножества для обучения, валидации и тестирования.

        :param data_dir: Путь к директории с изображениями.
        :param split_file: Путь к файлу, содержащему информацию о разбиении набора данных.
                           Каждая строка должна содержать имя файла изображения и индекс подмножества (0 для обучения,
                           1 для валидации, 2 для теста), разделенные пробелом.
        :param anno_file: Путь к файлу аннотаций, где каждая строка содержит имя файла изображения и метку класса,
                          разделенные пробелом.

        Функция сначала загружает информацию о разбиении и аннотациях из указанных файлов.
        Затем создает словарь аннотаций и разделяет изображения по подмножествам в соответствии с предоставленным
        разбиением. Для каждого изображения, если оно существует в указанной директории, функция добавляет его путь и
        метку класса в соответствующий список подмножества в словаре `datasets`.

        Возвращает словарь, ключами которого являются названия подмножеств ('train', 'val', 'test'), а значениями -
        списки кортежей (img_path, label), где `img_path` - путь к изображению, `label` - метка класса.
    """

    data_dir = Path(data_dir)

    # Загрузка файла разбиения
    with open(split_file, 'r') as file:
        split_lines = file.readlines()

    # Загрузка файла аннотаций
    with open(anno_file, 'r') as file:
        anno_lines = file.readlines()

    # Создание словаря аннотаций
    annotations = {line.split()[0]: int(line.split()[1]) for line in anno_lines}

    # Словари для хранения данных по подмножествам
    datasets = {'train': [], 'val': [], 'test': []}
    subset_map = {'train': 0, 'val': 1, 'test': 2}

    for line in split_lines:
        img_name, split = line.strip().split()
        img_path = data_dir / "celebA_imgs" / img_name
        if img_path.is_file():
            label = annotations[img_name]
            # Определяем подмножество по значению split
            for subset, code in subset_map.items():
                if int(split) == code:
                    datasets[subset].append((img_path, label))
                    break

    return datasets


class FaceDataset(Dataset):
    def __init__(self,
                 sample,
                 transform=None,
                 augmentations=None,
                 norm_params=None,
                 balance_classes=False,
                 load_to_device="",
                 batch_size=512):
        """
        Инициализирует датасет.

        :param sample: Список кортежей (img_path, label).
        :param transform: Преобразование, применяемое к каждому изображению.
        :param augmentations: Список аугментаций, применяемых к изображениям.
        :param balance_classes: Если True, балансирует классы через аугментацию.
        :param load_to_device: Если 'cpu' или 'gpu', 'cuda', то все изображения загружаются в один тензор
                               в память соответствующего устройства
        :param batch_size: Размер батча для обработки изображений.
        """
        self.sample = sample
        self.transform = transform
        self.augmentations = augmentations or []
        self.balance_classes = balance_classes
        self.batch_size = batch_size

        if load_to_device.lower() in ("cuda", "gpu"):
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        elif load_to_device.lower() == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = None

        self.images = None
        if self.device:
            self._load_images()

        if self.balance_classes:
            self.augmentation_counts = self._calculate_augmentation_counts()
        else:
            self.augmentation_counts = {}

        if norm_params:
            self.normalize = transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])
        else:
            self.normalize = None

    def _calculate_augmentation_counts(self):
        """
        Рассчитывает количество аугментаций, необходимое для балансировки классов.

        :return: Словарь с количеством аугментаций для каждого класса.
        """
        class_counts = {}
        for _, label in self.sample:
            class_counts[label] = class_counts.get(label, 0) + 1

        max_count = max(class_counts.values())
        augmentation_counts = {label: max_count - count for label, count in class_counts.items() if count < max_count}

        return augmentation_counts

    def _load_images(self):
        """
        Загружает все изображения датасета в тензор, хранящийся на заданном устройстве.
        """
        temp_images = []
        for i, (img_path, _) in enumerate(self.sample):
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            temp_images.append(image.unsqueeze(0))  # Добавляем размерность батча

            if (i + 1) % self.batch_size == 0 or (i + 1) == len(self.sample):
                batch = torch.cat(temp_images, dim=0).to(self.device)
                self.images = batch if self.images is None else torch.cat((self.images, batch), dim=0)
                temp_images = []
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        gc.collect()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def __len__(self):
        """
        Возвращает общее количество образцов в датасете, включая аугментации для балансировки классов.

        :return: Общее количество образцов в датасете.
        """
        if self.balance_classes:
            return len(self.sample) + sum(self.augmentation_counts.values())
        else:
            return len(self.sample)

    def _get_balanced_idx(self, idx):
        """
        Выбирает случайное изображение из класса, требующего аугментации, для балансировки классов.

        :param idx: Индекс, для которого требуется аугментация.
        :return: Индекс аугментированного изображения и его метка.
        """
        # Создаем список классов, которым требуется аугментация
        classes_needing_augmentation = [label for label, cnt in self.augmentation_counts.items() if cnt > 0]

        # Проверяем, не пуст ли список
        if not classes_needing_augmentation:
            # Если все аугментации выполнены, возвращаем обычный образец
            img_idx = idx % len(self.sample)
            label = self.sample[img_idx][1]
        else:
            # Случайно выбираем класс из этого списка
            label = random.choice(classes_needing_augmentation)
            self.augmentation_counts[label] -= 1

            # Выбираем случайный индекс изображения из этого класса
            img_indices = [i for i, data in enumerate(self.sample) if data[1] == label]
            img_idx = random.choice(img_indices)

        return img_idx, label

    def __getitem__(self, idx):
        """
        Возвращает изображение и метку по заданному индексу, с учетом балансировки классов.

        :param idx: Индекс образца.
        :return: Кортеж (изображение, метка).
        """

        # Выбираем индекс изображения и метку с учетом балансировки классов
        if idx < len(self.sample):
            label = self.sample[idx][1]
            img_idx = idx
        else:
            img_idx, label = self._get_balanced_idx(idx)

        # Получаем тензор изображения в зависимости от того, было ли оно в __init__ загружено в ОЗУ
        if self.images is not None:
            image = self.images[img_idx]
        else:
            img_path = self.sample[img_idx][0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

        # Применение аугментаций
        if self.augmentations:
            image = self.augmentations(image)

        # Удаление первой размерности, если она была добавлена
        if image.dim() > 3:
            image = image.squeeze(0)

        # Применение нормализации
        if self.normalize:
            image = self.normalize(image)

        return image, label


class FaceDatasetTriplet(Dataset):
    def __init__(self, face_dataset_tensor):
        """
        Инициализирует датасет для генерации триплетов, используя pandas DataFrame для управления данными.

        :param face_dataset_tensor: Экземпляр класса FaceDatasetTensor, содержащий данные датасета.
        """
        self.dataset = face_dataset_tensor
        # Создаем DataFrame из samples
        self.labels = pd.DataFrame(self.dataset.sample, columns=['idx', 'label'])
        self.labels['idx'] = self.labels.index

    def __getitem__(self, index):
        """
        Возвращает триплет (анкор, позитив, негатив) по заданному индексу.

        :param index: Индекс анкора.
        :return: Триплет изображений (анкор, позитив, негатив).
        """
        anchor_idx, anchor_label = self.labels.iloc[index]

        # Получаем позитивный пример (тот же класс, что и анкор)
        pos_labels = self.labels[self.labels['label'] == anchor_label]
        pos_idxs = pos_labels['idx'].tolist().remove(anchor_idx) # Удаляем индекс анкора
        pos_idx = random.choice(pos_idxs)

        # Получаем негативный пример (другой класс)
        neg_labels = self.labels[self.labels['label'] != anchor_label]
        neg_label = random.choice(neg_labels['label'].unique())
        neg_idxs = self.labels[self.labels['label'] == neg_label]['idx'].tolist()
        neg_idx = random.choice(neg_idxs)

        anchor_img = self.dataset[anchor_idx][0]
        pos_img = self.dataset[pos_idx][0]
        neg_img = self.dataset[neg_idx][0]

        return anchor_img, pos_img, neg_img

    def __len__(self):
        """
        Возвращает общее количество изображений в датасете.
        """
        return len(self.labels)
