import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class FaceClassifier(nn.Module):
    def __init__(self, backbone_model, num_classes=500, normalize_embeddings=True):
        super(FaceClassifier, self).__init__()

        self.backbone = backbone_model

        # Удаляем последний полносвязный слой предобученной модели
        num_features = self.backbone._fc.in_features  # Получаем количество фичей для последнего FC слоя
        self.backbone._fc = nn.Identity()  # Удаляем последний полносвязный слой

        self.normalize_embeddings = normalize_embeddings

        # Добавление нового полносвязного слоя для классификации на заданное количество классов
        self.classifier = nn.Linear(num_features, num_classes, bias=False)
        if normalize_embeddings:
            # Применение нормализации весов к классификатору
            self.classifier = weight_norm(self.classifier, name='weight', dim=0)

    def forward(self, x, return_features=False):
        features = self.backbone(x)

        if self.normalize_embeddings:
            # Нормализация признаков, если это требуется
            features = F.normalize(features, p=2, dim=1)

        if return_features:
            # Возвращение нормализованных признаков, если требуется
            return features

        logits = self.classifier(features)
        return logits


class FaceClassifier(nn.Module):
    def __init__(self, backbone_model, num_classes=500, normalize_embeddings=True):
        super(FaceClassifier, self).__init__()

        self.backbone = backbone_model

        # Удаляем последний полносвязный слой предобученной модели
        num_features = self.backbone._fc.in_features  # Получаем количество фичей для последнего FC слоя
        self.backbone._fc = nn.Identity()  # Удаляем последний полносвязный слой

        self.normalize_embeddings = normalize_embeddings

        # Добавление нового полносвязного слоя для классификации на заданное количество классов
        self.classifier = nn.Linear(num_features, num_classes, bias=False)
        if normalize_embeddings:
            # Применение нормализации весов к классификатору
            self.classifier = weight_norm(self.classifier, name='weight', dim=0)

    def forward(self, x, return_features=False):
        features = self.backbone(x)

        if self.normalize_embeddings:
            # Нормализация признаков, если это требуется
            features = F.normalize(features, p=2, dim=1)

        if return_features:
            # Возвращение нормализованных признаков, если требуется
            return features

        logits = self.classifier(features)
        return logits


class FaceEmbeddingExtractor(nn.Module):
    def __init__(self, backbone_model):
        super(FaceEmbeddingExtractor, self).__init__()

        self.backbone = backbone_model

        # Удаляем последний полносвязный слой предобученной модели
        self.backbone._fc = nn.Identity()  # Удаляем последний полносвязный слой

    def forward(self, x):
        features = self.backbone(x)
        return F.normalize(features, p=2, dim=1)
