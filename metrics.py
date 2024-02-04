import itertools
import numpy as np

from PIL import Image

import torch
import torch.nn.functional as F

from pathlib import PosixPath


def compute_embeddings(model, images_list, transform, device=None, batch_size=512):
    """
    compute embeddings from the trained model for list of images.
    params:
      model: trained nn model that takes images and outputs embeddings
      images_list: list of images paths to compute embeddings for
    output:
      list: list of model embeddings. Each embedding corresponds to images
            names from images_list
    """

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    face_images = [transform(Image.open(img_path).convert('RGB')) for img_path in images_list]
    face_images = torch.stack(face_images).to(device)

    model.eval()

    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(face_images), batch_size):
            batch = face_images[i:i + batch_size]
            embeddings = model(batch)
            embeddings_list.extend(embeddings.cpu().tolist())

    del face_images

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return embeddings_list


def compute_pairwise_similarities(embeddings_list, pairs_indices, normalize=False, batch_size=512):

    embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float32)

    if normalize:
        embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)

    # Разбивка списка пар на батчи
    n_pairs = len(pairs_indices)
    pairwise_similarities = []

    for i in range(0, n_pairs, batch_size):
        pairs_batch = pairs_indices[i:i + batch_size]
        vectors1 = embeddings_tensor[pairs_batch[:, 0]]
        vectors2 = embeddings_tensor[pairs_batch[:, 1]]

        # Вычисление сходства для каждой пары в батче
        similarity_batch = torch.sum(vectors1 * vectors2, dim=1).tolist()
        pairwise_similarities.extend(similarity_batch)

    return pairwise_similarities


def get_index_pairs_pos(query_dict, query_img_names):
    index_pairs = []

    # Создаем словарь для быстрого поиска индекса по имени изображения
    if isinstance(query_img_names[0], PosixPath):
        # в моих структурах в списке пути
        name_to_index = {file.name: idx for idx, file in enumerate(query_img_names)}
    else:
        # а в тестах ноутбука - строки
        name_to_index = {name: idx for idx, name in enumerate(query_img_names)}

    for img_class, img_names in query_dict.items():
        # Получаем индексы изображений для текущего класса
        class_indices = [name_to_index[name] for name in img_names if name in name_to_index]

        # Формируем все возможные пары индексов для изображений этого класса
        for pair in itertools.combinations(class_indices, 2):
            index_pairs.append(pair)

    return index_pairs


def get_index_pairs_neg(query_dict, query_img_names):
    cross_class_pairs = []

    # Создаем словарь для быстрого поиска индекса по имени изображения
    if isinstance(query_img_names[0], PosixPath):
        # в моих структурах в списке пути
        name_to_index = {file.name: idx for idx, file in enumerate(query_img_names)}
    else:
        # а в тестах ноутбука - строки
        name_to_index = {name: idx for idx, name in enumerate(query_img_names)}

    class_names = list(query_dict.keys())

    # Перебор всех пар классов
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            class1, class2 = class_names[i], class_names[j]

            # Получаем индексы изображений для каждого класса
            indices1 = [name_to_index[name] for name in query_dict[class1] if name in name_to_index]
            indices2 = [name_to_index[name] for name in query_dict[class2] if name in name_to_index]

            # Формируем все возможные пары индексов между классами
            for pair in itertools.product(indices1, indices2):
                cross_class_pairs.append(pair)

    return cross_class_pairs


def compute_cosine_query_pos(classes_dict, img_names, embeddings, normalize=False, batch_size=512):
    """
    compute cosine similarities between positive pairs from query (stage 1)
    params:
      query_dict: dict {class: [image_name_1, image_name_2, ...]}. Key: class in
                  the dataset. Value: images corresponding to that class
      query_img_names: list of images names
      query_embeddings: list of embeddings corresponding to query_img_names
    output:
      list of floats: similarities between embeddings corresponding
                      to the same people from query list
    """
    pairs_indices = get_index_pairs_pos(classes_dict, img_names)
    similarities = compute_pairwise_similarities(embeddings, pairs_indices, normalize, batch_size)

    return similarities


def compute_cosine_query_neg(classes_dict, img_names, embeddings, normalize=False, batch_size=512):
    """
    compute cosine similarities between negative pairs from query (stage 2)
    params:
      query_dict: dict {class: [image_name_1, image_name_2, ...]}. Key: class in
                  the dataset. Value: images corresponding to that class
      query_img_names: list of images names
      query_embeddings: list of embeddings corresponding to query_img_names
    output:
      list of floats: similarities between embeddings corresponding
                      to different people from query list
    """
    pairs_indices = get_index_pairs_neg(classes_dict, img_names)
    similarities = compute_pairwise_similarities(embeddings, pairs_indices, normalize, batch_size)

    return similarities


def compute_cosine_query_distractors(embeddings_1, embeddings_2, normalize=False, batch_size=512):
    """
    compute cosine similarities between negative pairs from query and distractors
    (stage 3)
    params:
      query_embeddings: list of embeddings corresponding to query_img_names
      distractors_embeddings: list of embeddings corresponding to distractors_img_names
    output:
      list of floats: similarities between pairs of people (q, d), where q is
                      embedding corresponding to photo from query, d —
                      embedding corresponding to photo from distractors
    """
    len_1 = len(embeddings_1)
    len_2 = len(embeddings_2)

    query_dict = {
        1: list(range(len_1)),
        2: [x + len_1 for x in range(len_2)]
    }

    fake_img_names = list(range(len_1 + len_2))

    pairs_indices = get_index_pairs_neg(query_dict, fake_img_names)

    embeddings = embeddings_1 + embeddings_2

    similarities = compute_pairwise_similarities(embeddings, pairs_indices, normalize, batch_size)

    return similarities


def compute_ir(cosine_query_pos, cosine_query_neg, cosine_query_distractors, fpr=0.1):
    """
    compute identification rate using precomputer cosine similarities between pairs
    at given fpr
    params:
      cosine_query_pos: cosine similarities between positive pairs from query
      cosine_query_neg: cosine similarities between negative pairs from query
      cosine_query_distractors: cosine similarities between negative pairs
                                from query and distractors
      fpr: false positive rate at which to compute TPR
    output:
      float: threshold for given fpr
      float: TPR at given FPR
    """
    false_pairs = np.concatenate([cosine_query_neg, cosine_query_distractors])
    false_pairs = np.sort(false_pairs)[::-1]

    num_false_pairs = len(false_pairs)
    allowed_false_positives = int(fpr * num_false_pairs)

    threshold_distance = false_pairs[allowed_false_positives]

    true_positives = np.sum(cosine_query_pos > threshold_distance)
    total_positives = len(cosine_query_pos)
    tpr = true_positives / total_positives

    return threshold_distance, tpr
