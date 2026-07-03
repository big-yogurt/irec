import os
import time

import cv2
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

import imggen


class DMSyntheticDataset(Dataset):
    """
    Датасет синтетических данных, который генерирует изображения datamatrix
    кодов на лету.
    """

    def __init__(self, dataset_len: int = 1000):
        self._dataset_len = dataset_len
        self._pairs = tuple(imggen.gen_random_img_pair()
            for _ in range(dataset_len)
        )

    def __len__(self):
        """
        Возвращает размер датасета.
        """
        return self._dataset_len

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        """
        Возвращает синтетические данные (изображения DM кодов в виде тензоров).
        Первый тензор - входное изображение, второй - маска сегментации.
        """
        damage_img, ideal_img = self._pairs[idx]
        ideal_gray = cv2.cvtColor(ideal_img, cv2.COLOR_RGB2GRAY)
        img = to_tensor(damage_img)
        mask = to_tensor(ideal_gray)
        return img, mask


class DMPreloadedDataset(Dataset):
    """
    Датасет данных, расположенных на диске. Формат данных:
    <папка_с_датасетом>
    |
    +-- imgs
    |   |
    |   +- 0.png
    |   +- 1.png
    |   +- ...
    |   +- <n>.png
    |
    +-- masks
        |
        +- 0.png
        +- 1.png
        +- ...
        +- <n>.png
    """

    def __init__(self, path: str):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Папка {path} не найдена")
        if not os.path.isdir(path + "/imgs"):
            raise FileNotFoundError(f"Папка {path + '/imgs'} не найдена")
        if not os.path.isdir(path + "/masks"):
            raise FileNotFoundError(f"Папка {path + '/masks'} не найдена")
        self._path = path
        imgs_file_count = len(os.listdir(path + "/imgs"))
        masks_file_count = len(os.listdir(path + "/masks"))
        if masks_file_count != imgs_file_count:
            raise ValueError(f"Количество файлов в {path + '/imgs'} не равно" \
                f"количеству файлов в {path + '/masks'}"
            )
        self._dataset_len = imgs_file_count

    def __len__(self) -> int:
        return self._dataset_len

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        img_path = self._path + "/imgs" + f"/{idx}.png"
        mask_path = self._path + "/masks" + f"/{idx}.png"
        if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Пары под индексом {idx} нет в датасете")
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        return to_tensor(img), to_tensor(mask)
