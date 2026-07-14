import os

import cv2
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


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
