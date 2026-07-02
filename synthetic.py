import cv2
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
        ideal_img, damage_img = self._pairs[idx]
        ideal_gray = cv2.cvtColor(ideal_img, cv2.COLOR_RGB2GRAY)
        mask = to_tensor(ideal_gray)
        img = to_tensor(damage_img)
        return img, mask
