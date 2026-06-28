import random

import cv2
import numpy as np
import albumentations as A
import pylibdmtx.pylibdmtx as dmtx
import torch
from torch import Tensor
from torch.utils.data import Dataset


DEFAULT_IMG_SIZE = (256, 256)


class DMSyntheticDataset(Dataset):
    """
    Датасет синтетических данных, который генерирует изображения datamatrix
    кодов на лету.
    """

    def __init__(self, dataset_len: int = 1000):
        self._dataset_len = dataset_len

    def __len__(self):
        """
        Возвращает размер датасета.
        """
        return self._dataset_len

    def __getitem__(self, _) -> tuple[Tensor, Tensor]:
        """
        Возвращает синтетические данные (изображения DM кодов в виде тензоров).
        Первый тензор - входное изображение, второй - маска сегментации.
        """
        bytes_count = random.randint(1, 64)
        data = random.randbytes(bytes_count)
        dmtx_img = dmtx.encode(data)

        ideal_img = np \
            .frombuffer(dmtx_img.pixels, dtype=np.uint8) \
            .reshape(dmtx_img.height, dmtx_img.width, 3)
        ideal_img = cv2 \
            .resize(ideal_img, (256, 256), interpolation=cv2.INTER_AREA)
        damage_img = _add_random_damage_to_img(ideal_img)

        mask = torch \
            .from_numpy(ideal_img.astype(np.float32) / 255.0) \
            .unsqueeze(0)
        img = torch \
            .from_numpy(damage_img.astype(np.float32) / 255.0) \
            .permute(1, 2, 0)
        return img, mask 


def _add_random_damage_to_img(ideal_img: np.ndarray) -> np.ndarray:
    """
    'Портит' изображение, добавляя различные помехи.
    """
    img = _add_random_background_to_img(ideal_img)
    final_transform = A.Compose([
        A.RandomBrightnessContrast(),
        A.RandomToneCurve(),
        A.MotionBlur(blur_limit=9, p=0.7),
        A.RandomSunFlare(src_radius=200),
    ])
    img = final_transform(image=img)['image']
    return img


def _add_random_background_to_img(img: np.ndarray) -> np.ndarray:
    bg = _gen_img_background()
    h_code, w_code = img.shape[:2]
    roi = bg[0:h_code, 0:w_code]
    mask = (img[:, :, 0] == 0)
    roi[mask] = img[mask]
    bg[0:h_code, 0:w_code] = roi
    return bg


def _gen_img_background() -> np.ndarray:
    h, w = DEFAULT_IMG_SIZE
    n = random.randint(0, 2)
    if n == 0: # Металл
        noise = np.random.normal(128, 30, (h, w)).astype(np.uint8)
        kernel = np.ones((1, 5), np.float32) / 5
        tex = cv2.filter2D(noise, -1, kernel)
        tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
        grad = np.linspace(0.8, 1.2, w).reshape(1, w, 1)
        tex = (tex * grad).astype(np.uint8)
        return tex
    elif n == 1: # Бумага
        return np.random.normal(220, 6, (h, w, 3)).astype(np.uint8)
    return np.random.normal(180, 10, (h, w, 3)).astype(np.uint8)


if __name__ == "__main__":
    dataset = DMSyntheticDataset(dataset_len=10)
    for i in range(10):
        img, mask = dataset[0]
        img, mask = (img.permute(2, 0, 1).numpy() * 255) \
            .astype(np.uint8), (mask.squeeze(0).numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"images/img{i}.png", img)
        cv2.imwrite(f"images/mask{i}.png", mask)
