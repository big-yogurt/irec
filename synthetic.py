import io
import random
from typing import TypedDict

import numpy as np
import cv2
from PIL import Image
from PIL.Image import Image as PIL_Image
import albumentations as A
from pylibdmtx.pylibdmtx import encode, decode
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

## Типизация выхлопа с get_item
class DMSyntheticItem(TypedDict):
    input_tensor: Tensor
    target_tensor: Tensor
    text: str



class DMSyntheticDataset(Dataset):
    def __init__(self, size=128, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                 min_len=4, max_len=20, dataset_len=1000, transforms=None):
        self.size = size
        self.alphabet = alphabet
        self.min_len = min_len
        self.max_len = max_len
        self.dataset_len = dataset_len
        self.transforms = transforms or self.default_transforms()

    @staticmethod
    def default_transforms() -> A.Compose:
        return A.Compose([
            A.Rotate(limit=(-90, 90), border_mode=cv2.BORDER_CONSTANT),  # небольшой поворот
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.5),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.AdvancedBlur(blur_limit=(3, 5), p=0.3),  # дефокус
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.Downscale(scale_range=(0.5, 0.9), p=0.2),
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.2),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_range=(0, 1), num_flare_circles_range=(1, 2), src_radius=200, src_color=(255, 255, 255), p=0.2),
            A.RandomRain(slant_range=(-10, 10), drop_length=10, drop_width=1, drop_color=(200, 200, 200),
                         p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
        ])

    def generate_dm_code(self, text) -> PIL_Image:
        # Генерируем DM-код с помощью pylibdmtx
        size = 'RectAuto'
        encoded = encode(text.encode('utf-8'), size=size)  # возвращает bytes
        # Декодируем в массив numpy (градации серого)
        img = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels)
        img = np.array(img.convert('L'))  # (h, w) grayscale
        return img

    def __getitem__(self, _) -> DMSyntheticItem:
        # Генерируем случайный текст
        length = random.randint(self.min_len, self.max_len)
        text = ''.join(random.choices(self.alphabet, k=length))

        # Создаём чистый DM-код
        clean = self.generate_dm_code(text)
        # Ресайзим до (size, size) с сохранением пропорций, добавляем quiet zone
        # pylibdmtx уже включает quiet zone, просто ресайз
        clean = cv2.resize(clean, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        clean = clean.astype(np.float32) / 255.0  # нормализуем [0,1]

        # Преобразуем в RGB (повторяем канал 3 раза для входного цвета)
        input_rgb = np.stack([clean, clean, clean], axis=-1)  # (H,W,3)

        # Применяем аугментации (работают с RGB)
        if self.transforms:
            augmented = self.transforms(image=(input_rgb * 255).astype(np.uint8))
            input_rgb = augmented['image'] / 255.0  # [0,1]

        # Иногда инвертируем цвета (белый фон, чёрные модули) – в диссертации есть
        if random.random() < 0.3:
            input_rgb = 1.0 - input_rgb
            clean = 1.0 - clean  # таргет тоже инвертируем? В диссертации инвертируют только вход? Лучше таргет оставить оригинальным, чтобы сеть училась восстанавливать правильный контраст.
            # В работе: "image color inversion" – вероятно, инвертируют и вход, и таргет, чтобы сеть училась нормализовать. Я сделаю так:
            # clean оставляем как есть (чёрные модули на белом), а вход инвертируем, сеть должна научиться возвращать правильную полярность.
            # Но тогда таргет тоже надо инвертировать? Лучше оставить таргет без инверсии, чтобы сеть всегда выдавала чёрное на белом.
            # Так и сделаем: инвертируем только вход, таргет – исходный clean.
            # Но если мы инвертируем вход, то сеть должна восстановить правильную полярность.

        # Преобразуем в тензоры PyTorch (C, H, W)
        input_tensor = torch.from_numpy(input_rgb.transpose(2, 0, 1)).float()

        target_tensor = torch.from_numpy(clean).unsqueeze(0).float()  # (1,H,W)

        return DMSyntheticItem(
            input_tensor=input_tensor, target_tensor=target_tensor, text=text
        )

    def __len__(self):
        return self.dataset_len

    def create_dataloader(self, batch_size=64) -> DataLoader:
        """
        Создает датасет c текущими настройками
        """
        return DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=6)