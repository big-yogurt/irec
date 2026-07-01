"""
Для обучения нейросети требуется датасет с различными изображениями. Создать
датасет с реальными фотографиями тяжело, поэтому было принято решение создать
генератор изображений datamatrix'ов.

В файле, помимо генерации datamatrix'ов, есть код для ухудшения качества
изображения, чтобы нейросеть училась исправлять дефекты и выдавать чистый
datamatrix.
"""


import random

import cv2
import numpy as np
import albumentations as A
import pylibdmtx.pylibdmtx as dmtx


DEFAULT_IMG_SIZE = (256, 256)


def gen_random_dmtx(max_data_len: int = 64) -> np.ndarray:
    """
    Генерирует datamatrix со случайными закодироваными данными. Возвращает в
    виде `np.ndarray` в RGB.
    Эта функция выдаёт маски, которые нейросеть должна давать на выходе.
    """
    data_len = random.randint(1, max_data_len)
    data = random.randbytes(data_len)
    encoded = dmtx.encode(data)
    img = np.frombuffer(encoded.pixels, dtype=np.uint8)
    img = img.reshape(encoded.height, encoded.width, 3)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return img


def make_img_realistic(img: np.ndarray) -> np.ndarray:
    """
    Добавляет к `img` фон (предполагается, что изображение чёрно-белое. Белый
    фон будет заменён на сгенерировный или заготовленное изображение). Так же
    случайным образом будут добавлены царапины, замыливание и др. дефекты,
    которые мешают сканированию datamatrix'а.
    Эта функция выдаёт изображения, которые нейросеть будет принимать на входе.
    """
    img = img.copy()
    if random.randint(1, 2) == 2:
        img = 255 - img
    if random.randint(1, 2) == 2:
        _add_scratches_to_img(img)
    _add_random_background_to_img(img)
    img = np.clip(img.astype(np.float32) * np.random.uniform(0.4, 2), 0, 255)
        .astype(np.uint8)
    final_transform = A.Compose([
        A.RandomBrightnessContrast(),
        A.RandomToneCurve(),
        A.MotionBlur(blur_limit=9, p=0.7),
        A.RandomSunFlare(src_radius=200),
    ])
    img = final_transform(image=img)['image']
    return img


def _add_random_background_to_img(img: np.ndarray):
    """
    Белый цвет заменяется на фоновое изображение.
    """
    bg = _gen_img_background()
    dmtx_mask = img[:, :, 0] != 0
    img[dmtx_mask] = bg[dmtx_mask]


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


def _add_scratches_to_img(img: np.ndarray, num_scratches=20, max_length=100):
    """
    Добавляет случайные царапины на изображение. Работает только с чёрно-белым
    изображением datamatrix'а, т.к. царапины - белые полосы, которые затирают
    datamatrix.
    """
    h, w = img.shape[:2]
    for _ in range(num_scratches):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(10, max_length)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        thickness = random.randint(1, 3)
        cv2.line(img, (x1, y1), (x2, y2), 255, thickness)


if __name__ == "__main__":
    for i in range(10):
        mask = gen_random_dmtx()
        img = make_img_realistic(mask)
        cv2.imwrite(f"images/img{i}.png", img)
        cv2.imwrite(f"images/mask{i}.png", mask)
