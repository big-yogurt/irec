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


def gen_random_dmtx(max_data_len: int = 64) -> np.ndarray | None:
    """
    Генерирует datamatrix со случайными закодироваными данными. Возвращает в
    виде `np.ndarray` в RGB.
    Эта функция выдаёт маски, которые нейросеть должна давать на выходе.
    """
    data_len = random.randint(1, max_data_len)
    data = random.randbytes(data_len)
    return gen_dmtx(data)


def gen_dmtx(data: bytes) -> np.ndarray | None:
    """
    Генерирует datamatrix со заданными данными. Возвращает в виде `np.ndarray`
    в RGB.
    Эта функция выдаёт маски, которые нейросеть должна давать на выходе.
    Возвращает `None`, если количество байт больше 64. Ограниченно из-за
    возможности выпадения исключения.
    """
    if len(data) > 64:
        return None
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
    img = np.clip(img.astype(np.float32) * np.random.uniform(0.4, 2), 0, 255) \
        .astype(np.uint8)
    final_transform = A.Compose([
        A.RandomBrightnessContrast(),
        A.RandomToneCurve(),
        A.MotionBlur(p=0.7),
        A.RandomSunFlare(src_radius=200),
        A.Perspective(scale=(0.005, 0.015)),
        #A.WaterRefraction(
        #    amplitude_range=(0.01, 0.02),
        #    wavelength_range=(0.01, 0.02),
        #    num_waves_range=(1, 1),
        #),
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


def _add_scratches_to_img(img: np.ndarray):
    """
    Добавляет случайные царапины на изображение. Работает только с чёрно-белым
    изображением datamatrix'а, т.к. царапины - белые полосы, которые затирают
    datamatrix.
    """
    h, w = img.shape[:2]
    num_scratches = random.randint(10, 50)
    max_length = random.randint(5, 100)
    for _ in range(num_scratches):
        # Начальные точки царапины
        x1, y1 = random.randint(0, w), random.randint(0, h)

        # Конечные точки царапины
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(1, max_length)
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


from collections.abc import Generator


def bezier(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) \
        -> Generator[np.ndarray, None, None]:
    def calc(t):
        return t * t * p1 + 2 * t * (1 - t) * p2 + (1 - t) * (1 - t) * p3

    # get the approximate pixel count of the curve
    approx = cv2.arcLength(np.array([calc(t)[:2] for t in np.linspace(0, 1, 10)], dtype=np.float32), False)
    for t in np.linspace(0, 1, round(approx * 1.2)):
        yield np.round(calc(t)).astype(np.int32)


def generate_scratch(img: np.ndarray, max_length: float,
        end_brush_range: tuple[float, float],
        mid_brush_range: tuple[float, float]) -> np.ndarray:
    H, W = img.shape
    # generate the 2 end points of the bezier curve
    x, y, rho1, theta1 = np.random.uniform(
        [0] * 4, [W, H, max_length, np.pi * 2]
    )
    p1 = np.array([x, y, 0])
    p3 = p1 + [rho1 * np.cos(theta1), rho1 * np.sin(theta1), 0]

    # generate the second point, make sure that it cannot be too far away from
    # the middle point of the 2 end points
    rho2, theta2 = np.random.uniform([0], [rho1 / 2, np.pi * 2])
    p2 = (p1 + p3) / 2 + [rho2 * np.cos(theta2), rho2 * np.sin(theta2), 0]

    # generate the brush sizes of the 3 points
    p1[2], p2[2], p3[2] = np.random.uniform(
        *np.transpose([end_brush_range, mid_brush_range, end_brush_range])
    )

    for x, y, brush in bezier(p1, p2, p3):
        cv2.circle(img, (x, y), brush, 255, -1)
    return img


