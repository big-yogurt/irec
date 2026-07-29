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


IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)


def gen_img_and_mask(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Генерация входного изображения и маски datamatrix.

    :param data: данные, которые будут в закодированы datamatrix.
    """
    encoded = dmtx.encode(data)
    mask = np.frombuffer(encoded.pixels, dtype=np.uint8)
    mask = mask.reshape(encoded.height, encoded.width, 3)

    dm_map = _extract_dm_region(mask)
    img = _gen_img(dm_map)

    img = cv2.resize(img, IMG_SIZE)
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    return img, mask


def _gen_img(dm_map: np.ndarray) -> np.ndarray:
    pattern, style = _gen_dm_pattern(dm_map)
    img = cv2.resize(pattern, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    dm_transform = A.Compose([
        A.Pad(
            padding=(
                random.randint(0,20),
                random.randint(0,20),
                random.randint(0,20),
                random.randint(0,20)
            ),
            fill=(255,255,255),
        ),
        A.RandomCrop(height=256, width=256, pad_if_needed=True,
            fill=(255,255,255)
        ),
        A.OpticalDistortion(distort_range=(-0.05, 0.05), fill=(255, 255, 255),
            p=0.3
        ),
        A.RandomRotate90(),
        A.ThinPlateSpline(scale_range=(0.01, 0.015), fill=(255, 255, 255), p=1),
        A.InvertImg(p=0.5 if style != "classic" else 0),
        A.AnnotationArtifacts(
            element_types=("line",),
            element_probabilities=(1.0,),
            count_range=(10, 60),
            thickness_range=(1, 2),
            line_geometry="random_angle",
            line_styles=("solid",),
            line_style_probabilities=(1.0,),
            color_palette=((255, 255, 255),),
            line_length_range=(1, random.randint(5, 50)),
            p=1,
        ),
    ])
    img = dm_transform(image=img)["image"]
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    _add_random_bg(img)
    img_transform = A.Compose([
        A.ISONoise(),
        A.OneOf([
            A.MotionBlur(blur_range=(11, 15), p=1),
        ]),
        A.RandomBrightnessContrast(
            brightness_range=(-0.3, 0.2),
            contrast_range=(-0.3, 0.2),
        ),
    ])
    img = img_transform(image=img)["image"]
    return img


def _extract_dm_region(encoded: np.ndarray) -> np.ndarray:
    """
    Извлечение области с datamatrix из изображения, генерируемого pylibdmtx.
    Убирает отступы и делат 1 ячейку datamatrix'а равной 1 пикселю.

    Извлечение области данных позволяет создавать большее различных видов
    datamatrix.

    Данная функция - костыль, т.к. у pylibdmtx нет возможности получить
    datamatrix с размером 1 ячейки в 1 пиксель.
    """
    if encoded.ndim == 3:
        encoded = encoded[:, :, 0]
    rows, cols = np.where(encoded == 0)
    top = rows.min()
    bottom = rows.max()
    left = cols.min()
    right = cols.max()
    dm = encoded[top:bottom+1, left:right+1]
    # В pybliencodedtx 1 ячейка занимает 5x5 пикселей. Приводим к 1x1.
    h = dm.shape[0] // 5
    w = dm.shape[1] // 5
    dm = cv2.resize(dm, (h, w))
    return dm


def _gen_dm_pattern(dm: np.ndarray) -> tuple[np.ndarray, str]:
    dm_height, dm_width = dm.shape
    cell_size = (IMG_HEIGHT // dm_height) - 1
    area_size = cell_size * dm_height
    img = np.full((area_size, area_size, 3), 255, dtype=np.uint8)
    if dm_height < 30:
        style = random.choice(("classic", "spaced", "dots"))
    else:
        style = random.choice(("classic", "spaced"))
    match style:
        case "classic":
            for y in range(0, area_size, cell_size):
                for x in range(0, area_size, cell_size):
                    if dm[int(y * (dm_width / area_size)),
                            int(x * (dm_height / area_size))] != 0:
                        continue
                    x0 = x
                    y0 = y
                    x1 = x + cell_size
                    y1 = y + cell_size
                    cv2.rectangle(img, (x0, y0), (x1, y1), 0, -1)
        case "spaced":
            space = 1
            for y in range(0, area_size, cell_size):
                for x in range(0, area_size, cell_size):
                    if dm[int(y * (dm_width / area_size)),
                            int(x * (dm_height / area_size))] != 0:
                        continue
                    x0 = x
                    y0 = y
                    x1 = x + (cell_size - 1) - space
                    y1 = y + (cell_size - 1) - space 
                    cv2.rectangle(img, (x0, y0), (x1, y1), 0, -1)
        case "dots":
            for y in range(0, area_size, cell_size):
                for x in range(0, area_size, cell_size):
                    if dm[int(y * (dm_width / area_size)),
                            int(x * (dm_height / area_size))] != 0:
                        continue
                    c0 = x + (cell_size - 1) // 2
                    c1 = y + (cell_size - 1) // 2
                    cv2.circle(img, (c0, c1), cell_size // 2, 0, -1)
    return img, style


def _add_random_bg(dm_pattern: np.ndarray):
    bg = _gen_bg()
    dmtx_mask = dm_pattern[:, :, 0] != 0
    dm_pattern[dmtx_mask] = bg[dmtx_mask]


def _gen_bg() -> np.ndarray:
    material = random.choice(("metal", "paper"))
    match material:
        case "metal":
            #img = np.full((256, 256, 3), random.randint(50, 200),
            #    dtype=np.uint8
            #)
            img = make_metal_texture()
        case "paper":
            img = np.full((256, 256, 3), random.randint(230, 255),
                dtype=np.uint8
            )
    bg_transform = A.Compose([
        #A.GaussNoise(std_range=(0.3, 0.5), p=1),
        A.Blur(blur_range=(7, 11), p=1),
        #A.Sharpen(alpha_range=(0.40, 0.50), p=1),
    ])
    img = bg_transform(image=img)["image"]
    return img


def make_metal_texture(size=(256, 256), seed=None):
    h, w = size
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Более светлая база
    base = rng.uniform(185, 235)
    img = np.full((h, w, 3), base, dtype=np.float32)
    img += rng.uniform(-4, 4, size=(1, 1, 3)).astype(np.float32)

    n_patches = rng.integers(3, 7)
    orientation_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32) + 1e-6

    for _ in range(n_patches):
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        rx = rng.uniform(w * 0.2, w * 0.8)
        ry = rng.uniform(h * 0.2, h * 0.8)
        ang = rng.uniform(0, np.pi)

        dx, dy = np.cos(ang), np.sin(ang)
        coord = (xx - cx) * dx + (yy - cy) * dy
        coord = coord / (np.max(np.abs(coord)) + 1e-6)

        mask = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * rng.uniform(1.0, 3.0))
        orientation_map += mask * coord
        weight_map += mask

    orientation_map /= weight_map

    global_angle = rng.uniform(0, np.pi)
    gx, gy = np.cos(global_angle), np.sin(global_angle)
    global_coord = xx * gx + yy * gy
    global_coord = (global_coord - global_coord.min()) / (global_coord.max() - global_coord.min() + 1e-6)

    brushing = 0.7 * orientation_map + 0.3 * global_coord

    # Ослабленные полосы
    freq = rng.uniform(20, 160)
    phase = rng.uniform(0, 2 * np.pi)
    amp = rng.uniform(1.5, 6.0)
    stripes = np.sin(brushing * freq * 2 * np.pi + phase) * amp
    img += stripes[:, :, None]

    # Мелкая текстура тоже слабее
    fine_freq = rng.uniform(100, 300)
    fine_amp = rng.uniform(0.5, 2.5)
    fine = np.sin(brushing * fine_freq * 2 * np.pi + rng.uniform(0, 2*np.pi)) * fine_amp
    img += fine[:, :, None]

    # Мягкий шум, без сильных провалов в тёмное
    for sigma, scale in [
        (rng.uniform(0.4, 1.2), rng.uniform(1, 3)),
        (rng.uniform(3, 10), rng.uniform(1, 5))
    ]:
        n = rng.normal(0, 1, (h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        img += n[:, :, None] * scale

    # Очень мягкий световой градиент
    if rng.random() < 0.95:
        ang = rng.uniform(0, 2 * np.pi)
        lx, ly = np.cos(ang), np.sin(ang)
        light = xx * lx + yy * ly
        light = (light - light.min()) / (light.max() - light.min() + 1e-6)
        light = (light - 0.5) * rng.uniform(5, 18)
        img += light[:, :, None]

    # Блики оставляем, но без пересвета
    for _ in range(rng.integers(1, 3)):
        hl = np.zeros((h, w), dtype=np.float32)
        cx = int(rng.integers(-w // 4, 5 * w // 4))
        cy = int(rng.integers(-h // 4, 5 * h // 4))
        ax = int(rng.integers(w // 10, w // 2))
        ay = int(rng.integers(h // 30, h // 8))
        ang = rng.uniform(0, 180)
        val = rng.uniform(8, 35)
        cv2.ellipse(hl, (cx, cy), (ax, ay), ang, 0, 360, val, -1)
        hl = cv2.GaussianBlur(hl, (0, 0), rng.uniform(8, 25))
        img += hl[:, :, None]

    # Тени сильно ослаблены
    for _ in range(rng.integers(0, 1)):
        sh = np.zeros((h, w), dtype=np.float32)
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        ax = int(rng.integers(w // 10, w // 3))
        ay = int(rng.integers(h // 30, h // 8))
        ang = rng.uniform(0, 180)
        val = rng.uniform(2, 8)
        cv2.ellipse(sh, (cx, cy), (ax, ay), ang, 0, 360, val, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(8, 16))
        img -= sh[:, :, None]

    # Царапины только светлые, не тёмные
    for _ in range(rng.integers(5, 18)):
        x1 = int(rng.integers(0, w))
        y1 = int(rng.integers(0, h))
        length = int(rng.integers(min(h, w) // 20, min(h, w) // 2))
        scratch_angle = rng.uniform(0, 2 * np.pi)
        x2 = int(np.clip(x1 + length * np.cos(scratch_angle), 0, w - 1))
        y2 = int(np.clip(y1 + length * np.sin(scratch_angle), 0, h - 1))
        c = int(np.clip(base + rng.integers(8, 25), 0, 255))
        cv2.line(img, (x1, y1), (x2, y2), (c, c, c), 1, lineType=cv2.LINE_AA)

    if rng.random() < 0.8:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.8))

    # Финальная нормализация: держим фон светлым
    alpha = rng.uniform(0.92, 1.00)
    beta = rng.uniform(0, 10)
    img = img * alpha + beta
    img = np.clip(img, 165, 255)

    return img.astype(np.uint8)


if __name__ == "__main__":
    for i in range(100):
        data_len = random.randint(1, 64)
        data = random.randbytes(data_len)
        img, mask = gen_img_and_mask(data)
        cv2.imwrite(f"images/img-{i}.png", img)
        cv2.imwrite(f"images/mask-{i}.png", mask)
