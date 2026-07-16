import itertools
import cv2
import numpy as np


BIN_LEVEL = 128   # Порог бинаризации (0-255): пиксели не ярче порога считаются чёрными
SURE_BLACK = 90   # Ячейка со средней яркостью не выше — точно чёрная
SURE_WHITE = 170  # Ячейка со средней яркостью не ниже — точно белая
CORE_PART = 0.6   # Доля центра ячейки, по которой меряется яркость (края отбрасываются)
NEAR_R = 2        # Радиус (в ячейках) поиска соседей для умной заливки
NEAR_SPREAD = 40  # Минимальный разброс яркости соседей, чтобы делить их на чёрные и белые
SHIFT_MAX = 3     # Максимальный сдвиг сетки в пикселях при подгонке
MIN_CELL_PX = 4   # Минимальный размер ячейки в пикселях; пятна мельче — шум
MIN_CELLS = 10    # Минимум ячеек на сторону (наименьший datamatrix — 10x10)
MAX_CELLS = 64    # Максимум ячеек на сторону


# Восстанавливает datamatrix на изображении, если код не найден — копия входа
def reconstruct(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, BIN_LEVEL, 255, cv2.THRESH_BINARY)

    # Шаг 1: кадр по чёрным пикселям
    box = _find_code_box(bw)
    if box is None:
        return gray.copy()
    y0, y1, x0, x1 = box
    crop = bw[y0:y1, x0:x1]
    h, w = crop.shape
    if h < MIN_CELLS * MIN_CELL_PX or w < MIN_CELLS * MIN_CELL_PX:
        return gray.copy()

    # Шаг 2: размер ячейки и число ячеек
    cell_px = _cell_size(crop)
    if cell_px is None:
        return gray.copy()
    edge = max(1, cell_px // 2)
    cols = _count_cells(cell_px, w, (crop[edge], crop[h - 1 - edge]))
    rows = _count_cells(cell_px, h, (crop[:, edge], crop[:, w - 1 - edge]))

    # Шаги 3-4: подгонка сетки и яркость ячеек
    means = _read_means(gray, box, rows, cols)

    # Шаг 5: чёрное/белое по порогам, спорные — по соседям
    cells = _fill_cells(means)

    # Шаг 6: рамка кода
    _fix_border(cells)

    # Шаг 7: отрисовка
    out = np.full_like(gray, 255)
    _draw(out[y0:y1, x0:x1], cells)
    return out

# Пытается декодировать datamatrix на изображении, данные кода или None, если декодировать не удалось
def try_decode(img: np.ndarray) -> bytes | None:
    from pylibdmtx.pylibdmtx import decode
    results = decode(img, timeout=500)
    return results[0].data if results else None

# Кадр (y0, y1, x0, x1) по крайним чёрным пикселям кода
def _find_code_box(bw: np.ndarray) -> tuple[int, int, int, int] | None:
    black = (bw == 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    boxes = [stats[i] for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= MIN_CELL_PX ** 2
    ]
    if not boxes:
        return None
    x0 = min(b[cv2.CC_STAT_LEFT] for b in boxes)
    y0 = min(b[cv2.CC_STAT_TOP] for b in boxes)
    x1 = max(b[cv2.CC_STAT_LEFT] + b[cv2.CC_STAT_WIDTH] for b in boxes)
    y1 = max(b[cv2.CC_STAT_TOP] + b[cv2.CC_STAT_HEIGHT] for b in boxes)
    return y0, y1, x0, x1

# Размер ячейки
def _cell_size(bw: np.ndarray) -> int | None:
    runs = []
    for line in bw:
        runs.extend(_runs(line))
    for line in bw.T:
        runs.extend(_runs(line))
    runs = [r for r in runs if r >= MIN_CELL_PX]
    if not runs:
        return None
    vals, counts = np.unique(runs, return_counts=True)
    return int(vals[np.argmax(counts)])

# Число ячеек вдоль стороны длиной length
def _count_cells(cell_px: int, length: int, edges: tuple) -> int:
    by_edge = max(_long_runs(line, cell_px) for line in edges)
    by_size = round(length / cell_px)
    n = by_edge if abs(by_edge - by_size) <= 2 else by_size
    n = int(n / 2 + 0.5) * 2  # размер datamatrix всегда чётный
    return max(MIN_CELLS, min(MAX_CELLS, n))

# Число серий длиной хотя бы в половину ячейки (отсев шума)
def _long_runs(line: np.ndarray, cell_px: int) -> int:
    return sum(1 for r in _runs(line) if r >= cell_px / 2)

# Длины серий подряд идущих одинаковых пикселей
def _runs(line: np.ndarray) -> list[int]:
    changes = np.where(np.diff(line) != 0)[0] + 1
    edges = np.concatenate(([0], changes, [len(line)]))
    return np.diff(edges).tolist()

# Границы ячеек
def _grid(h: int, w: int, rows: int, cols: int):
    ys = np.linspace(0, h, rows + 1).round().astype(int)
    xs = np.linspace(0, w, cols + 1).round().astype(int)
    return ys, xs

# Яркость ячеек при лучшем сдвиге сетки. Перебираем небольшие сдвиги и берём тот, где ячейки контрастнее
def _read_means(gray: np.ndarray, box: tuple, rows: int, cols: int) -> np.ndarray:
    y0, y1, x0, x1 = box
    ys, xs = _grid(y1 - y0, x1 - x0, rows, cols)
    ii = cv2.integral(gray)
    best, best_score = None, -1.0
    shifts = range(-SHIFT_MAX, SHIFT_MAX + 1)
    for dy, dx in itertools.product(shifts, repeat=2):
        means = _means_at(ii, ys + y0 + dy, xs + x0 + dx)
        score = np.abs(means - 128).mean()
        if score > best_score:
            best, best_score = means, score
    return best

# Средняя яркость центра каждой ячейки сетки
def _means_at(ii: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    h, w = ii.shape[0] - 1, ii.shape[1] - 1
    ys = np.clip(ys, 0, h)
    xs = np.clip(xs, 0, w)
    my = (np.diff(ys) * (1 - CORE_PART) / 2).astype(int)
    mx = (np.diff(xs) * (1 - CORE_PART) / 2).astype(int)
    ya = np.minimum(ys[:-1] + my, h - 1)
    xa = np.minimum(xs[:-1] + mx, w - 1)
    yb = np.clip(ys[1:] - my, ya + 1, h)
    xb = np.clip(xs[1:] - mx, xa + 1, w)
    sums = (ii[np.ix_(yb, xb)] - ii[np.ix_(ya, xb)] - ii[np.ix_(yb, xa)] + ii[np.ix_(ya, xa)])
    area = (yb - ya)[:, None] * (xb - xa)[None, :]
    return sums / area

# Решает цвет каждой ячейки
def _fill_cells(means: np.ndarray) -> np.ndarray:
    black = means <= SURE_BLACK
    white = means >= SURE_WHITE
    # общий порог кода: середина между опорным чёрным и белым
    black_ref = float(means[black].mean()) if black.any() else 0.0
    white_ref = float(means[white].mean()) if white.any() else 255.0
    mid = (black_ref + white_ref) / 2
    cells = np.full(means.shape, 255, dtype=np.uint8)
    cells[black] = 0
    for i, j in zip(*np.where(~black & ~white)):
        cells[i, j] = _near_color(means, i, j, mid)
    return cells

# Умная заливка: цвет спорной ячейки по соседям
def _near_color(means: np.ndarray, i: int, j: int, mid: float) -> int:
    y0, x0 = max(0, i - NEAR_R), max(0, j - NEAR_R)
    near = means[y0:i + NEAR_R + 1, x0:j + NEAR_R + 1].ravel()
    if near.max() - near.min() >= NEAR_SPREAD:
        mid = _split_level(near)
    return 0 if means[i, j] <= mid else 255

# Порог между двумя кластерами яркости (метод Оцу)
def _split_level(vals: np.ndarray) -> float:
    vals = np.sort(vals)
    n = len(vals)
    sums = np.cumsum(vals)
    k = np.arange(1, n)
    left = sums[:-1] / k
    right = (sums[-1] - sums[:-1]) / (n - k)
    spread = k * (n - k) * (left - right) ** 2
    best = int(np.argmax(spread))
    return (vals[best] + vals[best + 1]) / 2

# Рамка datamatrix: две сплошные чёрные стороны (L-линия) и пунктир на противоположных
def _fix_border(cells: np.ndarray):
    rows, cols = cells.shape
    # у сплошных сторон средняя яркость ниже (чёрный = 0)
    bottom_solid = cells[-1].mean() <= cells[0].mean()
    left_solid = cells[:, 0].mean() <= cells[:, -1].mean()
    dash_row = 0 if bottom_solid else rows - 1
    dash_col = cols - 1 if left_solid else 0
    # пунктир идёт от чёрного угла возле сплошной стороны через одну
    start_j = 0 if left_solid else cols - 1
    start_i = rows - 1 if bottom_solid else 0
    for j in range(cols):
        cells[dash_row, j] = 0 if (j - start_j) % 2 == 0 else 255
    for i in range(rows):
        cells[i, dash_col] = 0 if (i - start_i) % 2 == 0 else 255
    # сплошные стороны L-линии
    cells[-1 if bottom_solid else 0, :] = 0
    cells[:, 0 if left_solid else -1] = 0

# Рисует ячейки кода на области кадра
def _draw(canvas: np.ndarray, cells: np.ndarray):
    ys, xs = _grid(*canvas.shape, *cells.shape)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            canvas[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = cells[i, j]
