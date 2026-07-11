import torch
from torch import Tensor
from ultralytics import YOLO

class YoloCropModel():
    """
    Класс для обрезания финального датаматриксов.
    """

    model: YOLO
    """
    Моделька.
    """

    def __init__(self, yolo_path: str = "yolo26n-obb.pt"):
        """
        Конструктор класса.
        :param yolo_path: Путь к натренированной модели yolo.
        """
        self.model = YOLO(yolo_path)


    def train(self, path_to_dataset_yaml: str, epochs: int = 100, imgsz: int = 256) -> None:
        """
        Обучение модельки.
        :param path_to_dataset: Путь к датасету.
        :param epochs: Количество эпох.
        :param imgsz: Ресайз картинок под заданный размер (коробка imgsz X imgsz).
        """
        results = self.model.train(data=path_to_dataset_yaml, epochs=100, imgsz=imgsz)
        print(f"Метрики с тренировки: {results}")

    def crop(self, tensor: Tensor) -> Tensor | None:
        """
        Обрезает тензор по bbox найденному YOLO.
        :param tensor: Тензор [1, H, W] или [H, W] с предиктом нейронки.
        :return: Обрезанный тензор или None если YOLO ничего не нашла.
        """
        import numpy as np
        import cv2

        # тензор → numpy RGB для YOLO
        img_np = (tensor.squeeze().numpy() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        results = self.model(img_rgb, verbose=False)

        for result in results:
            if result.obb is None or len(result.obb) == 0:
                print("YOLO ничего не нашла")
                return None

            # берём самый уверенный bbox
            best_idx = result.obb.conf.argmax()
            xyxyxyxy = result.obb.xyxyxyxy[best_idx].cpu().numpy()

            pts = xyxyxyxy.reshape(4, 2).astype(np.int32)
            x, y, w, h = cv2.boundingRect(pts)

            # обрезаем оригинальный тензор
            cropped_np = img_np[y:y + h, x:x + w]
            cropped_tensor = torch.from_numpy(
                cropped_np.astype(np.float32) / 255.0
            ).unsqueeze(0)  # [1, H, W]

            return cropped_tensor

        return None