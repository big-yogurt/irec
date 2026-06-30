import csv
import itertools
import os.path
from os import mkdir
from typing import List, Optional, Callable, Any

from torch import Tensor

import loss
from train import DMTrainModel


class DMModelTester():
    """
    Класс для тестирования разных параметров нейросетки.
    Основное назначение - оптимизация уже созданной, но не затюненой сетки.
    """

    _epochs: int
    """
    Число прогонов
    """

    _train_dataset_len: int
    """
    Размер датасета
    """

    _validation_dataset_len: int
    """
    Размер датасета для валидации
    """

    _output_folder: str
    """
    Папка, в которую будет записываться все данные, связанные с тестированием.
    """

    def __init__(
        self,
        output_folder: str,
        epochs: int = 10,
        train_dataset: int = 100,
        validation_dataset: int = 100,
    ):
        """
        Конструктор тестировщика.

        :param output_folder: Папка, в которую происходит запись моделей, данных о тестировании и датасета.
            Если путь не существует, то во время инициализации он будет создана.
            Путь к папке не должен содержать слэша в конце! Например, вместо ./test/ должен идти ./test
        :param epochs: Число эпох.
        :param train_dataset: Размер датасета для обучения.
        :param validation_dataset: Размер датасета для финальной валидации, для сбора метрик.
        :return:
        """
        if output_folder[len(output_folder) - 1] == "/":
            raise ValueError("Путь к папке не должен содержать слэша в конце")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self._output_folder = output_folder
        self._epochs = epochs
        self._train_dataset_len = train_dataset
        self._validation_dataset_len = validation_dataset

    def test(self,
            encoder_name: List[str] = [None],
            loss_fn:  List[Callable[[Tensor, Tensor], Tensor]] = [None],
            encoder_weights:  List[str] = [None],
    ) -> list[dict[str, str]]:
        """
        Функция дла тестирования нейросетки на заданных параметрах.
        Все параметры - массив возможных опций для тренировки сетки по заданным параметрам.
        Если какой-то параметр не указан, то будет использован стандартный для нейронки параметр,
        Стандартный параметр создается в единичном виде, а не в виде массива.


        Если все параметры поданы, то число прогонов нейросети будет число, равное умножению длин всех массивов.
        :param encoder_name: список энкодеров,
        :param loss_fn: список функций потери,
        :param encoder_weights: список энкодеров с весами.
        :return: Массив словарей, где указаны параметры каждого прогона, путь к сохраненным нейронкам, и прочая информация.
            Также эта информация будет записана в указанную, в конструкторе, папку.
        """

        i = 0
        csv_data = []
        for t in itertools.product(encoder_name, loss_fn, encoder_weights):
            print(f"Обрабатываем {i}/{len(encoder_weights) * len(encoder_name) * len(loss_fn)} вариантов")
            kwargs = {
                "encoder_weights": t[0],
                "loss_fn": t[1],
                "encoder_name": t[2],
            }
            metrics, model = self._test_nn(**kwargs)
            out_path = f"{self._output_folder}/model_test_{i}"

            out_str = self._convert_to_string(**{
                **kwargs,
                **metrics,
                "path_id": out_path,
                "test_id": i,
                "epochs": self._epochs,
                "validation_dataset_len": self._validation_dataset_len,
                "train_dataset_len": self._train_dataset_len
            })
            model.save(out_path)
            csv_data.append(out_str)
            i += 1

        self._to_csv(csv_data)
        return csv_data

    def _to_csv(self, data: list[dict[str,Any]]) -> None:
        """
        Записывает данные в ксв файл.
        :param data: Набор данных.
        """
        with open(f"{self._output_folder}/out.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    def _test_nn(self, **kwargs) -> tuple[dict[str, Any], DMTrainModel]:
        """
        Делает один прогон нейросетки и возвращает результаты.
        :return: Кортеж с Метриками с прогона и моделью.
        """
        model = DMTrainModel(
            encoder_name= kwargs.get("encoder_weights", "efficientnet-b3") or "efficientnet-b3",
            loss_fn= kwargs.get("loss_fn", loss.hard_loss) or loss.hard_loss,
            encoder_weights= kwargs.get("encoder_name", "imagenet") or "imagenet"
        )
        model.start_training(self._epochs, self._train_dataset_len)
        return model.test_metrics(self._validation_dataset_len), model

    def _convert_to_string(self, **kwargs) -> dict[str, str]:
        """
        Конвертирует все аргументы в строки
        :param kwargs: данные, которые будут сконвертированны в строку
        :return: Словарь, со строковыми значениями.
        """
        out = {}
        for k, v in kwargs.items():
            out[k] = kwargs[k] if isinstance(kwargs[k], str) else (kwargs[k].__name__ if callable(kwargs[k]) else str(kwargs[k]))
        return out