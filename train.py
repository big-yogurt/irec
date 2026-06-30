from statistics import mean
from typing import Any, Tuple, Callable, Mapping

import PIL.Image
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
import torch.nn as nn
import pytorch_lightning as pl
from segmentation_models_pytorch.losses import DiceLoss
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from loss import hard_loss
from synthetic import DMSyntheticDataset
from pytorch_lightning.callbacks import RichProgressBar
import torchvision.transforms as T
from PIL.Image import Image

class DMTrainModel(pl.LightningModule):
    """
    Класс модели.
    """

    model: smp.UnetPlusPlus
    """ 
    Используемая модель
    """

    loss_fn: Callable[[Tensor, Tensor], Tensor]
    """ 
    Функция, которая измеряет, насколько вычисления модели отклонились от настоящего результатата.
    Первый аргумент - предикт нейронки, второй - настоящая маска.
    Возращает, скаляр, обозначающий, насколько сильно нейронка ошиблась.
    """

    validation_step_outputs: list[Any] = []
    """
    Хранимые метрики, накапливаемые с одного прохода во время обучения нейросети
    """

    mean: Tensor
    """
    Среднее для нормализации изображения
    
    """
    std: Tensor
    """
    Стандартное отклонение для нормализации изображения
    """

    T_MAX: int = -1
    """
    число эпох * размер датасета, используется в оптимизаторе, во время обучения
    """
    def __init__(
            self,
            encoder_name: str = "efficientnet-b3",
            loss_fn: Callable[[Tensor, Tensor], Tensor] = hard_loss,
            encoder_weights: str = "imagenet"
    ):
        """
        Коснтруктор модельки
        :param encoder_name: имя энкодера, полный список можно посмотреть в документации segmantation models
        :param loss_fn: Функция, которая измеряет, насколько вычисления модели отклонились от настоящего результатата.
        Первый аргумент - предикт нейронки, второй - настоящая маска.
        Возращает, скаляр, обозначающий, насколько сильно нейронка ошиблась.
        :param encoder_weights: имя весового энкодера, полный список можно посмотреть в документации segmantation models
        """
        super().__init__()
        # Инициализация модельки
        self.model = smp.UnetPlusPlus(
             encoder_name=encoder_name,
             encoder_weights=encoder_weights,
             in_channels=3,
             classes=1,
             activation=None,
             decoder_norm_layer=torch.nn.InstanceNorm2d,
        )

        self.loss_fn = loss_fn
        # dice пойдет
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # самопал надо проверить работает вообще или нет
        # self.loss_fn = hard_loss
        # mse выдает шлак на 50 эпох и 500 картинках
        #self.loss_fn = nn.MSELoss()

        # тоже не лучше
        # self.loss_fn = smp.losses.FocalLoss(mode="binary")
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.mean = torch.tensor(params["std"]).view(1, 3, 1, 1)
        self.std  = torch.tensor(params["mean"]).view(1, 3, 1, 1)

    ## Для pl.LightningModule
    def forward(self, data: Tensor):
        """
        Скармливание данных нейронной сети и получение выхлопа.
        :arg data Картинка, на основе которой нейронка делает предикт
        """
        # нормализация картиночки
        image = (data - self.mean) / self.std
        return self.model(image)

    ## Для pl.LightningModule
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        """
        Процесс обучения сетки, вызывается во время обучения
        """

        assert batch[0].ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = batch[0].shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        assert batch[1].ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert batch[1].max() <= 1.0 and batch[1].min() >= 0
        predicted_mask = self.forward(batch[0])
        loss = self.loss_fn(predicted_mask, batch[1])
        return loss

    ## Для pl.LightningModule
    def test_step(self, train_batch: tuple[Tensor, Tensor], batch_idx: int):
        """
        Процесс проверки сетки, вызывается после обучения
        """
        predicted_mask = self.forward(train_batch[0])
        loss = self.loss_fn(predicted_mask, train_batch[1])
        return loss

    ## Для pl.LightningModule
    def validation_step(self, valid_batch: tuple[Tensor, Tensor], batch_idx: int):
        """
        Процесс Валидации сетки, вызывается во время обучения
        """
        predicted_mask = self.forward(valid_batch[0])
        loss = self.loss_fn(predicted_mask, valid_batch[1])

        prob_mask = predicted_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), valid_batch[1].long(), mode="binary"
        )

        out = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        self.validation_step_outputs.append(out)
        return out

    ## Для pl.LightningModule
    def on_validation_epoch_end(self) -> None:
        """
        Собираем все метрики с текущего прохода валидации
        """
        outputs = self.validation_step_outputs

        # Объединяем тенсоры для каждого показателя
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        # Собираем метрики, методом iou (Коэффициент Жаккара)
        # Метрика проверяет, насколько отличается маска созданная сеткой, и наша оригинальная.

        # Каждая картинка(один шаг) весит одинаково в метрике.
        # Пустые картинки влияют больше на результат этой метрики.
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # Каждый пиксель весит одинаково в метрике.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        self.log_dict({
            "loss": outputs[0]["loss"],
            "micro_acc": dataset_iou,
            "per_image_acc": per_image_iou
        }, on_step=False, on_epoch=True, prog_bar=True, )

        self.validation_step_outputs.clear()

    ## Для pl.LightningModule
    def configure_optimizers(self):
        """
        Параметры оптимизации модели во время её обучения
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def start_training(self, epochs: int, dataset_size: int):
        """
        Запуск тренировки модели
        """
        # Датасеты

        train_dataset = DataLoader(DMSyntheticDataset(dataset_len=dataset_size), batch_size=64, shuffle=False, num_workers=6)
        val_dataset = DataLoader(DMSyntheticDataset(dataset_len=dataset_size), batch_size=64, shuffle=False, num_workers=6, )

        trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=1, callbacks=RichProgressBar(leave=True))

        trainer.fit(
            self,
            train_dataloaders=train_dataset,
            val_dataloaders=val_dataset,
        )

    def test_metrics(self, dataset_len: int = 100) -> dict[str, float]:
        """
        Проверка модели на синтетическом датасете, возвращает метрики.
        :return: Метрики модели.
        """
        trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1, callbacks=RichProgressBar(leave=True))
        return trainer.validate(self, dataloaders=DataLoader(DMSyntheticDataset(dataset_len=dataset_len), batch_size=64, shuffle=False, num_workers=6, ), verbose=False)[0]

    def test(self):
        """
        Проверока модельки
        """
        ## Генерим рандомный экземпляр и сохраняем его.
        synth = DMSyntheticDataset()
        data = synth[0]

        import torchvision.transforms as T
        from PIL.Image import Image
        input: Image = T.ToPILImage()(data[0])
        target: Image = T.ToPILImage()(data[1])

        with torch.inference_mode():
            self.model.eval()
            logits = self.model((data[0] - self.mean) / self.std)

        predicted: Image = T.ToPILImage()(logits.sigmoid().numpy().squeeze())

        input.save("in.png")
        target.save("out.png")
        predicted.save("predicted.png")

    def test_img(self, img_path: str):
        """
        Проверка модельки на существующей картинке
        :param img_path: путь к файлу
        """
        pic = PIL.Image.open(img_path).resize((256, 256)).convert("RGB")
        img = T.ToTensor()(pic)  # (C, H, W), float [0,1]
        img = img.unsqueeze(0)  # (1, C, H, W) — батч из одной картинки

        self.model.eval()
        with torch.inference_mode():
            logits = self.forward(img)

        predicted = T.ToPILImage()(logits.sigmoid().squeeze())
        predicted.save("predicted.png")

    def save(self, path: str):
        """
        Сохраняем модельку в путь.
        """
        self.model.save_pretrained(
            save_directory=path
        )

    def load(self, path: str):
        """
        Загружаем модельку из пути.
        """
        self.model = self.model.from_pretrained(path)