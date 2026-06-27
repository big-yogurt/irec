from statistics import mean
from typing import Any

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
import pytorch_lightning as pl
from segmentation_models_pytorch.losses import DiceLoss
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from synthetic import DMSyntheticDataset, DMSyntheticItem
from pytorch_lightning.callbacks import RichProgressBar

EPOCHS = 10
## train length should be
## T_MAX = EPOCHS * len(train_loader)
DATASET_SIZE = 100
T_MAX = EPOCHS * DATASET_SIZE

class DMTrainModel(pl.LightningModule):
    """
    Класс модели.
    """

    model: smp.UnetPlusPlus
    """ 
    Используемая модель
    """

    loss_fn: DiceLoss
    """ 
    Функция, которая измеряет, насколько вычисления модели отклонились от настоящего результатата
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
    def __init__(self):
        super().__init__()
        # Инициализация модельки
        encoder_name = "efficientnet-b3"
        self.model = smp.UnetPlusPlus(
            # TODO: Нужно поиграться с настройками
             encoder_name=encoder_name,
             encoder_weights=None,
             in_channels=3,
             classes=1,
             activation=None,
        )
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.mean = torch.tensor(params["std"]).view(1, 3, 1, 1)
        self.std  = torch.tensor(params["mean"]).view(1, 3, 1, 1)

    ## Для pl.LightningModule
    def forward(self, data: DMSyntheticItem):
        """
        Скармливание данных нейронной сети и получение выхлопа
        """
        # нормализация картиночки
        image = (data["input_tensor"] - self.mean) / self.std
        return self.model(image)

    ## Для pl.LightningModule
    def training_step(self, batch: DMSyntheticItem, batch_idx: int):
        """
        Процесс обучения сетки, вызывается во время обучения
        """
        image = batch["input_tensor"]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["target_tensor"]
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0


        predicted_mask = self.forward(batch)

        loss = self.loss_fn(predicted_mask, batch["target_tensor"])
        return loss

    ## Для pl.LightningModule
    def test_step(self, train_batch: DMSyntheticItem, batch_idx: int):
        """
        Процесс проверки сетки, вызывается после обучения
        """
        predicted_mask = self.forward(train_batch)
        loss = self.loss_fn(predicted_mask, train_batch["target_tensor"])
        return loss

    ## Для pl.LightningModule
    def validation_step(self, valid_batch: DMSyntheticItem, batch_idx: int):
        """
        Процесс Валидации сетки, вызывается во время обучения
        """
        predicted_mask = self.forward(valid_batch)
        loss = self.loss_fn(predicted_mask, valid_batch["target_tensor"])

        tp, fp, fn, tn = smp.metrics.get_stats(
            predicted_mask.long(), valid_batch["target_tensor"].long(), mode="binary"
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
            "micro_acc": dataset_iou,
            "per_image_acc": per_image_iou
        }, on_step=False, on_epoch=True, prog_bar=True, )

        self.validation_step_outputs.clear()

    ## Для pl.LightningModule
    def configure_optimizers(self):
        """
        Параметры оптимизации модели во время её обучения
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def start_training(self):
        """
        Запуск тренировки модели
        """
        # Датасеты
        train_dataset = DMSyntheticDataset(dataset_len=DATASET_SIZE).create_dataloader(batch_size=64)
        val_dataset = DMSyntheticDataset(dataset_len=DATASET_SIZE).create_dataloader(batch_size=64)

        trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1, callbacks=RichProgressBar(leave=True))

        trainer.fit(
            self,
            train_dataloaders=train_dataset,
            val_dataloaders=val_dataset,
        )

    def test(self):
        """
        Проверока модельки
        """
        ## Генерим рандомный экземпляр и сохраняем его.
        synth = DMSyntheticDataset()
        data = synth[0]

        import torchvision.transforms as T
        from PIL.Image import Image
        input: Image = T.ToPILImage()(data["input_tensor"])
        target: Image = T.ToPILImage()(data["target_tensor"])

        with torch.inference_mode():
            self.model.eval()
            logits = self.model((data["input_tensor"] - self.mean) / self.std)

        predicted: Image = T.ToPILImage()(logits.sigmoid().numpy().squeeze())

        # # display the PIL image
        input.save("in.png")
        target.save("out.png")
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
        self.model.from_pretrained(path)