import logging

import torch
import numpy as np
import segmentation_models_pytorch as smp

def main():
    log = logging.getLogger(__name__)
    client = None
    try:
        client = listen_port(1234)
    except Exception:
        log.critical()

    model = smp.UnetPlusPlus(
        # TODO: Нужно поиграться с настройками
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    while True:
        img = recv_image(client)
        recovered_img = recover_image(model, img)
        send_image(client)


def recv_image(client: ) -> np.ndarray:
    """
    Принимает изображение от клиента (сервиса распознавания), переводит его в
    numpy.ndarray и возвращает
    """
    ...

def send_image(client: , img: np.ndarray):
    """
    Отправляет изображение клиенту
    """
    ...

def recover_image(model: torch.nn.Module, img: np.ndarray) -> np.ndarray:
    """
    Восстанавливает изображение 'img' через нейросеть 'model'
    """
    ...

if __name__ == "__main__":
    main()
