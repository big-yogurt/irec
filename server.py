import io
import json
import logging

import PIL.Image
from PIL import Image
import cv2
import torch
import torch.nn
import numpy as np
import grpc
import irec_pb2
import irec_pb2_grpc
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from torchvision.transforms.functional import to_tensor

import train


logging.basicConfig(level=logging.INFO)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def serve(path_to_nn: str) -> None:
    config = load_config("config.json")
    server = grpc.aio.server()
    irec_pb2_grpc.add_IrecServicer_to_server(
        ImageRecoveryServicer(path_to_nn), server
    )
    listen_addr = f"{config['server']['host']}:{config['server']['port']}"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


ext_map = {
    "image/webp": "WEBP",
    "image/png": "PNG",
    "image/jpeg": "JPG",
}


class ImageRecoveryServicer(irec_pb2_grpc.IrecServicer):
    def __init__(self, path_to_nn: str):
        self.model = train.DMTrainModel()
        self.model.load(path_to_nn)

    async def RecoveryImage(self, request: irec_pb2.RecoveryRequest,
            context: grpc.ServicerContext) -> irec_pb2.RecoveryResponse:
        ext = ext_map.get(request.mime_type)
        if ext is None:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                "Формат изображения не поддерживается"
            )
        img = await self._recover_img(request.image)
        buffer = io.BytesIO()
        img.save(buffer, format=ext)
        
        return irec_pb2.RecoveryResponse(image=buffer.getvalue(), mime_type=request.mime_type)

    async def _recover_img(self, img: bytes) -> Image:
        return self.model.test_img(img)
