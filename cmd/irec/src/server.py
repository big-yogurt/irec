import sys
import json
import logging

import cv2
import numpy as np
import grpc
from . import irec_pb2
from . import irec_pb2_grpc
import tritonclient.grpc.aio as grpcclient

from . import adaptive_grid


logging.basicConfig(level=logging.INFO)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def serve(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    server = grpc.aio.server()
    irec_pb2_grpc.add_IrecServicer_to_server(
        ImageRecoveryServicer(config["triton"]), server
    )
    host = config["server"]["host"]
    port = config["server"]["port"]
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


ext_map = {
    "image/webp": ".webp",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


class ImageRecoveryServicer(irec_pb2_grpc.IrecServicer):
    def __init__(self, config: dict):
        self._triton_client = _get_triton_client(config["url"])
        self._config = config

    async def RecoveryImage(
            self,
            request: irec_pb2.RecoveryRequest,
            context: grpc.ServicerContext
            ) -> irec_pb2.RecoveryResponse:
        ext = ext_map.get(request.mime_type)
        if ext is None:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Формат изображения не поддерживается"
            )

        img = np.frombuffer(request.image, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR_RGB)
        if img is None:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Невозможно декодировать изображение"
            )

        img = await self._recover_img(img)
        is_ok, img = cv2.imencode(ext, img)
        if not is_ok:
            context.abort(
                grpc.StatusCode.INTERNAL,
                "Ошибка кодирования изображения"
            )

        return irec_pb2.RecoveryResponse(
            image=img.tobytes(),
            mime_type=request.mime_type
        )

    async def _recover_img(self, img: np.ndarray) -> np.ndarray:
        # Предобработка
        img = cv2.resize(img, (256, 256))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Отправка на Triton
        inputs = grpcclient.InferInput(
            self._config["input_name"], img.shape, "FP32"
        )
        inputs.set_data_from_numpy(img)
        outputs = grpcclient.InferRequestedOutput(self._config["output_name"])
        response = await self._triton_client.infer(
            model_name=self._config["model_name"],
            model_version=self._config["model_version"],
            inputs=[inputs],
            outputs=[outputs]
        )
        output = response.as_numpy(self._config["output_name"])

        # Приведение к (H, W, C) в диапазоне [0, 255]
        output = np.squeeze(output).astype(np.uint8) * 255

        # Постобработка
        output = adaptive_grid.reconstruct(output)

        return output


def _get_triton_client(url: str) -> grpcclient.InferenceServerClient:
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=url,
        )
    except Exception as e:
        print("Ошибка создания канала triton: " + str(e))
        sys.exit()
    return triton_client
