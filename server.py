import json
import logging

import cv2
import numpy as np
import grpc
import irec_pb2
import irec_pb2_grpc


logging.basicConfig(level=logging.INFO)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def serve() -> None:
    config = load_config("config.json")
    server = grpc.aio.server()
    irec_pb2_grpc.add_IrecServicer_to_server(ImageRecoveryServicer(), server)
    listen_addr = f"{config['server']['host']}:{config['server']['port']}"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


ext_map = {
    "image/webp": ".webp",
}


class ImageRecoveryServicer(irec_pb2_grpc.IrecServicer):
    async def RecoveryImage(self, request: irec_pb2.RecoveryRequest,
            context: grpc.ServicerContext) -> irec_pb2.RecoveryResponse:
        ext = ext_map.get(request.mime_type)
        if ext is None:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                "Формат изображения не поддерживается"
            )
        img = np.frombuffer(request.image, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        if img is None:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                "Некорректные данные изображения"
            )
        # TODO: обработка изображения
        # img = ...
        ok, img = cv2.imencode(ext, img)
        if not ok:
            context.abort(grpc.StatusCode.INTERNAL,
                "Ошибка кодировки изображения"
            )
        return irec_pb2.RecoveryResponse(image=img, mime_type=request.mime_type)
