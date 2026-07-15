#!/bin/sh

pip install -r requirements.txt
python -m grpc_tools.protoc \
    -I./ \
    --python_out=. \
    --pyi_out=. \
    --grpc_python_out=. \
    src/irec.proto
