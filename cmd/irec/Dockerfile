FROM python:trixie

WORKDIR /irec

COPY requirements.txt .
COPY src/irec.proto src/.

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m grpc_tools.protoc \
    -I./ \
    --python_out=. \
    --pyi_out=. \
    --grpc_python_out=. \
    src/irec.proto

COPY src/ src/
COPY config.json .

CMD ["python", "-m", "src"]
