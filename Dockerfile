FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    "tensorflow>=2.6.0,<2.16" \
    "numpy<1.24" \
    && pip install --no-cache-dir .

ENTRYPOINT ["deepDNAshape"]
