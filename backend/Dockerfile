# backend/Dockerfile

FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY app ./app

EXPOSE 8000

CMD ["/bin/sh", "-c", "echo 'Backend running on http://localhost:8000 (mapped via docker-compose)'; uvicorn app.main:app --host 0.0.0.0 --port 8000"]