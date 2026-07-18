FROM python:3.12-slim

WORKDIR /app

# opencv-python (a rapidocr dependency) needs these X11/GL shared libs at
# import time - python:3.12-slim doesn't ship them (confirmed live on Cloud
# Run: "libxcb.so.1: cannot open shared object file", worker never boots)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# Cloud Run (and Render) inject PORT - bind to it, not a fixed port
CMD exec gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
