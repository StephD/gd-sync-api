FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# Cloud Run (and Render) inject PORT - bind to it, not a fixed port
CMD exec gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
