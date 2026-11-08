FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
# BuildKit cache mount: downloaded wheels stay cached between builds,
# so re-installs (after requirements.txt changes) are much faster.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt
# torch (required to serve the LSTM/GRU checkpoints) from the CPU-only wheel
# index - considerably smaller than the default PyPI CUDA-bundled wheel.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
