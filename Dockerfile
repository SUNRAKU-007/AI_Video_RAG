FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

ENV HF_HOME=/app/model_cache/huggingface
ENV TORCH_HOME=/app/model_cache/torch
ENV XDG_CACHE_HOME=/app/model_cache

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first
RUN python -m pip install --upgrade pip \
    && python -m pip install \
       --retries 10 \
       --timeout 120 \
       --index-url https://download.pytorch.org/whl/cpu \
       "torch>=2.2.0,<3.0.0" \
    && python -m pip install \
       --retries 10 \
       --timeout 120 \
       -r requirements.txt

# Copy application
COPY . .

# Create persistent storage directories
RUN mkdir -p \
    /app/downloads \
    /app/vector_db \
    /app/model_cache \
    /app/input

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]