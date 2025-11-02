# Single-stage build for diarasr API image with global Python package installation
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

# Install Python 3.12.3 (dev includes runtime), build tools, and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-dev \
    python3-pip \
    build-essential \
    git \
    ffmpeg \
    libavcodec-extra \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Set Python path (skip pip upgrade to avoid system package conflicts)
ENV PYTHONPATH="/app"

# Copy requirements and install Python packages globally
COPY app/requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt --break-system-packages \
    && apt-get remove -y python3.12-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app/ /app/

# Set Hugging Face cache directory for runtime model downloads
# Models will be downloaded on first use and cached in a Docker volume
ENV HF_HOME=/home/app/.cache/huggingface

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --user-group --uid 1001 app

# Create writable directories for read-only filesystem (before switching user)
RUN mkdir -p /tmp /var/tmp /app/tmp /app/logs /home/app/.lhotse /home/app/.cache/huggingface \
    && chown -R app:app /tmp /var/tmp /app/tmp /app/logs /home/app/.lhotse /home/app/.cache

# Switch to non-root user
USER app

# Set working directory
WORKDIR /app

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Start the application
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]