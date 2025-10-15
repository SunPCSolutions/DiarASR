# Multi-stage build for optimized diarasr API image
# Builder stage: Install Python dependencies
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04 AS builder

# Install Python 3.12.3 and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Copy requirements and install Python packages in virtual environment
COPY app/requirements.txt .
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Runtime stage: Minimal image with CUDA runtime
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

# Install Python 3.12 runtime and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
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

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Copy application code
COPY app/ /app/

# Set Hugging Face cache directory to mounted volume location
ENV HF_HOME=/home/app/.cache/huggingface

# Pre-load ML models during build to cache them in Hugging Face cache directory
# This avoids download delays during runtime and ensures models are available
RUN cd /app && python preload_models.py

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --user-group --uid 1001 app \
    && chown -R app:app /opt/venv

# Create writable directories for read-only filesystem
RUN mkdir -p /tmp /var/tmp /app/tmp /app/logs /home/app/.lhotse \
    && chown -R app:app /tmp /var/tmp /app/tmp /app/logs /home/app/.lhotse

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
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]