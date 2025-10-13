# Multi-stage build for optimized diarasr API image
# Builder stage: Install Python dependencies
FROM nvidia/cuda:12.8-runtime-ubuntu22.04 AS builder

# Install Python and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir --user -r requirements.txt

# Runtime stage: Minimal image with CUDA runtime
FROM nvidia/cuda:12.8-runtime-ubuntu22.04

# Install Python runtime only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local/lib/python3.11/site-packages /root/.local/lib/python3.11/site-packages
COPY --from=builder /root/.local/bin /root/.local/bin

# Add Python user packages to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy application code
COPY app.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --user-group --uid 1001 app \
    && chown -R app:app /root/.local

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]