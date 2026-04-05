# Dockerfile for DCASS Steganography System
# Multi-stage build for optimized image size

FROM python:3.10-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
FROM base AS dependencies

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional RL/GAN dependencies
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
    gymnasium \
    tensorboard \
    git+https://github.com/openai/CLIP.git

# Final stage
FROM dependencies AS final

# Copy application code
COPY . /app

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Create directories for data and checkpoints
RUN mkdir -p /app/data \
    /app/checkpoints \
    /app/logs \
    /app/shared_channel

# Expose ports for monitoring
EXPOSE 8888 6006

# Default command (can be overridden)
CMD ["python", "-u", "src/cli/main.py"]
