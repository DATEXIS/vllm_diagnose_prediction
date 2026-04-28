# Use a stable Python base image
FROM python:3.10-slim

# Prevent Python from writing pyc files & enable logs immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first (important!)
RUN pip install --upgrade pip

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies (more robust)
RUN pip install \
    --no-cache-dir \
    --timeout 1000 \
    --prefer-binary \
    -r requirements.txt

# Copy project files
COPY src/ src/
COPY data/ data/

# Default command
CMD ["python", "src/main.py", "--config", "/app/config/config.yaml"]