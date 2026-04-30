# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any essential build tools and the Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application code, prompt templates, and data.
# /app/config/config.yaml is mounted by the K8s ConfigMap; /app/configs/
# (the prompts directory) ships with the image so prompt_loader.py can find it.
COPY src/ src/
COPY configs/prompts/ configs/prompts/
COPY data/mimic data/mimic
COPY data/cooccurrence.parquet data/cooccurrence.parquet

# Set Python to run unbuffered so logs appear immediately
ENV PYTHONUNBUFFERED=1

# Command to run the inference client by default (expects config via ConfigMap mount)
CMD ["python", "src/main.py", "--config", "/app/config/config.yaml"]
