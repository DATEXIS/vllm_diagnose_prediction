# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Build tools for pip wheels compiled at container start (vllm, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Copy the application code, prompt templates, and data.
# /app/config/config.yaml is mounted by the K8s ConfigMap; /app/configs/
# (the prompts directory) ships with the image so prompt_loader.py can find it.
COPY src/ src/
COPY configs/prompts/ configs/prompts/
COPY configs/admission_note_sections.yaml configs/admission_note_sections.yaml
COPY data/mimic data/mimic
COPY data/cooccurrence.parquet data/cooccurrence.parquet
# Optional: python scripts/build_rareness_factors.py (full-train IDF). If absent at
# build time, omit this line; runtime uses compute_rareness_at_load from config.
# COPY data/rareness_factors.parquet data/rareness_factors.parquet

# Set Python to run unbuffered so logs appear immediately
ENV PYTHONUNBUFFERED=1

# Command to run the inference client by default (expects config via ConfigMap mount)
CMD ["python", "src/main.py", "--config", "/app/config/config.yaml"]
