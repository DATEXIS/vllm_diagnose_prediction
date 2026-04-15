import argparse
import logging
import subprocess
import yaml
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
             return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

def build_and_push(image_name: str):
    """Builds and pushes the Docker container."""
    logger.info(f"Building Docker image: {image_name}")
    try:
        subprocess.run(["docker", "build", "-t", image_name, "."], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker build failed: {e}")
        sys.exit(1)

    logger.info(f"Pushing Docker image: {image_name}")
    try:
        subprocess.run(["docker", "push", image_name], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker push failed: {e}")
        sys.exit(1)
        
    logger.info("Successfully built and pushed image!")

def main():
    parser = argparse.ArgumentParser(description="Build and push the inference client Docker image.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    
    try:
        image_name = config['k8s']['client']['image']
    except KeyError:
        logger.error("Configuration missing 'k8s.client.image' key.")
        sys.exit(1)

    if image_name == "your-registry/vllm_diagnose_prediction:latest" or "your-user" in image_name:
        logger.warning("You are using the default image placeholder. Please update 'k8s.client.image' in your config!")
        # We'll allow it to attempt the build locally anyway to test the context, but push will likely fail.

    build_and_push(image_name)

if __name__ == "__main__":
    main()
