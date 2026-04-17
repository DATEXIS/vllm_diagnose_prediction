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

def build_and_push(image_uri: str, platform: str):
    """Builds and pushes the Docker container."""
    logger.info(f"Building Docker image for platform {platform}: {image_uri}")
    try:
        subprocess.run(["docker", "build", "--platform", platform, "-t", image_uri, "."], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker build failed: {e}") 
        sys.exit(1)

    logger.info(f"Pushing Docker image: {image_uri}")
    try:
        subprocess.run(["docker", "push", image_uri], check=True)
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
        docker_cfg = config['docker']
        registry = docker_cfg['registry']
        image_name = docker_cfg['image_name']
        tag = docker_cfg['tag']
        platform = docker_cfg.get('platform', 'linux/amd64')
        image_uri = f"{registry}/{image_name}:{tag}"
    except KeyError as e:
        logger.error(f"Configuration missing key: {e}")
        sys.exit(1)

    build_and_push(image_uri, platform)

if __name__ == "__main__":
    main()
