import argparse
import logging
import subprocess
import sys
import os
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merges override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config():
    """Loads setup config and merges experiment config."""
    base_path = "configs"
    setup_path = os.path.join(base_path, "setup.yaml")
    experiment_path = os.path.join(base_path, "experiment.yaml")

    try:
        with open(setup_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load setup config: {e}")
        sys.exit(1)

    try:
        with open(experiment_path, "r") as f:
            experiment_cfg = yaml.safe_load(f)
            if experiment_cfg:
                config = deep_merge(config, experiment_cfg)
    except FileNotFoundError:
        logger.warning(f"Experiment config not found, using setup only.")
    except Exception as e:
        logger.warning(f"Failed to load experiment config: {e}")

    return config


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
    args = parser.parse_args()

    config = load_config()
    
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
