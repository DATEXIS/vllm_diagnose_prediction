import argparse
import logging
import yaml
import sys
import os
from jinja2 import Template

from k8s_templates import server_template, client_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
             return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

def generate_k8s_files(cfg: dict, out_dir: str = "k8s"):
    """Renders the Jinja templates with the provided configuration and writes them to the k8s directory."""
    os.makedirs(out_dir, exist_ok=True)

    # Render Server
    server_tmpl = Template(server_template)
    server_yaml = server_tmpl.render(cfg=cfg)
    server_path = os.path.join(out_dir, "vllm_server.yaml")
    
    with open(server_path, 'w') as f:
        f.write(server_yaml)
    logger.info(f"Generated server manifest at {server_path}")

    # Render Client
    client_tmpl = Template(client_template)
    client_yaml = client_tmpl.render(cfg=cfg)
    client_path = os.path.join(out_dir, "inference_job.yaml")
    
    with open(client_path, 'w') as f:
        f.write(client_yaml)
    logger.info(f"Generated client manifest at {client_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Kubernetes manifests from config.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Simple default checking to avoid key errors if fields are missing
    if 'k8s' not in config:
        logger.error("Configuration missing 'k8s' block.")
        sys.exit(1)

    generate_k8s_files(config)

if __name__ == "__main__":
    main()
