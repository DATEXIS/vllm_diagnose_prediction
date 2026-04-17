import logging
import sys
from utils import load_config, render_k8s_template, run_kubectl
from k8s_templates import server_template

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def start_server():
    logger.info("Initializing vLLM Server deployment...")
    
    # 1. Load config
    cfg = load_config()
    
    # 2. Render manifest
    yaml_str = render_k8s_template(cfg, server_template)
    
    # 3. Apply to cluster
    logger.info(f"Applying dynamic server manifest to cluster (namespace: {cfg['k8s']['namespace']})...")
    run_kubectl(yaml_str, command="apply")
    
    logger.info("Server started successfully.")

if __name__ == "__main__":
    start_server()
