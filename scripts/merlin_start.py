import logging
from utils import load_config, render_k8s_template, run_kubectl
from k8s_templates import client_template

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def start_merlin():
    logger.info("Initializing MERLIN pipeline deployment...")
    
    # 1. Load config
    cfg = load_config()
    
    # 2. Render manifest
    yaml_str = render_k8s_template(cfg, client_template)
    
    # 3. Apply to cluster
    logger.info(f"Applying dynamic MERLIN manifest to cluster (namespace: {cfg['k8s']['namespace']})...")
    run_kubectl(yaml_str, command="apply")
    
    logger.info("MERLIN job started successfully.")

if __name__ == "__main__":
    start_merlin()
