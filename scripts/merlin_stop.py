import logging
from utils import load_config, render_k8s_template, run_kubectl
from k8s_templates import client_template

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def stop_merlin():
    logger.info("Initializing MERLIN pipeline shutdown...")
    
    # 1. Load config
    cfg = load_config()
    
    # 2. Render manifest
    yaml_str = render_k8s_template(cfg, client_template)
    
    # 3. Delete from cluster
    logger.info("Deleting dynamic MERLIN manifest from cluster...")
    run_kubectl(yaml_str, command="delete")
    
    logger.info("MERLIN job stopped.")

if __name__ == "__main__":
    stop_merlin()
