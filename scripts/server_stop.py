import logging
from utils import load_config, render_k8s_template, run_kubectl
from k8s_templates import server_template

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def stop_server():
    logger.info("Initializing vLLM Server shutdown...")
    
    # 1. Load config
    cfg = load_config()
    
    # 2. Render manifest
    yaml_str = render_k8s_template(cfg, server_template)
    
    # 3. Delete from cluster
    logger.info(f"Deleting dynamic server manifest from cluster (namespace: {cfg['k8s']['namespace']})...")
    run_kubectl(yaml_str, command="delete")
    
    logger.info("Server stopped.")

if __name__ == "__main__":
    stop_server()
