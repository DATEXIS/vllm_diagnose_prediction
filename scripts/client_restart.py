import logging
import time

from client_stop import stop_client
from client_start import start_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restart_client():
    logger.info("Restarting Inference Client...")
    stop_client()
    
    # Wait a moment to ensure Kubernetes processes the deletion before re-applying
    logger.info("Waiting 3 seconds for pod termination...")
    time.sleep(3)
    
    start_client()
    logger.info("Restart complete.")

if __name__ == "__main__":
    restart_client()
