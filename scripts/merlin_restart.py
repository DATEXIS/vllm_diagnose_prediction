import logging
import time

from merlin_stop import stop_merlin
from merlin_start import start_merlin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restart_merlin():
    logger.info("Restarting MERLIN pipeline...")
    stop_merlin()
    
    # Wait a moment to ensure Kubernetes processes the deletion before re-applying
    logger.info("Waiting 3 seconds for pod termination...")
    time.sleep(3)
    
    start_merlin()
    logger.info("Restart complete.")

if __name__ == "__main__":
    restart_merlin()
