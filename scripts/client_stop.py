import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stop_client():
    logger.info("Stopping Inference Client...")
    try:
        subprocess.run(["kubectl", "delete", "-f", "k8s/inference_job.yaml"], check=True)
        logger.info("Inference Job deleted successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop Inference Job: {e}")
        # Not exiting with 1 here because it might already be deleted during a restart

if __name__ == "__main__":
    stop_client()
