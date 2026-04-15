import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_client():
    logger.info("Starting Inference Client...")
    try:
        subprocess.run(["kubectl", "apply", "-f", "k8s/inference_job.yaml"], check=True)
        logger.info("Inference Job deployed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Inference Job: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_client()
