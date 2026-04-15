import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_server():
    logger.info("Starting vLLM Server...")
    try:
        subprocess.run(["kubectl", "apply", "-f", "k8s/vllm_server.yaml"], check=True)
        logger.info("vLLM Server manifest applied successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start vLLM Server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()
