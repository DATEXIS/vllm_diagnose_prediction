import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stop_server():
    logger.info("Stopping vLLM Server...")
    try:
        subprocess.run(["kubectl", "delete", "-f", "k8s/vllm_server.yaml"], check=True)
        logger.info("vLLM Server manifest deleted successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop vLLM Server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    stop_server()
