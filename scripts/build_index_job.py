import logging
import subprocess
import sys
import time
from utils import load_config, render_k8s_template, run_kubectl
from k8s_templates import index_builder_template

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

POLL_INTERVAL = 10
TIMEOUT = 7200  # 2 hours


def wait_for_job(namespace: str, job_name: str) -> bool:
    logger.info(f"Waiting for job '{job_name}' to complete (timeout: {TIMEOUT}s)...")
    for _ in range(0, TIMEOUT, POLL_INTERVAL):
        result = subprocess.run(
            ["kubectl", "get", "job", job_name, "-n", namespace,
             "-o", "jsonpath={.status.conditions[*].type}"],
            capture_output=True, text=True,
        )
        conditions = result.stdout.strip().split()
        if "Complete" in conditions:
            logger.info(f"Job '{job_name}' completed successfully.")
            return True
        if "Failed" in conditions:
            logger.error(f"Job '{job_name}' failed.")
            return False
        time.sleep(POLL_INTERVAL)
    logger.error(f"Job '{job_name}' did not complete within {TIMEOUT}s.")
    return False


def print_job_logs(namespace: str, job_name: str):
    result = subprocess.run(
        ["kubectl", "logs", f"job/{job_name}", "-n", namespace],
        capture_output=True, text=True,
    )
    if result.stdout:
        logger.error(f"Job logs:\n{result.stdout}")


def build_index():
    cfg = load_config()

    if not cfg.get("rag", {}).get("enabled"):
        logger.info("RAG not enabled, skipping index build.")
        return

    namespace = cfg["k8s"]["namespace"]
    job_name = f"index-builder-{cfg['job_name']}"

    # Remove any previous run of this job so we can resubmit cleanly
    subprocess.run(
        ["kubectl", "delete", "job", job_name, "-n", namespace, "--ignore-not-found"],
        capture_output=True,
    )

    yaml_str = render_k8s_template(cfg, index_builder_template)
    run_kubectl(yaml_str, command="apply")
    logger.info(f"Index builder job '{job_name}' submitted.")

    success = wait_for_job(namespace, job_name)
    if not success:
        print_job_logs(namespace, job_name)
        sys.exit(1)


if __name__ == "__main__":
    build_index()
