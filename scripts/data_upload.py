import logging
import subprocess
import sys
import time
from pathlib import Path
from utils import load_config, run_kubectl

PROJECT_ROOT = Path(__file__).parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PVC_TEMPLATE = """apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {pvc_name}
  namespace: {namespace}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 200Gi  # 4 precomputed chunks ≈ 19 GB + headroom for index + existing data
"""

UPLOADER_POD_TEMPLATE = """apiVersion: v1
kind: Pod
metadata:
  name: data-uploader
  namespace: {namespace}
spec:
  restartPolicy: Never
  containers:
    - name: uploader
      image: busybox
      command: ["sleep", "3600"]
      volumeMounts:
        - name: data
          mountPath: /app/data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: {pvc_name}
"""


def wait_for_pod(namespace: str, pod_name: str, timeout: int = 120):
    logger.info(f"Waiting for pod {pod_name} to be ready...")
    for _ in range(timeout):
        result = subprocess.run(
            ["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "jsonpath={.status.phase}"],
            capture_output=True, text=True
        )
        if result.stdout.strip() == "Running":
            return
        time.sleep(1)
    logger.error(f"Pod {pod_name} did not become ready within {timeout}s")
    sys.exit(1)


def kubectl_cp(src: str, dest: str):
    result = subprocess.run(["kubectl", "cp", src, dest], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"kubectl cp failed: {result.stderr.strip()}")
        sys.exit(1)


def upload_data():
    cfg = load_config()
    namespace = cfg["k8s"]["namespace"]
    pvc_name = cfg["k8s"]["pvc_name"]

    logger.info(f"Creating PVC '{pvc_name}' in namespace '{namespace}'...")
    run_kubectl(PVC_TEMPLATE.format(pvc_name=pvc_name, namespace=namespace), command="apply")

    logger.info("Starting temporary uploader pod...")
    run_kubectl(UPLOADER_POD_TEMPLATE.format(pvc_name=pvc_name, namespace=namespace), command="apply")

    wait_for_pod(namespace, "data-uploader")

    logger.info("Copying data/mimic/ to PVC...")
    kubectl_cp(str(PROJECT_ROOT / "data" / "mimic"), f"{namespace}/data-uploader:/app/data/mimic")

    logger.info("Copying data/pubmed/ to PVC...")
    kubectl_cp(str(PROJECT_ROOT / "data" / "pubmed"), f"{namespace}/data-uploader:/app/data/pubmed")

    logger.info("Cleaning up uploader pod...")
    subprocess.run(["kubectl", "delete", "pod", "data-uploader", "-n", namespace], check=True)

    logger.info("Data upload complete. PVC is ready.")


if __name__ == "__main__":
    upload_data()
