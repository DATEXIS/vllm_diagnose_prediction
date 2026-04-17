import os
import subprocess
import yaml
import logging
import sys
from jinja2 import Template

logger = logging.getLogger(__name__)

def load_config(config_path="configs/config.yaml"):
    """Loads the main project configuration."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)

def render_k8s_template(cfg, template_str):
    """Renders a Jinja2 K8s template string using the provided config."""
    tmpl = Template(template_str)
    return tmpl.render(cfg=cfg)

def run_kubectl(yaml_str, command="apply"):
    """Pipes a YAML string directly to kubectl (apply or delete)."""
    cmd = ["kubectl", command, "-f", "-"]
        
    try:
        # We use subprocess.run with input to pipe the YAML string
        process = subprocess.run(
            cmd,
            input=yaml_str,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(process.stdout.strip())
    except subprocess.CalledProcessError as e:
        logger.error(f"Kubectl {command} failed: {e.stderr.strip()}")
        # We don't necessarily exit(1) on delete as it might already be gone
        if command == "apply":
            sys.exit(1)
