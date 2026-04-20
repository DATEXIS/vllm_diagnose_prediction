import os
import subprocess
import yaml
import logging
import sys
from jinja2 import Template

logger = logging.getLogger(__name__)

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merges override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(setup_path="configs/setup.yaml", experiment_path="configs/experiment.yaml"):
    """Loads setup config and merges experiment config (experiment overrides setup)."""
    try:
        with open(setup_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load setup config from {setup_path}: {e}")
        sys.exit(1)

    try:
        with open(experiment_path, "r") as f:
            experiment_cfg = yaml.safe_load(f)
            if experiment_cfg:
                config = deep_merge(config, experiment_cfg)
    except FileNotFoundError:
        logger.warning(f"Experiment config not found at {experiment_path}, using setup config only.")
    except Exception as e:
        logger.warning(f"Failed to load experiment config: {e}")

    return config

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
