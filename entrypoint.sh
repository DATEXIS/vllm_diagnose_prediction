#!/bin/bash
set -euo pipefail

pip install --no-cache-dir -r /app/requirements.txt
exec "$@"
