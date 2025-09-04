#!/usr/bin/env bash
set -euo pipefail
# Gunicorn will import deploy/inference.py -> app
exec gunicorn -b 0.0.0.0:8080 inference:app
