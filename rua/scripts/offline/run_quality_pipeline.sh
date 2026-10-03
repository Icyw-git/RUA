#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/runtime_env.sh"
exec "$RUA_VENV_PYTHON" -m data_dump.quality_pipeline "$@"
