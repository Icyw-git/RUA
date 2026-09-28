#!/usr/bin/env bash
set -euo pipefail

export RUA_ROOT="${RUA_ROOT:-/data1/wcz}"
export RUA_VENV_PYTHON="${RUA_VENV_PYTHON:-$RUA_ROOT/conda-envs/rua-sim/bin/python}"
source "$(dirname "$0")/../runtime_env.sh"

export LD_LIBRARY_PATH="$RUA_OSMESA_LIB_DIR:$RUA_RUNTIME/egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LIBERO_CONFIG_PATH=/data1/hny/experiment-data/configs/libero-pro
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUA_CODE/vendor/show_harness:$RUA_CODE:$RUA_CODE/scripts:/data1/hny/experiment-data/benchmarks/LIBERO-PRO/libero:$WLA_ROOT/experiments/libero"

cd "$WLA_ROOT"
exec "$RUA_VENV_PYTHON" -m data_dump.oracle_feedback "$@"
