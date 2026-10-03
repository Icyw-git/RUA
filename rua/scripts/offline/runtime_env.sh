#!/usr/bin/env bash
# Shared environment for offline replay and quality export.
RUA_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
RUA_CHECKOUT_ROOT="$(cd "$RUA_CODE/.." && pwd -P)"
export RUA_ROOT="${RUA_ROOT:-$(dirname "$RUA_CHECKOUT_ROOT")}"
export RUA_VENV_PYTHON="${RUA_VENV_PYTHON:-$RUA_ROOT/conda-envs/rua-sim/bin/python}"
source "$RUA_CODE/scripts/runtime_env.sh"

if [[ ! -x "$RUA_VENV_PYTHON" ]]; then
    printf 'Offline Python is not executable: %s\nSet RUA_VENV_PYTHON to the simulator Python.\n' "$RUA_VENV_PYTHON" >&2
    return 1
fi
if [[ -n "${LIBERO_PYTHONPATH:-}" && ! -d "$LIBERO_PYTHONPATH" ]]; then
    printf 'LIBERO_PYTHONPATH is not a directory: %s\n' "$LIBERO_PYTHONPATH" >&2
    return 1
fi

export LD_LIBRARY_PATH="$RUA_OSMESA_LIB_DIR:$RUA_RUNTIME/egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$RUA_ROOT/configs/libero}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUA_CODE/vendor/show_harness:$RUA_CODE:$RUA_CODE/scripts${LIBERO_PYTHONPATH:+:$LIBERO_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"

cd "$RUA_CODE"
