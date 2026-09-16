#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh"
export LD_LIBRARY_PATH="$RUA_RUNTIME/egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$RUA_CODE/configs/egl_nvidia.json"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES="$RUA_GPU"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$RUA_ROOT/configs/libero}"
mode="${1:?Usage: run_stage1.sh probe|test|models|verify-models|verify-runtime|infer|debug|pilot|audit|replay [arguments]}"
shift
case "$mode" in
    probe)
        exec timeout 180 "$RUA_VENV_PYTHON" "$RUA_CODE/scripts/probe_libero.py" "$@"
        ;;
    test)
        cd "$RUA_CODE"
        exec "$RUA_VENV_PYTHON" -m pytest -q -p no:cacheprovider tests "$@"
        ;;
    models)
        exec "$RUA_VENV_PYTHON" "$RUA_CODE/scripts/prepare_models.py" "$@"
        ;;
    verify-runtime)
        exec "$RUA_VENV_PYTHON" "$RUA_CODE/scripts/verify_runtime.py" "$@"
        ;;
    verify-models)
        exec "$RUA_VENV_PYTHON" "$RUA_CODE/scripts/verify_models.py" "$@"
        ;;
    infer)
        exec timeout --kill-after=30s 900 "$RUA_VENV_PYTHON" "$RUA_CODE/scripts/probe_wla.py" "$@"
        ;;
    debug)
        exec timeout --kill-after=30s 1020 "$RUA_VENV_PYTHON" "$RUA_CODE/scripts/eval_native.py" --debug "$@"
        ;;
    pilot)
        exec timeout --kill-after=30s 13800 "$RUA_VENV_PYTHON" "$RUA_CODE/scripts/eval_native.py" "$@"
        ;;
    audit)
        exec "$RUA_VENV_PYTHON" "$RUA_CODE/scripts/audit_native.py" "$@"
        ;;
    replay)
        exec timeout --kill-after=15s 240 "$RUA_VENV_PYTHON" "$RUA_CODE/scripts/replay_native.py" "$@"
        ;;
    *)
        printf 'Unknown mode: %s\n' "$mode" >&2
        exit 2
        ;;
esac
