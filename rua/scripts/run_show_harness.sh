#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh"
export LD_LIBRARY_PATH="$RUA_RUNTIME/egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RUA_RENDER_BACKEND="${RUA_RENDER_BACKEND:-osmesa}"
case "$RUA_RENDER_BACKEND" in
  egl)
    export __EGL_VENDOR_LIBRARY_FILENAMES="$RUA_CODE/configs/egl_nvidia.json"
    export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES="$RUA_GPU"
    ;;
  osmesa)
    test -f "$RUA_OSMESA_LIB_DIR/libOSMesa.so.8"
    export LD_LIBRARY_PATH="$RUA_OSMESA_LIB_DIR:$LD_LIBRARY_PATH"
    unset __EGL_VENDOR_LIBRARY_FILENAMES MUJOCO_EGL_DEVICE_ID
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES=""
    export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2
    ;;
  *) echo "Unsupported RUA_RENDER_BACKEND" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$RUA_ROOT/configs/libero}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$RUA_CODE/vendor/show_harness:$RUA_CODE:$RUA_CODE/scripts:$WLA_ROOT/LIBERO:$WLA_ROOT/experiments/libero${PYTHONPATH:+:$PYTHONPATH}"
mode="${1:?Usage: run_show_harness.sh setup-local|run|test|calibrate|probe-renderer|probe-model|paired|audit|wla-debug|handoff-check}"
shift
case "$mode" in
  run)
    test "$RUA_RENDER_BACKEND" = osmesa
    exec timeout --kill-after=30s 1250 "$RUA_VENV_PYTHON" -m libero_harness.portable_run "$@"
    ;;
  setup-local)
    exec "$RUA_VENV_PYTHON" -m libero_harness.portable_setup "$@"
    ;;
  test)
    cd "$RUA_CODE"
    exec "$RUA_VENV_PYTHON" -m pytest -q -p no:cacheprovider tests_stage2 tests_data_dump "$@"
    ;;
  calibrate)
    # Full 48-probe CPU-rendered installation calibration; not an episode budget.
    exec timeout --kill-after=15s 900 "$RUA_VENV_PYTHON" -m libero_harness.calibrate "$@"
    ;;
  probe-renderer)
    exec timeout --kill-after=15s 120 "$RUA_VENV_PYTHON" -m libero_harness.render_info
    ;;
  probe-model)
    exec timeout --kill-after=15s 180 "$RUA_VENV_PYTHON" -m libero_harness.probe_model "$@"
    ;;
  debug|pilot)
    exec timeout --kill-after=30s 14000 "$RUA_VENV_PYTHON" -m libero_harness.runner "$mode" "$@"
    ;;
  wla-debug)
    exec timeout --kill-after=30s 1250 "$RUA_VENV_PYTHON" -m libero_harness.validate_wla "$@"
    ;;
  hybrid-debug)
    test "$RUA_RENDER_BACKEND" = osmesa
    exec timeout --kill-after=30s 1250 "$RUA_VENV_PYTHON" -m libero_harness.hybrid_debug "$@"
    ;;
  paired)
    test "$RUA_RENDER_BACKEND" = osmesa
    exec "$RUA_VENV_PYTHON" -m libero_harness.paired "$@"
    ;;
  paired-batch)
    test "$RUA_RENDER_BACKEND" = osmesa
    exec "$RUA_VENV_PYTHON" "$RUA_CODE/scripts/paired_driver.py" "$@"
    ;;
  handoff-check)
    test "$RUA_RENDER_BACKEND" = osmesa
    exec timeout --kill-after=15s 400 "$RUA_VENV_PYTHON" -m libero_harness.check_handoff "$@"
    ;;
  switch-diagnostic)
    test "$RUA_RENDER_BACKEND" = osmesa
    exec timeout --kill-after=15s 500 "$RUA_VENV_PYTHON" -m libero_harness.diagnose_switch "$@"
    ;;
  audit)
    exec "$RUA_VENV_PYTHON" -m libero_harness.audit "$@"
    ;;
  *) exit 2 ;;
esac
