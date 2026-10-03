#!/usr/bin/env bash
# Source from task launchers only; never modify the user's shell configuration.
RUA_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WLA_ROOT="$(cd "${WLA_ROOT:-$RUA_CODE/..}" && pwd -P)"
RUA_ROOT="${RUA_ROOT:-$(dirname "$WLA_ROOT")}"
RUA_RUNTIME="${RUA_RUNTIME:-$RUA_ROOT/runtime/wla-libero}"
export RUA_CODE WLA_ROOT RUA_ROOT RUA_RUNTIME
export RUA_GPU="${RUA_GPU:-0}"
export RUA_VENV_PYTHON="${RUA_VENV_PYTHON:-$RUA_RUNTIME/venv/bin/python}"
export RUA_OSMESA_LIB_DIR="${RUA_OSMESA_LIB_DIR:-$RUA_RUNTIME/osmesa-24.0.5/usr/lib/x86_64-linux-gnu}"
test ! -L "$RUA_RUNTIME" || {
    printf 'Refusing a symlinked task runtime: %s\n' "$RUA_RUNTIME" >&2
    return 1
}
RUA_UV="$RUA_RUNTIME/tools/uv"
export UV_PYTHON_INSTALL_DIR="$RUA_RUNTIME/python"
RUA_PYTHON="$UV_PYTHON_INSTALL_DIR/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11"
export UV_CACHE_DIR="$RUA_RUNTIME/uv-cache"
# Cache and venv share a filesystem. uv safely falls back if links are unsupported.
export UV_LINK_MODE=hardlink
export UV_CONCURRENT_DOWNLOADS=3
export UV_HTTP_TIMEOUT=180
export XDG_CACHE_HOME="$RUA_RUNTIME/cache"
export XDG_CONFIG_HOME="$RUA_RUNTIME/config"
export TMPDIR="$RUA_RUNTIME/tmp"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TORCH_HOME="$RUA_ROOT/models/torch"
export HF_HOME="$RUA_ROOT/models/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
# Use the user-selected HTTPS mirror's normal HTTP download path.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
