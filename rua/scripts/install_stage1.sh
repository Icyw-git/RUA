#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh"
LIBERO_REV=8f1084e3132a39270c3a13ebe37270a43ece2a01
RUA_PIP_INDEX="${RUA_PIP_INDEX:-https://pypi.org/simple}"

test "$(git -C "$WLA_ROOT" rev-parse HEAD)" = 5662058f3ead8a1b4fec9865acce21dd2f5efbf1
test ! -L "$RUA_RUNTIME"
mkdir -p "$RUA_RUNTIME"
chmod 700 "$RUA_RUNTIME"
if ! test -x "$RUA_UV"; then
    mkdir -p "$RUA_RUNTIME/tools"
    cp "$(command -v uv)" "$RUA_UV"
fi
if ! test -x "$RUA_PYTHON"; then
    if test "${RUA_OFFLINE:-0}" = 1; then
        printf 'Offline install requires the workspace Python input first.\n' >&2
        exit 1
    fi
    "$RUA_UV" python install 3.11.15
fi
RUA_INSTALL_ARGS=()
if test "${RUA_OFFLINE:-0}" = 1; then
    RUA_INSTALL_ARGS+=(--offline)
fi
if test -n "${RUA_CONSTRAINTS:-}"; then
    RUA_INSTALL_ARGS+=(-c "$RUA_CONSTRAINTS")
fi

# This container has NVIDIA EGL but not GLVND's dispatch library. Extract one
# checksum-pinned Ubuntu package privately; never apt-install / replace host libs.
mkdir -p "$RUA_RUNTIME/egl"
if ! test -f "$RUA_RUNTIME/libglvnd0.deb"; then
    test "${RUA_OFFLINE:-0}" != 1 || { printf 'Offline EGL package missing.\n' >&2; exit 1; }
    curl --fail --location --retry 2 --connect-timeout 10 --max-time 60 \
        https://mirrors.aliyun.com/ubuntu/pool/main/libg/libglvnd/libglvnd0_1.7.0-1build1_amd64.deb \
        --output "$RUA_RUNTIME/libglvnd0.deb"
fi
"$RUA_PYTHON" - "$RUA_RUNTIME/libglvnd0.deb" <<'PY'
import hashlib, pathlib, sys
expected = "33f5e07c74f73c2bfb44086ae0f9e6da52acd6c104d30ffe8fd768d8253d5e82"
assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() == expected
PY
dpkg-deb -x "$RUA_RUNTIME/libglvnd0.deb" "$RUA_RUNTIME/egl"

if ! test -f "$RUA_RUNTIME/libero-$LIBERO_REV.tar.gz"; then
    test "${RUA_OFFLINE:-0}" != 1 || { printf 'Offline LIBERO archive missing.\n' >&2; exit 1; }
    curl --fail --location --retry 2 --connect-timeout 10 --max-time 180 \
        "https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/$LIBERO_REV" \
        --output "$RUA_RUNTIME/libero-$LIBERO_REV.tar.gz"
fi
"$RUA_PYTHON" - "$RUA_RUNTIME/libero-$LIBERO_REV.tar.gz" "$LIBERO_REV" "$WLA_ROOT/LIBERO" <<'PY'
import pathlib, sys, tarfile
archive, rev, checkout = sys.argv[1:]
with tarfile.open(archive) as source:
    for item in source.getmembers():
        parts = item.name.split("/")
        if parts[0] != "LIBERO-" + rev or ".." in parts or item.issym() or item.islnk():
            raise SystemExit("Unsafe or unexpected archive member: " + item.name)
    if pathlib.Path(checkout).exists():
        for name in ("libero/libero/__init__.py", "libero/libero/envs/env_wrapper.py",
                     "libero/libero/benchmark/__init__.py",
                     "libero/libero/benchmark/libero_suite_task_map.py"):
            expected = source.extractfile("LIBERO-" + rev + "/" + name).read()
            if (pathlib.Path(checkout) / name).read_bytes() != expected:
                raise SystemExit("Existing LIBERO source differs; preserve it and stop: " + name)
PY
"$RUA_PYTHON" - "$RUA_RUNTIME/libero-$LIBERO_REV.tar.gz" <<'PY'
import hashlib, pathlib, sys
assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() == (
    "05ffcf8349b2e7ef31b038451253d76ca757debbf88c3a0c1de569ca38a80b14"
), "LIBERO archive differs from the pinned evaluation source"
PY
if ! test -e "$WLA_ROOT/LIBERO"; then
    mkdir "$WLA_ROOT/LIBERO"
    tar -xzf "$RUA_RUNTIME/libero-$LIBERO_REV.tar.gz" \
        --strip-components=1 -C "$WLA_ROOT/LIBERO"
fi

if ! test -x "$RUA_VENV_PYTHON"; then
    "$RUA_UV" venv --python "$RUA_PYTHON" "$RUA_RUNTIME/venv"
fi
"$RUA_UV" pip install "${RUA_INSTALL_ARGS[@]}" --python "$RUA_VENV_PYTHON" \
    --index-url "$RUA_PIP_INDEX" \
    -r "$RUA_CODE/requirements.lock"
"$RUA_UV" pip install "${RUA_INSTALL_ARGS[@]}" --python "$RUA_VENV_PYTHON" \
    --index-url "$RUA_PIP_INDEX" \
    --no-deps -e "$WLA_ROOT/LIBERO"
# Robosuite declares opencv-python. Both distributions share cv2; install the
# matching headless binary last and explicitly verify it (no host GLX changes).
"$RUA_UV" pip install "${RUA_INSTALL_ARGS[@]}" --python "$RUA_VENV_PYTHON" \
    --index-url "$RUA_PIP_INDEX" \
    --reinstall --no-deps opencv-python-headless==4.11.0.86
"$RUA_VENV_PYTHON" -c \
    'import cv2, re; assert re.search(r"GUI:\s+NONE", cv2.getBuildInformation())'
"$RUA_UV" pip check --python "$RUA_VENV_PYTHON"
"$RUA_VENV_PYTHON" "$RUA_CODE/scripts/record_environment.py" \
    --runtime "$RUA_RUNTIME" --libero-revision "$LIBERO_REV"
