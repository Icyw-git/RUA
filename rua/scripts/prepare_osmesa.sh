#!/usr/bin/env bash
# Private renderer dependencies from Ubuntu's configured, signed package index.
# No apt install, system modification, or change to the model runtime.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh"
RUA_OSMESA="$RUA_RUNTIME/osmesa-24.0.5"
mkdir -p "$RUA_OSMESA/packages"
cd "$RUA_OSMESA/packages"
if test ! -f libosmesa6_24.0.5-1ubuntu1_amd64.deb; then
  apt-get download libosmesa6=24.0.5-1ubuntu1
fi
if test ! -f libglapi-mesa_24.0.5-1ubuntu1_amd64.deb; then
  apt-get download libglapi-mesa=24.0.5-1ubuntu1
fi
if test ! -f libdrm2_2.4.120-2build1_amd64.deb; then
  apt-get download libdrm2=2.4.120-2build1
fi
if test ! -f libdrm-common_2.4.120-2build1_all.deb; then
  apt-get download libdrm-common=2.4.120-2build1
fi
test "$(sha256sum libosmesa6_24.0.5-1ubuntu1_amd64.deb | cut -d ' ' -f 1)" = f27463d400dba6d8862060fded4d7ec5be73a837344844cbeb082ef32d269df5
test "$(sha256sum libglapi-mesa_24.0.5-1ubuntu1_amd64.deb | cut -d ' ' -f 1)" = 28191867c3a68d59aca31755789328d28960223127b05db6a46b7e8f5646a495
test "$(sha256sum libdrm2_2.4.120-2build1_amd64.deb | cut -d ' ' -f 1)" = f5fb4e7ce17921cc466fb7911abf91495ffb181b36772f68e2e82cb621703112
test "$(sha256sum libdrm-common_2.4.120-2build1_all.deb | cut -d ' ' -f 1)" = 84d60f8726a47b57c2ba9bd4105d838ff8554b7309035a61543e2bee4ed7914a
dpkg-deb -x libosmesa6_24.0.5-1ubuntu1_amd64.deb "$RUA_OSMESA"
dpkg-deb -x libglapi-mesa_24.0.5-1ubuntu1_amd64.deb "$RUA_OSMESA"
dpkg-deb -x libdrm2_2.4.120-2build1_amd64.deb "$RUA_OSMESA"
dpkg-deb -x libdrm-common_2.4.120-2build1_all.deb "$RUA_OSMESA"
export LD_LIBRARY_PATH="$RUA_OSMESA/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$RUA_OSMESA/usr/lib/x86_64-linux-gnu/libOSMesa.so.8"
