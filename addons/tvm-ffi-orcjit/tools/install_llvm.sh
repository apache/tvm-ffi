#!/bin/bash
# Install LLVM from conda-forge using micromamba.
# Usage: bash tools/install_llvm.sh [version]
#   version defaults to LLVM_VERSION env var, then 22.1.0
set -ex

LLVM_VERSION="${LLVM_VERSION:-${1:-22.1.0}}"
PREFIX="${LLVM_PREFIX:-/opt/llvm}"

# Detect micromamba platform
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)   PLATFORM="linux-64" ;;
  Linux-aarch64)  PLATFORM="linux-aarch64" ;;
  Darwin-x86_64)  PLATFORM="osx-64" ;;
  Darwin-arm64)   PLATFORM="osx-arm64" ;;
  *)              echo "Unsupported: $(uname -s)-$(uname -m)"; exit 1 ;;
esac

# Install micromamba
curl -Ls "https://micro.mamba.pm/api/micromamba/${PLATFORM}/latest" \
  | tar -xvj -C /usr/local bin/micromamba

# Install static zlib; zstd is built from source below (no static package available)
yum install -y zlib-static

# Create environment with LLVM
/usr/local/bin/micromamba create -p "${PREFIX}" -c conda-forge \
  "llvmdev=${LLVM_VERSION}" "clangdev=${LLVM_VERSION}" "compiler-rt=${LLVM_VERSION}" \
  -y

# Build libzstd.a from source (no static package on AlmaLinux 8 or 9)
ZSTD_VERSION="1.5.7"
curl -sL "https://github.com/facebook/zstd/releases/download/v${ZSTD_VERSION}/zstd-${ZSTD_VERSION}.tar.gz" \
  | tar -xz
cmake -S "zstd-${ZSTD_VERSION}/build/cmake" -B _zstd_build \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DZSTD_BUILD_SHARED=OFF -DZSTD_BUILD_STATIC=ON \
  -DZSTD_BUILD_PROGRAMS=OFF
cmake --build _zstd_build --target install -j"$(nproc)"
rm -rf "zstd-${ZSTD_VERSION}" _zstd_build
