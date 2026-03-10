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

# Install static zlib and zstd for static linking
yum install -y zlib-static libzstd-devel

# Create environment with LLVM
/usr/local/bin/micromamba create -p "${PREFIX}" -c conda-forge \
  "llvmdev=${LLVM_VERSION}" "clangdev=${LLVM_VERSION}" "compiler-rt=${LLVM_VERSION}" \
  -y
