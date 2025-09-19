# syntax=docker/dockerfile:1

# Using NVIDIA CUDA base image https://hub.docker.com/r/nvidia/cuda
# Based on your host CUDA driver version, you may need to select an older CUDA base image.
# See https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 

ARG DEBIAN_FRONTEND=noninteractive

# Install prerequisites for external repositories
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        wget \
        lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Add Kitware APT repository for CMake >= 3.24
RUN wget -O /tmp/kitware-archive.sh https://apt.kitware.com/kitware-archive.sh \
    && bash /tmp/kitware-archive.sh \
    && rm -f /tmp/kitware-archive.sh

# Install build essentials, Git, and Python >= 3.9
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        clang-format \
        clang-tidy \
        cmake \
        doxygen \
        git \
        graphviz \
        ninja-build \
        pandoc \
        pkg-config \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-venv \
        python3-wheel \
        unzip \
        zip \
    && rm -rf /var/lib/apt/lists/* 

# Provide a working directory for the project
WORKDIR /workspace

# Optionally clone a repository during build with --build-arg REPO_URL=...
ARG REPO_URL
RUN if [ -n "${REPO_URL}" ]; then git clone "${REPO_URL}" repo; fi

CMD ["/bin/bash"]
