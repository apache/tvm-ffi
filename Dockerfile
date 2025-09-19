# syntax=docker/dockerfile:1

FROM ubuntu:22.04

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
        python3-distutils \
        python3-pip \
        python3-setuptools \
        python3-venv \
        python3-wheel \
        unzip \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and provide a python convenience symlink
RUN python3 -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir numpy pytest \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# Install documentation tooling that depends on pip packages
COPY docs/requirements.txt /tmp/docs-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/docs-requirements.txt \
    && rm -f /tmp/docs-requirements.txt

# Provide a working directory for the project
WORKDIR /workspace

# Optionally clone a repository during build with --build-arg REPO_URL=...
ARG REPO_URL
RUN if [ -n "${REPO_URL}" ]; then git clone "${REPO_URL}" repo; fi

CMD ["/bin/bash"]
