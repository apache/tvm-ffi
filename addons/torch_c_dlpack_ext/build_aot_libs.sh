#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
set -eux

arch=$1
python_version=$2

tvm_ffi="$PWD"
torch_c_dlpack_ext="$tvm_ffi"/addons/torch_c_dlpack_ext


function get_torch_url() {
    local version="$1"
    if [[ "$version" == "2.4" ]]; then
        echo "https://download.pytorch.org/whl/cu124"
    elif [[ "$version" == "2.5" ]]; then
        echo "https://download.pytorch.org/whl/cu124"
    elif [[ "$version" == "2.6" ]]; then
        echo "https://download.pytorch.org/whl/cu126"
    elif [[ "$version" == "2.7" ]]; then
        echo "https://download.pytorch.org/whl/cu128"
    elif [[ "$version" == "2.8" ]]; then
        echo "https://download.pytorch.org/whl/cu129"
    elif [[ "$version" == "2.9" ]]; then
        echo "https://download.pytorch.org/whl/cu129"
    else
        echo "Unknown or unsupported torch version: $version" >&2
        return 1
    fi
}


function check_availability() {
    local torch_version="$1"
    if [[ "$torch_version" == "2.4" ]]; then
        return $([[ "$arch" == "aarch64" || "$python_version" == "cp313" || "$python_version" == "cp314" ]] && echo 1 || echo 0)
    elif [[ "$torch_version" == "2.5" ]]; then
        return $([[ "$arch" == "aarch64" || "$python_version" == "cp314" ]] && echo 1 || echo 0)
    elif [[ "$torch_version" == "2.6" ]]; then
        return $([[ "$arch" == "aarch64" || "$python_version" == "cp314" ]] && echo 1 || echo 0)
    elif [[ "$torch_version" == "2.7" ]]; then
        return $([[ "$python_version" == "cp314" ]] && echo 1 || echo 0)
    elif [[ "$torch_version" == "2.8" ]]; then
        return $([[ "$python_version" == "cp314" ]] && echo 1 || echo 0)
    elif [[ "$torch_version" == "2.9" ]]; then
        return $([[ "$python_version" == "cp39" ]] && echo 1 || echo 0)
    else
        echo "Unknown or unsupported torch version: $torch_version" >&2
        return 1
    fi
}


function build_libs() {
    local torch_version=$1
    if check_availability "$torch_version"; then
        mkdir "$tvm_ffi"/.venv -p
        uv venv "$tvm_ffi"/.venv/torch"$torch_version" --python $python_version
        source "$tvm_ffi"/.venv/torch"$torch_version"/bin/activate
        uv pip install setuptools ninja
        uv pip install torch=="$torch_version" --index-url $(get_torch_url "$torch_version")
        uv pip install -v .
        mkdir "$tvm_ffi"/lib -p
        python -m tvm_ffi.utils._build_optional_torch_c_dlpack --output-dir "$tvm_ffi"/lib
        python -m tvm_ffi.utils._build_optional_torch_c_dlpack --output-dir "$tvm_ffi"/lib --build-with-cuda
        ls "$tvm_ffi"/lib
        deactivate
        rm -rf "$tvm_ffi"/.venv/torch"$torch_version"
    fi
}

build_libs "2.4"
build_libs "2.5"
build_libs "2.6"
build_libs "2.7"
build_libs "2.8"
build_libs "2.9"