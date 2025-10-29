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

# shellcheck disable=SC1091
set -eux
set -o pipefail

pip install uv

# if [[ "$1" != "3.13" && "$1" != "3.14" ]]; then
#   bash ./addons/torch_c_dlpack_ext/build_so.sh 2.4 "$1" cu124
# fi
if [[ "$1" != "3.14" ]]; then
  # bash ./addons/torch_c_dlpack_ext/build_so.sh 2.5 "$1" cu124
  # bash ./addons/torch_c_dlpack_ext/build_so.sh 2.6 "$1" cu126
  # bash ./addons/torch_c_dlpack_ext/build_so.sh 2.7 "$1" cu126
  bash ./addons/torch_c_dlpack_ext/build_so.sh 2.8 "$1" cu128
fi
bash ./addons/torch_c_dlpack_ext/build_so.sh 2.9 "$1" cu128

uv venv /base --python "$1" && source /base/bin/activate
uv pip install setuptools auditwheel
cd ./addons/torch_c_dlpack_ext
python setup.py sdist bdist_wheel --python-tag py"${1//\./}"
auditwheel repair --exclude libtorch.so --exclude libtorch_cpu.so --exclude libc10.so --exclude libtorch_python.so --plat manylinux_2_28_x86_64 dist/*.whl -w wheelhouse
