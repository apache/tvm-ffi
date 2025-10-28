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

# shellcheck disable=SC1090
set -eux
set -o pipefail

uv venv /torch"$1" --python "$2" && source /torch"$1"/bin/activate
uv pip install torch=="$1" --index-url https://download.pytorch.org/whl/"$3"
uv pip install setuptools
uv pip install -v .
python python/tvm_ffi/utils/_build_optional_c_dlpack.py --build_dir ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext
mv ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext/libtorch_c_dlpack_addon.so ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext/libtorch_c_dlpack_addon_torch"$1".so
rm /torch"$1" -rf
