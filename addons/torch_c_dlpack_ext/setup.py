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
# specific language governing permissions and limitations.
# Base logic to load library for extension package
"""Setup for torch c dlpack extension."""

from pathlib import Path

import torch
from setuptools import setup
from torch.utils import cpp_extension

include_dirs = []
root_dir = Path(__file__).parent.parent.parent

dlpack_include = root_dir / "3rdparty" / "dlpack" / "include"
if dlpack_include.is_dir():
    include_dirs.append(str(dlpack_include))

extra_compile_args = ["-O3", "-std=c++17"]
define_macros = []
if torch.cuda.is_available():
    include_dirs += cpp_extension.include_paths("cuda")
    define_macros += [("BUILD_WITH_CUDA", None)]

setup(
    name="torch_c_dlpack_ext",
    ext_modules=[
        cpp_extension.CppExtension(
            "torch_c_dlpack_ext",
            ["torch_c_dlpack_ext.cc"],
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
