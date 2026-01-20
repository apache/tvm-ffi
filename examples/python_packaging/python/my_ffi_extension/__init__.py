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
"""Package my_ffi_extension."""

# tvm-ffi-stubgen(begin): export/_ffi_api
# fmt: off
# isort: off
from ._ffi_api import *  # noqa: F403
from ._ffi_api import __all__ as _ffi_api__all__
if "__all__" not in globals():
    __all__ = []
__all__.extend(_ffi_api__all__)
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)
