# cython: freethreading_compatible = True
# cython: language_level=3
# cython: annotation_typing=False
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

# N.B. Make sure `_register_object_by_index` is called in inheritance order,
# where the base class has to be registered before the derived class.
# Otherwise, `TypeInfo.parent_type_info` may not be properly propagated to the derived class.
include "./base.pxi"
include "./type_info.pxi"
include "./object.pxi"
_register_object_by_index(kTVMFFIObject, Object)
include "./error.pxi"
_register_object_by_index(kTVMFFIError, Error)
include "./dtype.pxi"
_register_object_by_index(kTVMFFIDataType, DataType)
include "./device.pxi"
_register_object_by_index(kTVMFFIDevice, Device)
include "./string.pxi"
_register_object_by_index(kTVMFFIStr, String)
_register_object_by_index(kTVMFFIBytes, Bytes)
include "./tensor.pxi"
_register_object_by_index(kTVMFFITensor, Tensor)
include "./function.pxi"
_register_object_by_index(kTVMFFIFunction, Function)
