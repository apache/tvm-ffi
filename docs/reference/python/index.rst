..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Python API
==========

.. automodule:: tvm_ffi
  :no-members:

.. currentmodule:: tvm_ffi

.. contents:: Table of Contents
   :local:
   :depth: 2


Object
------
.. autosummary::
  :toctree: generated/

  Object


Tensor
~~~~~~
.. autosummary::
  :toctree: generated/

  Tensor
  from_dlpack
  Shape
  dtype
  Device
  DLDeviceType
  device


Function
~~~~~~~~
.. autosummary::
  :toctree: generated/

  Function


Module
~~~~~~
.. autosummary::
  :toctree: generated/

  Module
  system_lib
  load_module


Containers
~~~~~~~~~~
.. autosummary::
  :toctree: generated/

  Array
  Map


Global Registry
---------------
.. autosummary::
  :toctree: generated/

  register_error
  register_object
  register_global_func
  get_global_func
  get_global_func_metadata
  init_ffi_api
  remove_global_func


Stream Context
--------------
.. autosummary::
  :toctree: generated/

  StreamContext
  use_torch_stream
  use_raw_stream
  get_raw_stream


C++ Extension
--------------

C++ integration helpers for building and loading inline modules.

.. autosummary::
  :toctree: cpp/generated/

  cpp.load_inline
  cpp.build_inline
  cpp.load
  cpp.build

Misc
----
.. autosummary::
  :toctree: generated/

  serialization.from_json_graph_str
  serialization.to_json_graph_str
  access_path.AccessKind
  access_path.AccessPath
  access_path.AccessStep
  convert
  ObjectConvertible

.. (Experimental) Dataclasses
.. --------------------------

.. .. autosummary::
..   :toctree: generated/

..   dataclasses.c_class
..   dataclasses.field
