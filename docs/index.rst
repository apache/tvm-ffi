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

Apache TVM FFI Documentation
============================

Welcome to the documentation for TVM FFI. You can get started by reading the get started section,
or reading through the guides and concepts sections.


Installation
------------

To install TVM-FFI via pip or uv, run:

.. code-block:: bash

   pip install apache-tvm-ffi
   pip install torch-c-dlpack-ext  # compatibility package for torch <= 2.9


Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/quickstart.rst
   get_started/stable_c_abi.rst

.. toctree::
   :maxdepth: 1
   :caption: Guides

   guides/kernel_library_guide.rst
   guides/compiler_integration.md
   guides/cubin_launcher.rst
   guides/python_lang_guide.md
   guides/cpp_lang_guide.md
   guides/rust_lang_guide.md

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   concepts/abi_overview.md

.. toctree::
   :maxdepth: 1
   :caption: Packaging

   packaging/python_packaging.rst
   packaging/cpp_packaging.md

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/python/index.rst
   reference/cpp/index.rst
   reference/rust/index.rst

.. toctree::
   :maxdepth: 1
   :caption: Developer Manual

   dev/build_from_source.md
