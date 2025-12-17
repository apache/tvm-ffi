.. Licensed to the Apache Software Foundation (ASF) under one
.. or more contributor license agreements.  See the NOTICE file
.. distributed with this work for additional information
.. regarding copyright ownership.  The ASF licenses this file
.. to you under the Apache License, Version 2.0 (the
.. "License"); you may not use this file except in compliance
.. with the License.  You may obtain a copy of the License at
..
..   http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing,
.. software distributed under the License is distributed on an
.. "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
.. KIND, either express or implied.  See the License for the
.. specific language governing permissions and limitations
.. under the License.

CUBIN Launcher Guide
====================

This guide demonstrates how to load and launch CUDA kernels from CUBIN (CUDA Binary) modules using TVM-FFI. The CUBIN launcher enables you to execute pre-compiled or runtime-compiled CUDA kernels efficiently through the CUDA Runtime API.

Overview
--------

TVM-FFI provides utilities for loading and launching CUDA kernels from CUBIN modules. The implementation is in ``tvm/ffi/extra/cuda/cubin_launcher.h`` and provides:

- :cpp:class:`tvm::ffi::CubinModule`: RAII wrapper for loading CUBIN modules from memory
- :cpp:class:`tvm::ffi::CubinKernel`: Handle for launching CUDA kernels with specified parameters
- :c:macro:`TVM_FFI_EMBED_CUBIN`: Macro for embedding CUBIN data at compile time (legacy / object-linking approach)
- :c:macro:`TVM_FFI_LOAD_LIBRARY_FROM_BYTES`: Macro for loading CUBIN from byte arrays (modern / C++23 ``#embed`` approach)
- :c:macro:`TVM_FFI_EMBED_CUBIN_GET_KERNEL`: Macro for retrieving kernels from embedded CUBIN

The CUBIN launcher supports:

- Loading CUBIN from memory (embedded data or runtime-generated)
- Multi-GPU execution using CUDA primary contexts
- Kernel parameter management and launch configuration
- Integration with NVRTC, Triton, and other CUDA compilation tools

**Build Integration:**

TVM-FFI provides convenient tools for embedding CUBIN data at build time:

- **CMake utilities** (``cmake/Utils/EmbedCubin.cmake``): Functions for compiling CUDA to CUBIN/FATBIN and embedding it into C++ code or linking it.
- **Python utility** (``python -m tvm_ffi.utils.embed_cubin``): Command-line tool for embedding CUBIN into object files.
- **Python API** (:py:func:`tvm_ffi.cpp.load_inline`): Runtime embedding via ``embed_cubin`` parameter.

Python Usage
------------

Basic Workflow
~~~~~~~~~~~~~~

The typical workflow for launching CUBIN kernels from Python involves:

1. **Generate CUBIN**: Compile your CUDA kernel to CUBIN format
2. **Define C++ Wrapper**: Write C++ code to load and launch the kernel
3. **Load Module**: Use :py:func:`tvm_ffi.cpp.load_inline` with ``embed_cubin`` parameter
4. **Call Kernel**: Invoke the kernel function from Python

Example: NVRTC Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example using NVRTC to compile CUDA source at runtime.

**Step 1: Compile CUDA source to CUBIN using NVRTC**

.. literalinclude:: ../../examples/cubin_launcher/example_nvrtc_cubin.py
   :language: python
   :start-after: [cuda_source.begin]
   :end-before: [cuda_source.end]
   :dedent: 4

**Step 2: Define C++ wrapper with embedded CUBIN**

.. literalinclude:: ../../examples/cubin_launcher/example_nvrtc_cubin.py
   :language: python
   :start-after: [cpp_wrapper.begin]
   :end-before: [cpp_wrapper.end]
   :dedent: 4

**Key Points:**

- The ``embed_cubin`` parameter is a dictionary mapping CUBIN names to their binary data
- CUBIN names in ``embed_cubin`` must match names in :c:macro:`TVM_FFI_EMBED_CUBIN`
- Use ``cuda_sources`` parameter (instead of ``cpp_sources``) to automatically link with CUDA libraries
- The C++ wrapper handles device management, stream handling, and kernel launching

Example: Using Triton Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can compile Triton kernels to CUBIN and launch them through TVM-FFI.

**Step 1: Define and compile Triton kernel**

.. literalinclude:: ../../examples/cubin_launcher/example_triton_cubin.py
   :language: python
   :start-after: [triton_kernel.begin]
   :end-before: [triton_kernel.end]
   :dedent: 4

**Step 2: Define C++ wrapper to launch the Triton kernel**

.. literalinclude:: ../../examples/cubin_launcher/example_triton_cubin.py
   :language: python
   :start-after: [cpp_wrapper.begin]
   :end-before: [cpp_wrapper.end]
   :dedent: 4

.. note::

   Triton kernels may require extra dummy parameters in the argument list. Check the compiled kernel's signature to determine the exact parameter count needed.

C++ Usage
---------

Embedding CUBIN at Compile Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard and most convenient way to embed CUBIN/FATBIN data in C++ is using the TVM-FFI build utilities. This method keeps your source code clean by linking the binary data as an object file.

**Standard Approach: Build System Integration**

This approach uses CMake utilities to compile and link the CUBIN data. It works across all supported compilers and handles the low-level details of object file generation and symbol naming.

.. literalinclude:: ../../examples/cubin_launcher/embedded_cubin/embed_with_tvm_ffi/src/lib_embedded.cc
   :language: cpp
   :start-after: [example.begin]
   :end-before: [example.end]

**Alternative Approaches (Manual Embedding)**

TVM-FFI is designed to be flexible. If you prefer to manage the binary data explicitly in your source code or have specific compiler requirements (like C++23), you can use the ``TVM_FFI_LOAD_LIBRARY_FROM_BYTES`` macro. These approaches require you to modify your source code to include the binary data.

*Option A: Modern C++ Embedding (C++23 or Compiler Extensions)*

Using C++23 ``#embed`` (or compiler extensions like ``#embed`` in Clang/GCC) allows you to include the binary data directly.

.. code-block:: cpp

   #include <tvm/ffi/extra/cuda/cubin_launcher.h>

   // Define the image bytes using #embed
   constexpr unsigned char image[]{
   // clang >= 20 or gcc >= 14 or C++23
   #embed "kernel.fatbin"
   };

   // Register the embedded CUBIN with a name ("env" in this case)
   TVM_FFI_LOAD_LIBRARY_FROM_BYTES(env, image);

   void MyFunction() {
     // Get kernel from embedded CUBIN
     static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(env, "my_kernel");
     // ...
   }

*Option B: bin2c / Header Inclusion*

You can also use tools like ``bin2c`` to generate a header file containing the byte array and include it.

.. code-block:: cpp

   #include <tvm/ffi/extra/cuda/cubin_launcher.h>

   // Include generated header that defines 'image' array
   // e.g. const unsigned char image[] = { ... };
   #include "kernel_fatbin.h"

   TVM_FFI_LOAD_LIBRARY_FROM_BYTES(env, image);
   // ...

**Key Points:**

- Use ``static auto kernel`` to cache the kernel lookup for efficiency
- Kernel arguments must be pointers to the actual values (use ``&`` for addresses)
- :cpp:type:`tvm::ffi::dim3` supports 1D, 2D, or 3D configurations: ``dim3(x)``, ``dim3(x, y)``, ``dim3(x, y, z)``
- ``TVMFFIEnvGetStream`` retrieves the correct CUDA stream for the device
- Always check kernel launch results with :c:macro:`TVM_FFI_CHECK_CUDA_ERROR` (which checks CUDA Runtime API errors)

Loading CUBIN at Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~

You can also load CUBIN modules dynamically from memory:

.. literalinclude:: ../../examples/cubin_launcher/dynamic_cubin/src/lib_dynamic.cc
   :language: cpp
   :start-after: [example.begin]
   :end-before: [example.end]

Embedding CUBIN with CMake Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TVM-FFI provides CMake utility functions that simplify the CUBIN embedding process. This is the recommended approach for CMake-based projects.

**Using CMake Utilities:**

.. literalinclude:: ../../examples/cubin_launcher/embedded_cubin/embed_with_tvm_ffi/CMakeLists.txt
   :language: cmake
   :start-after: [cmake_example.begin]
   :end-before: [cmake_example.end]

**Available CMake Functions:**

- ``add_tvm_ffi_cubin(<target> CUDA <source>)``:
  Creates an object library that compiles CUDA source to CUBIN format.
  This is a compatibility wrapper; for CMake >= 3.27, you can use standard ``CUDA_CUBIN_COMPILATION`` property.

- ``add_tvm_ffi_fatbin(<target> CUDA <source>)``:
  Creates an object library that compiles CUDA source to FATBIN format.
  This is a compatibility wrapper; for CMake >= 3.27, you can use standard ``CUDA_FATBIN_COMPILATION`` property.

- ``tvm_ffi_embed_bin_into(<target> <name> BIN <bin_file>)``:
  Embeds a CUBIN/FATBIN file into an existing object library target.
  This works by linking the binary data into the target, allowing access via ``TVM_FFI_EMBED_CUBIN(<name>)``.

  - ``target``: The target to embed into (must be an object library or have object files).
  - ``name``: Symbol name to use (must match ``TVM_FFI_EMBED_CUBIN(name)``).
  - ``BIN``: Path to the CUBIN/FATBIN file (e.g., from ``$<TARGET_OBJECTS:...>``).

Embedding CUBIN with Python Utility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more advanced use cases or non-CMake build systems, you can use the Python command-line utility to embed CUBIN data into existing object files.

**Command-Line Usage:**

.. code-block:: bash

   # Step 1: Compile C++ source to object file
   g++ -c -fPIC -std=c++17 -I/path/to/tvm-ffi/include mycode.cc -o mycode.o

   # Step 2: Embed CUBIN into the object file
   python -m tvm_ffi.utils.embed_cubin \
       --output-obj mycode_with_cubin.o \
       --input-obj mycode.o \
       --cubin kernel.cubin \
       --name my_kernels

   # Step 3: Link into final library
   g++ -o mylib.so -shared mycode_with_cubin.o -lcudart

**Python API:**

.. code-block:: python

   from pathlib import Path
   from tvm_ffi.utils.embed_cubin import embed_cubin

   embed_cubin(
       cubin_path=Path("kernel.cubin"),
       input_obj_path=Path("mycode.o"),
       output_obj_path=Path("mycode_with_cubin.o"),
       name="my_kernels",
       verbose=True  # Optional: print detailed progress
   )

The Python utility performs these steps:

1. Creates intermediate CUBIN object file using ``ld -r -b binary``
2. Adds ``.note.GNU-stack`` section for security
3. Renames symbols to match TVM-FFI format (``__tvm_ffi__cubin_<name>``)
4. Merges with input object file using relocatable linking
5. Localizes symbols to prevent conflicts when multiple object files use the same name


Manual CUBIN Embedding
~~~~~~~~~~~~~~~~~~~~~~

For reference, here's how to manually embed CUBIN using objcopy and ld:

**Step 1: Compile CUDA kernel to CUBIN**

.. code-block:: bash

   nvcc --cubin -arch=sm_75 kernel.cu -o kernel.cubin

**Step 2: Convert CUBIN to object file**

.. code-block:: bash

   ld -r -b binary -o kernel_data.o kernel.cubin

**Step 3: Rename symbols with objcopy**

.. code-block:: bash

   objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents \
           --redefine-sym _binary_kernel_cubin_start=__tvm_ffi__cubin_my_kernels \
           --redefine-sym _binary_kernel_cubin_end=__tvm_ffi__cubin_my_kernels_end \
           kernel_data.o

**Step 4: Link with your library**

.. code-block:: bash

   g++ -o mylib.so -shared mycode.cc kernel_data.o -Wl,-z,noexecstack -lcudart

The symbol names must match the name used in :c:macro:`TVM_FFI_EMBED_CUBIN`.

**When to Use Each Approach:**

- **CMake utilities**: Best for CMake-based projects, provides cleanest integration (recommended)
- **Python utility**: Best for custom build systems, Makefile-based projects, or advanced workflows (recommended)
- **Manual objcopy**: Low-level approach, useful for understanding the process or debugging (only for customized use cases)

Advanced Topics
---------------

Multi-GPU Support
~~~~~~~~~~~~~~~~~

The CUBIN launcher automatically handles multi-GPU execution through CUDA primary contexts. Kernels will execute on the device associated with the input tensors:

.. code-block:: cpp

   void MultiGPUExample(tvm::ffi::TensorView x_gpu0, tvm::ffi::TensorView x_gpu1) {
     static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_kernels, "process");

     // Launch on GPU 0 (device determined by x_gpu0.device())
     LaunchOnDevice(kernel, x_gpu0);

     // Launch on GPU 1 (device determined by x_gpu1.device())
     LaunchOnDevice(kernel, x_gpu1);
   }

The :cpp:class:`tvm::ffi::CubinKernel` automatically uses the device context from the input tensors.

Kernel Launch Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When writing the C++ wrapper, important considerations include:

- **Grid/Block Dimensions**: Use :cpp:type:`tvm::ffi::dim3` for 1D, 2D, or 3D configurations

  - 1D: ``dim3(x)`` → ``(x, 1, 1)``
  - 2D: ``dim3(x, y)`` → ``(x, y, 1)``
  - 3D: ``dim3(x, y, z)`` → ``(x, y, z)``

- **Kernel Arguments**: Must be pointers to actual values

  - For device pointers: ``void* ptr = tensor.data_ptr(); args[] = {&ptr}``
  - For scalars: ``int n = 42; args[] = {&n}``

- **Stream Management**: Use ``TVMFFIEnvGetStream`` to get the correct CUDA stream for synchronization with DLPack tensors

- **Error Checking**: Always use :c:macro:`TVM_FFI_CHECK_CUDA_ERROR` to validate CUDA Runtime API results

Dynamic Shared Memory
~~~~~~~~~~~~~~~~~~~~~

To use dynamic shared memory, specify the size in the :cpp:func:`tvm::ffi::CubinKernel::Launch` call:

.. code-block:: cpp

   // Allocate 1KB of dynamic shared memory
   uint32_t shared_mem_bytes = 1024;
   cudaError_t result = kernel.Launch(args, grid, block, stream, shared_mem_bytes);

Integration with Different Compilers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CUBIN launcher works with various CUDA compilation tools:

- **NVCC**: Standard NVIDIA compiler, produces highly optimized CUBIN
- **NVRTC**: Runtime compilation for JIT scenarios (via :py:mod:`tvm_ffi.cpp.nvrtc`)
- **Triton**: High-level DSL that compiles to CUBIN
- **Custom compilers**: Any tool that generates valid CUDA CUBIN

Complete Examples
-----------------

For complete working examples, see the ``examples/cubin_launcher/`` directory:

- ``embedded_cubin/`` - Pre-compiled CUBIN embedded at build time
- ``dynamic_cubin/`` - CUBIN data passed dynamically at runtime
- ``example_nvrtc_cubin.py`` - NVRTC runtime compilation
- ``example_triton_cubin.py`` - Triton kernel compilation

These examples demonstrate:

- Compiling CUDA kernels to CUBIN
- Embedding CUBIN in C++ modules
- Launching kernels with proper error handling
- Testing and verification

API Reference
-------------

C++ Classes
~~~~~~~~~~~

- :cpp:class:`tvm::ffi::CubinModule`: RAII wrapper for CUBIN module lifecycle

  - :cpp:func:`tvm::ffi::CubinModule::CubinModule`: Load CUBIN from memory
  - :cpp:func:`tvm::ffi::CubinModule::GetKernel`: Get kernel by name
  - :cpp:func:`tvm::ffi::CubinModule::GetKernelWithMaxDynamicSharedMemory`: Get kernel by name with maximum dynamic shared memory set
  - :cpp:func:`tvm::ffi::CubinModule::operator[]`: Convenient kernel access

- :cpp:class:`tvm::ffi::CubinKernel`: Handle for launching kernels

  - :cpp:func:`tvm::ffi::CubinKernel::Launch`: Launch kernel with specified parameters

- :cpp:type:`tvm::ffi::dim3`: 3D dimension structure

  - ``dim3()``: Default (1, 1, 1)
  - ``dim3(unsigned int x)``: 1D
  - ``dim3(unsigned int x, unsigned int y)``: 2D
  - ``dim3(unsigned int x, unsigned int y, unsigned int z)``: 3D

C++ Macros
~~~~~~~~~~

- :c:macro:`TVM_FFI_EMBED_CUBIN`: Declare embedded CUBIN module
- :c:macro:`TVM_FFI_LOAD_LIBRARY_FROM_BYTES`: Load CUBIN from byte array
- :c:macro:`TVM_FFI_EMBED_CUBIN_GET_KERNEL`: Get kernel from embedded module
- :c:macro:`TVM_FFI_CHECK_CUDA_ERROR`: Check CUDA Runtime API result

Python Functions
~~~~~~~~~~~~~~~~

- :py:func:`tvm_ffi.cpp.nvrtc.nvrtc_compile`: Compile CUDA source to CUBIN
- :py:func:`tvm_ffi.cpp.load_inline`: Load inline module with embedded CUBIN

Python Utilities
~~~~~~~~~~~~~~~~

- ``python -m tvm_ffi.utils.embed_cubin``: Command-line utility to embed CUBIN into object files

  - ``--output-obj PATH``: Output combined object file path
  - ``--input-obj PATH``: Input object file containing C++ code with ``TVM_FFI_EMBED_CUBIN``
  - ``--cubin PATH``: Input CUBIN file to embed
  - ``--name NAME``: Symbol name matching ``TVM_FFI_EMBED_CUBIN(name)`` macro
  - ``--verbose``: Print detailed command output (optional)

- :py:func:`tvm_ffi.utils.embed_cubin.embed_cubin`: Python API for embedding CUBIN

  - ``cubin_path``: Path to input CUBIN file
  - ``input_obj_path``: Path to existing object file
  - ``output_obj_path``: Path to output combined object file
  - ``name``: Symbol name for the embedded CUBIN
  - ``verbose``: Enable detailed output (default: False)

CMake Functions
~~~~~~~~~~~~~~~

- ``add_tvm_ffi_cubin(<target> CUDA <source>)``: Compile CUDA source to CUBIN
- ``add_tvm_ffi_fatbin(<target> CUDA <source>)``: Compile CUDA source to FATBIN
- ``tvm_ffi_embed_bin_into(<target> <name> BIN <bin_file>)``: Embed CUBIN/FATBIN into object target
