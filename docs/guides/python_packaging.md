<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->
# Python Binding and Packaging

This guide explains how to leverage tvm-ffi to expose C++ functions into Python and package them into a wheel.
At a high level, packaging with tvm-ffi offers several benefits:

- **Ship one wheel** that can be used across Python versions, including free-threaded Python.
- **Multi-language access** to functions from Python, C++, Rust and other languages that support the ABI.
- **ML Systems Interop** with ML frameworks, DSLs, and libraries while maintaining minimal dependency.

## Directly using Exported Library

If you just need to expose a simple set of functions,
you can declare an exported symbol in C++:

```c++
// Compiles to mylib.so
#include <tvm/ffi/function.h>

int add_one(int x) {
  return x + 1;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, add_one)
```

You then load the exported function in your Python project via {py:func}`tvm_ffi.load_module`.

```python
# In your __init__.py
import tvm_ffi

_LIB = tvm_ffi.load_module("/path/to/mlib.so")

def add_one(x):
    """Expose mylib.add_one"""
    return _LIB.add_one(x)
```

This approach is like using {py:mod}`ctypes` to load and run DLLs, except we have more powerful features:

- We can pass in `torch.Tensor` (or any other DLPack-compatible arrays).
- We can pass in a richer set of data structures such as strings, tuples, and dicts.
- {py:class}`tvm_ffi.Function` enables natural callbacks to Python lambdas or other languages.
- Exceptions are propagated naturally across language boundaries.

## Pybind11 and Nanobind style Usage

For advanced use cases where users may wish to register global functions or custom object types,
we also provide a pybind11/nanobind style API to register functions and custom objects.

```c++
#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/registry.h>

namespace my_ffi_extension {

namespace ffi = tvm::ffi;

/*!
 * \brief Example of a custom object that is exposed to the FFI library
 */
class IntPairObj : public ffi::Object {
 public:
  int64_t a;
  int64_t b;

  IntPairObj() = default;
  IntPairObj(int64_t a, int64_t b) : a(a), b(b) {}

  int64_t GetFirst() const { return this->a; }

  // Required: declare type information
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("my_ffi_extension.IntPair", IntPairObj, ffi::Object);
};

/*!
 * \brief Defines an explicit reference to IntPairObj
 *
 * A reference wrapper serves as a reference-counted pointer to the object.
 * You can use obj->field to access the fields of the object.
 */
class IntPair : public tvm::ffi::ObjectRef {
 public:
  // Constructor
  explicit IntPair(int64_t a, int64_t b) {
    data_ = tvm::ffi::make_object<IntPairObj>(a, b);
  }

  // Required: define object reference methods
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntPair, tvm::ffi::ObjectRef, IntPairObj);
};

void RaiseError(ffi::String msg) { TVM_FFI_THROW(RuntimeError) << msg; }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
    .def("my_ffi_extension.raise_error", RaiseError);
  // register object definition
  refl::ObjectDef<IntPairObj>()
      .def(refl::init<int64_t, int64_t>())
       // Example static method that returns the second element of the pair
      .def_static("static_get_second", [](IntPair pair) -> int64_t { return pair->b; })
      // Example to bind an instance method
      .def("get_first", &IntPairObj::GetFirst)
      .def_ro("a", &IntPairObj::a)
      .def_ro("b", &IntPairObj::b);
}
}  // namespace my_ffi_extension
```

Then these functions and objects can be accessed from Python as long as the library is loaded.
You can use {py:func}`tvm_ffi.load_module` or simply use {py:class}`ctypes.CDLL`. Then you can access
the function through {py:func}`tvm_ffi.get_global_func` or {py:func}`tvm_ffi.init_ffi_api`.
We also allow direct exposure of object via {py:func}`tvm_ffi.register_object`.

```python
# __init__.py
import tvm_ffi

def raise_error(msg: str):
    """Wrap raise error function."""
    # Usually we reorganize these functions into a _ffi_api.py and load once
    func = tvm_ffi.get_global_func("my_ffi_extension.raise_error")
    func(msg)


@tvm_ffi.register_object("my_ffi_extension.IntPair")
class IntPair(tvm_ffi.Object):
    """IntPair object."""

    def __init__(self, a: int, b: int) -> None:
        """Construct the object."""
        # __ffi_init__ call into the refl::init<> registered
        # in the static initialization block of the extension library
        self.__ffi_init__(a, b)


def run_example():
    pair = IntPair(1, 2)
    # prints 1
    print(pair.get_first())
    # prints 2
    print(IntPair.static_get_second(pair))
    # Raises a RuntimeError("error happens")
    raise_error("error happens")
```

### Relations to Existing Solutions

Most current binding systems focus on creating one-to-one bindings
that take a source language and bind to an existing target language runtime and ABI.
We deliberately take a more decoupled approach here:

- Build stable, minimal ABI convention that is agnostic to the target language.
- Create bindings to connect the source and target language to the ABI.

The focus of this project is the ABI itself which we believe can help the overall ecosystem.
We also anticipate there are possibilities for existing binding generators to also target the tvm-ffi ABI.

**Design philosophy**. We have the following design philosophies focusing on ML systems.

- FFI and cross-language interop should be first-class citizens in ML systems rather than an add-on.
- Enable multi-environment support in both source and target languages.
- The same ABI should be minimal and targetable by DSL compilers.

Of course, there is always a tradeoff. It is by design impossible to support arbitrary advanced language features
in the target language, as different programming languages have their own design considerations.
We do believe it is possible to build a universal, effective, and minimal ABI for machine learning
system use cases. Based on the above design philosophies, we focus our cross-language
interaction interface through the FFI ABI for machine learning systems.

So if you are building projects related to machine learning compilers, runtimes,
libraries, frameworks, DSLs, or generally scientific computing, we encourage you
to try it out. The extension mechanism can likely support features in other domains as well
and we welcome you to try it out as well.

### Mix with Existing Solutions

Because the global registry mechanism only relies on the code being linked,
you can also partially use tvm-ffi-based registration together with pybind11/nanobind in your project.
Just add the related code, link to `libtvm_ffi` and make sure you `import tvm_ffi` before importing
your module to ensure related symbols are available.
This approach may help to quickly leverage some of the cross-language features we have.
It also provides more powerful interaction with the host Python language, but of course the tradeoff
is that the final library will now also depend on the Python ABI.

## Example Project Walk Through

To get hands-on experience with the packaging flow,
you can try out an example project in our folder.
First, obtain a copy of the tvm-ffi source code.

```bash
git clone https://github.com/apache/tvm-ffi --recursive
cd tvm-ffi
```

The examples are now in the examples folder. You can quickly build
and install the example using the following commands.

```bash
cd examples/packaging
pip install -v .
```

Then you can run examples that leverage the built wheel package.

```bash
python run_example.py add_one
```

## Setup pyproject.toml

A typical tvm-ffi-based project has the following structure:

```text
├── CMakeLists.txt          # CMake build configuration
├── pyproject.toml          # Python packaging configuration
├── src/
│   └── extension.cc        # C++ source code
├── python/
│   └── my_ffi_extension/
│       ├── __init__.py     # Python package initialization
│       ├── base.py         # Library loading logic
│       └── _ffi_api.py     # FFI API registration
└── README.md               # Project documentation
```

The `pyproject.toml` file configures the build system and project metadata.

```toml
[project]
name = "my-ffi-extension"
version = "0.1.0"
# ... more project metadata omitted ...

[build-system]
requires = ["scikit-build-core>=0.10.0", "apache-tvm-ffi"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
# ABI-agnostic wheel
wheel.py-api = "py3"
# ... more build configuration omitted ...
```

We use scikit-build-core for building the wheel. Make sure you add tvm-ffi as a build-system requirement.
Importantly, we should set `wheel.py-api` to `py3` to indicate it is ABI-generic.

### Setup CMakeLists.txt

The CMakeLists.txt handles the build and linking of the project.
There are two ways you can build with tvm-ffi:

- Link the pre-built `libtvm_ffi` shipped from the pip package
- Build tvm-ffi from source

For common cases, using the pre-built library and linking tvm_ffi_shared is sufficient.
To build with the pre-built library, you can do:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_ffi_extension)

find_package(Python COMPONENTS Interpreter REQUIRED)
# find the prebuilt package
find_package(tvm_ffi CONFIG REQUIRED)

# ... more cmake configuration omitted ...

# linking the library
target_link_libraries(my_ffi_extension tvm_ffi_shared)
```

There are cases where one may want to cross-compile or bundle part of tvm_ffi objects directly
into the project. In such cases, you should build from source.

```cmake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --sourcedir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_ROOT)
# add the shipped source code as a cmake subdirectory
add_subdirectory(${tvm_ffi_ROOT} tvm_ffi)

# ... more cmake configuration omitted ...

# linking the library
target_link_libraries(my_ffi_extension tvm_ffi_shared)
```

Note that it is always safe to build from source, and the extra cost of building tvm-ffi is small
because tvm-ffi is a lightweight library. If you are in doubt,
you can always choose to build tvm-ffi from source.
In Python or other cases when we dynamically load libtvm_ffi shipped with the dedicated pip package,
you do not need to ship libtvm_ffi.so in your package even if you build tvm-ffi from source.
The built objects are only used to supply the linking information.

### Exposing C++ Functions

The C++ implementation is defined in `src/extension.cc`.
There are two ways one can expose a function in C++ to the FFI library.
First, `TVM_FFI_DLL_EXPORT_TYPED_FUNC` can be used to expose the function directly as a C symbol that follows the tvm-ffi ABI,
which can later be accessed via `tvm_ffi.load_module`.

Here's a basic example of the function implementation:

```c++
void AddOne(ffi::TensorView x, ffi::TensorView y) {
  // ... implementation omitted ...
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, my_ffi_extension::AddOne);
```

We can also register a function into the global function table with a given name:

```c++
void RaiseError(ffi::String msg) {
  TVM_FFI_THROW(RuntimeError) << msg;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
    .def("my_ffi_extension.raise_error", RaiseError);
}
```

Make sure to have a unique name across all registered functions when registering a global function.
Always prefix with a package namespace name to avoid name collisions.
The function can then be found via `tvm_ffi.get_global_func(name)`
and is expected to stay throughout the lifetime of the program.

We recommend using `TVM_FFI_DLL_EXPORT_TYPED_FUNC` for functions that are supposed to be dynamically
loaded (such as JIT scenarios) so they won't be exposed to the global function table.

### Library Loading in Python

The base module handles loading the compiled extension:

```python
import tvm_ffi
import os
import sys

def _load_lib():
    file_dir = os.path.dirname(os.path.realpath(__file__))

    # Platform-specific library names
    if sys.platform.startswith("win32"):
        lib_name = "my_ffi_extension.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = "my_ffi_extension.dylib"
    else:
        lib_name = "my_ffi_extension.so"

    lib_path = os.path.join(file_dir, lib_name)
    return tvm_ffi.load_module(lib_path)

_LIB = _load_lib()
```

Effectively, it leverages the `tvm_ffi.load_module` call to load the library
extension DLL shipped along with the package. The `_ffi_api.py` contains a function
call to `tvm_ffi.init_ffi_api` that registers all global functions prefixed
with `my_ffi_extension` into the module.

```python
# _ffi_api.py
import tvm_ffi
from .base import _LIB

# Register all global functions prefixed with 'my_ffi_extension.'
# This makes functions registered via TVM_FFI_STATIC_INIT_BLOCK available
tvm_ffi.init_ffi_api("my_ffi_extension", __name__)
```

Then we can redirect the calls to the related functions.

```python
from .base import _LIB
from . import _ffi_api

def add_one(x, y):
    # ... docstring omitted ...
    return _LIB.add_one(x, y)

def raise_error(msg):
    # ... docstring omitted ...
    return _ffi_api.raise_error(msg)
```

### Build and Use the Package

First, build the wheel:

```bash
pip wheel -v -w dist .
```

Then install the built wheel:

```bash
pip install dist/*.whl
```

Then you can try it out:

```python
import torch
import my_ffi_extension

# Create input and output tensors
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.empty_like(x)

# Call the function
my_ffi_extension.add_one(x, y)
print(y)  # Output: tensor([2., 3., 4., 5., 6.])
```

You can also run the following command to see how errors are raised and propagated
across language boundaries:

```bash
python run_example.py raise_error
```

When possible, tvm-ffi will try to preserve backtraces across language boundaries. You will see outputs like:

```text
File "src/extension.cc", line 45, in void my_ffi_extension::RaiseError(tvm::ffi::String)
```

## Wheel Auditing

When using `auditwheel`, exclude `libtvm_ffi` as it will be shipped with the `tvm_ffi` package.

```bash
auditwheel repair --exclude libtvm_ffi.so dist/*.whl
```

As long as you import `tvm_ffi` first before loading the library, the symbols will be available.
