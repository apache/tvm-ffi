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
# Python Guide

This guide introduces the `tvm_ffi` Python package.
At a high level, the `tvm_ffi` Python package provides first-class Python support for

- Pythonic classes to represent values in TVM FFI Any ABI.
- Mechanisms to call into TVM FFI ABI compatible functions.
- Conversion between Python values and `tvm_ffi` values.

In this guide, we will run examples that make use of pre-registered testing functions in `tvm_ffi`.
If so, we will also briefly copy snippets that show the corresponding C++ behavior.

## Load and Run Module

The most common use case of TVM FFI is to load a runnable module and run the corresponding function.
You can follow the [quickstart guide](../get_started/quickstart.rst) for details on building the
library `build/add_one_cpu.so`. Let's walk through the load and run example again for NumPy

```python
import tvm_ffi
import numpy as np

# Load the compiled module
mod = tvm_ffi.load_module("build/add_one_cpu.so")

# Create input and output arrays
x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.empty_like(x)

# Call the function
mod.add_one_cpu(x, y)
```

In this case, {py:func}`tvm_ffi.load_module` will return a {py:class}`tvm_ffi.Module` class that contains
the exported functions. You can access the functions by their names.

## Tensor

`tvm_ffi` provides a managed DLPack-compatible Tensor.

```python
import numpy as np
import tvm_ffi

# Demonstrate DLPack conversion between NumPy and TVM FFI
np_data = np.array([1, 2, 3, 4], dtype=np.float32)
tvm_array = tvm_ffi.from_dlpack(np_data)
# Convert back to NumPy
np_result = np.from_dlpack(tvm_array)
```

In most cases, however, you do not have to explicitly create Tensors.
The Python interface can take in `torch.Tensor` and `numpy.ndarray` objects
and automatically convert them to {py:class}`tvm_ffi.Tensor`.

## Functions and Callbacks

{py:class}`tvm_ffi.Function` provides the Python interface for `ffi::Function` in the C++.
You can retrieve globally registered functions via {py:func}`tvm_ffi.get_global_func`.

```python
import tvm_ffi

# testing.echo is defined and registered in C++
# [](ffi::Any x) { return x; }
fecho = tvm_ffi.get_global_func("testing.echo")
assert fecho(1) == 1
```

You can pass a Python function as an argument to another FFI function as callbacks.
Under the hood, {py:func}`tvm_ffi.convert` is called to convert the Python function into a
{py:class}`tvm_ffi.Function`.

```python
import tvm_ffi

# testing.apply is registered in C++
# [](ffi::Function f, ffi::Any val) { return f(x); }
fapply = tvm_ffi.get_global_func("testing.apply")
# invoke fapply with lambda callback as f
assert fapply(lambda x: x + 1, 1) == 2
```

This is a very powerful pattern that allows us to inject Python callbacks into the C++ code.
You can also register a Python callback as a global function.

```python
import tvm_ffi

@tvm_ffi.register_global_func("example.add_one")
def add_one(a):
    return a + 1

assert tvm_ffi.get_global_func("example.add_one")(1) == 2
```

## Container Types

When an FFI function takes arguments from lists/tuples, they will be converted into {py:class}`tvm_ffi.Array`.

```python
import tvm_ffi

# Lists become Arrays
arr = tvm_ffi.convert([1, 2, 3, 4])
assert isinstance(arr, tvm_ffi.Array)
assert len(arr) == 4
assert arr[0] == 1
```

Dictionaries will be converted to {py:class}`tvm_ffi.Map`

```python
import tvm_ffi

map_obj = tvm_ffi.convert({"a": 1, "b": 2})
assert isinstance(map_obj, tvm_ffi.Map)
assert len(map_obj) == 2
assert map_obj["a"] == 1
assert map_obj["b"] == 2
```

When container values are returned from FFI functions, they are also stored in these
types respectively.

## Inline Module

You can also load a _inline module_ where the C++/CUDA code is directly embedded in the Python script and then compiled
on the fly. For example, we can define a simple kernel that adds one to each element of an array as follows:

```python
import torch
from tvm_ffi import Module
import tvm_ffi.cpp

# define the cpp source code
cpp_source = '''
     void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
       // implementation of a library function
       TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
       DLDataType f32_dtype{kDLFloat, 32, 1};
       TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
       TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
       TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
       TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
       for (int i = 0; i < x.size(0); ++i) {
         static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
       }
     }
'''

# compile the cpp source code and load the module
mod: Module = tvm_ffi.cpp.load_inline(
    name='hello', cpp_sources=cpp_source, functions='add_one_cpu'
)

# use the function from the loaded module to perform
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.empty_like(x)
mod.add_one_cpu(x, y)
torch.testing.assert_close(x + 1, y)
```

The above code defines a C++ function `add_one_cpu` in Python script, compiles it on the fly and then loads the compiled
{py:class}`tvm_ffi.Module` object via {py:func}`tvm_ffi.cpp.load_inline`. You can then call the function `add_one_cpu`
from the module as usual.

We can also use {py:func}`tvm_ffi.cpp.build_inline` to build the inline module without loading it. The built shared library is returned
and can be loaded via {py:func}`tvm_ffi.load_module`.

## Error Handling

An FFI function may raise an error. In such cases, the Python package will automatically
translate the error to the corresponding error kind in Python

```python
import tvm_ffi

# defined in C++
# [](String kind, String msg) { throw Error(kind, msg, backtrace); }
test_raise_error = tvm_ffi.get_global_func("testing.test_raise_error")

test_raise_error("ValueError", "message")
```

The above code shows an example where an error is raised in C++, resulting in the following error trace

```text
Traceback (most recent call last):
File "example.py", line 7, in <module>
  test_raise_error("ValueError", "message")
  ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
File "python/tvm_ffi/cython/function.pxi", line 325, in core.Function.__call__
  raise move_from_last_error().py_error()
  ^^^
File "src/ffi/extra/testing.cc", line 60, in void tvm::ffi::TestRaiseError(tvm::ffi::String, tvm::ffi::String)
  throw ffi::Error(kind, msg, TVMFFITraceback(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0));

```

We register common error kinds. You can also register extra error dispatch via the {py:func}`tvm_ffi.register_error` function.

## Advanced: Register Your Own Object

For advanced use cases, you may want to register your own objects. This can be achieved through the
reflection registry in the TVM-FFI API. First, let's review the C++ side of the code. For this
example, you do not need to change the C++ side as this code is pre-shipped with the testing module of the `tvm_ffi` package.

```cpp
#include <tvm/ffi/reflection/registry.h>

// Step 1: Define the object class (stores the actual data)
class TestIntPairObj : public tvm::ffi::Object {
public:
  int64_t a;
  int64_t b;

  TestIntPairObj() = default;
  TestIntPairObj(int64_t a, int64_t b) : a(a), b(b) {}

  // Required: declare type information
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.TestIntPair", TestIntPairObj, tvm::ffi::Object);
};

// Step 2: Define the reference wrapper (user-facing interface)
class TestIntPair : public tvm::ffi::ObjectRef {
public:
  // Constructor
  explicit TestIntPair(int64_t a, int64_t b) {
    data_ = tvm::ffi::make_object<TestIntPairObj>(a, b);
  }

  // Required: define object reference methods
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestIntPair, tvm::ffi::ObjectRef, TestIntPairObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // register the object into the system
  // register field accessors and a global static function `__ffi_init__` as ffi::Function
  refl::ObjectDef<TestIntPairObj>()
    .def(refl::init<int64_t, int64_t>())
    .def_ro("a", &TestIntPairObj::a)
    .def_ro("b", &TestIntPairObj::b);
}
```

You can then create wrapper classes for objects that are in the library as follows:

```python
import tvm_ffi

# Register the class
@tvm_ffi.register_object("testing.TestIntPair")
class TestIntPair(tvm_ffi.Object):
    def __init__(self, a, b):
        # This is a special method to call an FFI function whose return
        # value exactly initializes the object handle of the object
        self.__ffi_init__(a, b)

test_int_pair = TestIntPair(1, 2)
# We can access the fields by name
# The properties are populated by the reflection mechanism
assert test_int_pair.a == 1
assert test_int_pair.b == 2
```

Under the hood, we leverage the information registered through the reflection registry to
generate efficient field accessors and methods for each class.

Importantly, when you have multiple inheritance, you need to call {py:func}`tvm_ffi.register_object`
on both the base class and the child class.
