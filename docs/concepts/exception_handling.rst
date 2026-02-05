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

.. _sec-exception-handling:

Exception Handling
==================

TVM-FFI gracefully handles exceptions across language boundaries without requiring manual
error code management. This document covers throwing, catching, and propagating exceptions
in TVM-FFI functions.

.. important::
   Stack traces from all languages are properly preserved and concatenated in the TVM-FFI Stable C ABI.


Cross-language exceptions are **first-class citizens** in TVM-FFI.
TVM-FFI provides a stable C ABI for propagating exceptions across language boundaries,
and wraps this ABI to provide language-native exception handling without exposing the underlying C machinery.

**Error in C**. :cpp:class:`tvm::ffi::Error` is a TVM-FFI :doc:`object <object_and_class>` with a :cpp:class:`TVMFFIErrorCell` payload containing:

- ``kind``: Error type name (e.g., ``"ValueError"``, ``"RuntimeError"``)
- ``message``: Human-readable error message
- ``backtrace``: Stack trace from the point of error

**Propagating Errors in C**. For call chains, simply propagate return codes—TLS carries the error details:

.. code-block:: cpp

   int my_function(...) {
     int rc = some_ffi_call(...);
     if (rc != 0) return rc;  // Propagate error
     // Continue on success
     return 0;
   }

The following sections describe how to throw and catch exceptions across ABI and language boundaries.

Throwing Exceptions
-------------------

Python
~~~~~~

Raise native :py:class:`Exception <Exception>` instances or derived classes.
TVM-FFI catches these at the ABI boundary and converts them to :cpp:class:`tvm::ffi::Error` objects.
When C++ code calls into Python and a Python exception occurs, it propagates back to C++ as a
:cpp:class:`tvm::ffi::Error`, which C++ code can handle appropriately.

.. code-block:: python

   def my_function(x: int) -> int:
       if x < 0:
           raise ValueError(f"x must be non-negative, got {x}")
       return x + 1

TVM-FFI automatically maps common Python exception types to their corresponding error kinds:
``RuntimeError``, ``ValueError``, ``TypeError``, ``AttributeError``, ``KeyError``,
``IndexError``, ``AssertionError``, and ``MemoryError``. Custom exception types can be
registered using :py:func:`tvm_ffi.register_error`.

C++
~~~

Use :cpp:class:`tvm::ffi::Error` or the :c:macro:`TVM_FFI_THROW` macro:

.. code-block:: cpp

   #include <tvm/ffi/error.h>

   void ThrowError(int x) {
     if (x < 0) {
       TVM_FFI_THROW(ValueError) << "x must be non-negative, got " << x;
     }
   }

The :c:macro:`TVM_FFI_THROW` macro captures the current file name, line number, stack trace,
and error message, then constructs a :cpp:class:`tvm::ffi::Error` object. At the ABI boundary,
this error is stored in TLS and the function returns ``-1`` per the :cpp:type:`TVMFFISafeCallType`
calling convention (see :ref:`sec:function-calling-convention`).

Additional check macros are available for common validation patterns:

.. code-block:: cpp

   TVM_FFI_CHECK(condition, ErrorKind) << "message";  // Custom error kind
   TVM_FFI_ICHECK(condition) << "message";            // InternalError
   TVM_FFI_ICHECK_EQ(x, y) << "message";              // Check equality
   TVM_FFI_ICHECK_LT(x, y) << "message";              // Check less than
   // Also: TVM_FFI_ICHECK_GT, TVM_FFI_ICHECK_LE, TVM_FFI_ICHECK_GE, TVM_FFI_ICHECK_NE

.. hint::
   A detailed implementation of such graceful handling behavior can be found
   in :c:macro:`TVM_FFI_SAFE_CALL_BEGIN` / :c:macro:`TVM_FFI_SAFE_CALL_END` macros.

ANSI C
~~~~~~

For LLVM code generation and other C-based environments, use :cpp:func:`TVMFFIErrorSetRaisedFromCStr`
to set the TLS error and return ``-1``:

.. literalinclude:: ../../examples/abi_overview/example_code.c
   :language: c
   :start-after: [Error.RaiseException.begin]
   :end-before: [Error.RaiseException.end]

For constructing error messages from multiple parts (useful in code generators),
use :cpp:func:`TVMFFIErrorSetRaisedFromCStrParts`:

.. code-block:: cpp

   const char* parts[] = {"Expected ", "2", " arguments, got ", "1"};
   TVMFFIErrorSetRaisedFromCStrParts("ValueError", parts, 4);
   return -1;

.. _sec-exception:

Catching Exceptions in C
------------------------

In C++, Python, and many other languages, TVM-FFI exceptions are **first-class citizens**,
meaning they can be caught and handled like native exceptions:

.. code-block:: cpp

   try {
      // Calls a TVM-FFI function that throws an exception
   } catch (const tvm::ffi::Error& e) {
      // Handle the exception
      std::cout << e.kind() << ": " << e.message() << "\n" << e.backtrace() << "\n";
   }

.. important::

   This section covers the **pure C** low-level details of exception handling.
   For C++ and other languages, the example above is sufficient.


Checking Return Codes
~~~~~~~~~~~~~~~~~~~~~

When a TVM-FFI function returns a non-zero code, an error occurred.
The error object is stored in thread-local storage (TLS) and can be retrieved
with :cpp:func:`TVMFFIErrorMoveFromRaised`:

.. literalinclude:: ../../examples/abi_overview/example_code.c
   :language: c
   :start-after: [Error.HandleReturnCode.begin]
   :end-before: [Error.HandleReturnCode.end]

.. important::
   The caller must release the error object via :cpp:func:`TVMFFIObjectDecRef` to avoid memory leaks.

Accessing Error Details
~~~~~~~~~~~~~~~~~~~~~~~

The error payload is a :cpp:type:`TVMFFIErrorCell` structure containing the error kind, message,
and backtrace. Access it by skipping the :cpp:type:`TVMFFIObject` header:

.. literalinclude:: ../../examples/abi_overview/example_code.c
   :language: c
   :start-after: [Error.Print.begin]
   :end-before: [Error.Print.end]

Return Code Reference
~~~~~~~~~~~~~~~~~~~~~

- **Error code 0:** Success
- **Error code -1:** Error occurred, retrieve via :cpp:func:`TVMFFIErrorMoveFromRaised`

Further Reading
---------------

- :doc:`func_module`: Functions and modules that use this exception handling mechanism
- :doc:`object_and_class`: The object system that backs :cpp:class:`~tvm::ffi::Error`
- :doc:`any`: How errors are stored and transported in :cpp:class:`~tvm::ffi::Any` containers
- :doc:`abi_overview`: Low-level C ABI details for exceptions
- `tvm/ffi/error.h <https://github.com/apache/tvm-ffi/blob/main/include/tvm/ffi/error.h>`_: C++ error handling API
- `tvm/ffi/c_api.h <https://github.com/apache/tvm-ffi/blob/main/include/tvm/ffi/c_api.h>`_: C ABI error functions
