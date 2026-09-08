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


import copy
import gc
import pickle
import weakref
from typing import NoReturn

import pytest
import tvm_ffi


def test_parse_backtrace() -> None:
    backtrace = """
    File "test.py", line 1, in <module>
    File "test.py", line 3, in run_test
    """
    parsed = tvm_ffi.error._parse_backtrace(backtrace)
    assert len(parsed) == 2
    assert parsed[0] == ("test.py", 1, "<module>")
    assert parsed[1] == ("test.py", 3, "run_test")


def test_error_from_cxx() -> None:
    test_raise_error = tvm_ffi.get_global_func("testing.test_raise_error")

    try:
        test_raise_error("ValueError", "error XYZ")
    except ValueError as e:
        assert e.__tvm_ffi_error__.kind == "ValueError"  # ty: ignore[unresolved-attribute]
        assert e.__tvm_ffi_error__.message == "error XYZ"  # ty: ignore[unresolved-attribute]
        assert e.__tvm_ffi_error__.backtrace.find("TestRaiseError") != -1  # ty: ignore[unresolved-attribute]

    fapply = tvm_ffi.convert(lambda f, *args: f(*args))

    with pytest.raises(TypeError):
        fapply(test_raise_error, "TypeError", "error XYZ")

    # wrong number of arguments
    with pytest.raises(TypeError):
        tvm_ffi.convert(lambda x: x)()


def test_error_from_nested_pyfunc() -> None:
    fapply = tvm_ffi.convert(lambda f, *args: f(*args))
    cxx_test_raise_error = tvm_ffi.get_global_func("testing.test_raise_error")
    cxx_test_apply = tvm_ffi.get_global_func("testing.apply")

    record_object = []

    def raise_error() -> None:
        try:
            fapply(cxx_test_raise_error, "ValueError", "error XYZ")
        except ValueError as e:
            assert e.__tvm_ffi_error__.kind == "ValueError"  # ty: ignore[unresolved-attribute]
            assert e.__tvm_ffi_error__.message == "error XYZ"  # ty: ignore[unresolved-attribute]
            assert e.__tvm_ffi_error__.backtrace.find("TestRaiseError") != -1  # ty: ignore[unresolved-attribute]
            record_object.append(e.__tvm_ffi_error__)  # ty: ignore[unresolved-attribute]
            raise e

    try:
        cxx_test_apply(raise_error)
    except ValueError as e:
        backtrace = e.__tvm_ffi_error__.backtrace  # ty: ignore[unresolved-attribute]
        assert e.__tvm_ffi_error__.same_as(record_object[0])  # ty: ignore[unresolved-attribute]
        assert backtrace.count("TestRaiseError") == 1
        # The following lines may fail if debug symbols are missing
        try:
            assert backtrace.count("TestApply") == 1
            assert backtrace.count("<lambda>") == 1
            pos_cxx_raise = backtrace.find("TestRaiseError")
            pos_cxx_apply = backtrace.find("TestApply")
            pos_lambda = backtrace.find("<lambda>")
            assert pos_cxx_raise < pos_lambda
            assert pos_lambda < pos_cxx_apply
        except Exception as e:
            pytest.xfail("May fail if debug symbols are missing")  # ty: ignore[invalid-argument-type, too-many-positional-arguments]


def test_error_traceback_update() -> None:
    fecho = tvm_ffi.get_global_func("testing.echo")

    def raise_error() -> NoReturn:
        raise ValueError("error XYZ")

    try:
        raise_error()
    except ValueError as e:
        ffi_error = tvm_ffi.convert(e)
        assert ffi_error.backtrace.find("raise_error") != -1

    def raise_cxx_error() -> None:
        cxx_test_raise_error = tvm_ffi.get_global_func("testing.test_raise_error")
        cxx_test_raise_error("ValueError", "error XYZ")

    try:
        raise_cxx_error()
    except ValueError as e:
        assert e.__tvm_ffi_error__.backtrace.find("raise_cxx_error") == -1  # ty: ignore[unresolved-attribute]
        ffi_error1 = tvm_ffi.convert(e)
        ffi_error2 = fecho(e)
        assert ffi_error1.backtrace.find("raise_cxx_error") != -1
        assert ffi_error2.backtrace.find("raise_cxx_error") != -1


def test_error_no_cyclic_reference() -> None:
    # This test case ensures that when an error is raised from C++ side,
    # there is no cyclic reference that slows down the garbage collection.
    # Please see `_with_append_backtrace` in error.py

    # temporarily disable gc
    gc.disable()

    try:
        # We should create a class as a probe to detect gc activity
        # beacuse weakref doesn't support list, dict or other trivial types
        class SampleObject: ...

        # trigger a C++ side KeyError by accessing a non-existent key
        def trigger_cpp_side_error() -> None:
            try:
                tmp_map: tvm_ffi.Map = tvm_ffi.Map(dict())
                tmp_map["a"]
            except KeyError:
                pass

        def may_create_cyclic_reference() -> weakref.ReferenceType:
            obj = SampleObject()
            trigger_cpp_side_error()
            return weakref.ref(obj)

        wref = may_create_cyclic_reference()

        # if the object is not collected, wref() will return the object
        assert wref() is None, "Cyclic reference occurs inside error handling pipeline"

    finally:
        # re-enable gc whenever exception occurs
        gc.enable()


def _raise_from_cxx() -> BaseException:
    """Return a Python exception produced by a C++-side FFI error."""
    test_raise_error = tvm_ffi.get_global_func("testing.test_raise_error")
    try:
        test_raise_error("ValueError", "error XYZ")
    except ValueError as e:
        return e
    raise AssertionError("expected the FFI call to raise")


def test_exception_pickle_roundtrip() -> None:
    """Exceptions carrying ``__tvm_ffi_error__`` must survive pickling.

    Regression test: ``ffi.Error`` has no JSON-graph creator, so the inherited
    ``CObject.__reduce__`` used to fail with a ``ToJSONGraph`` ``TypeError``
    whenever a test harness pickled an FFI-originated exception.
    """
    err = _raise_from_cxx()
    restored = pickle.loads(pickle.dumps(err))

    assert isinstance(restored, ValueError)
    assert restored.args == err.args
    ffi_error = restored.__tvm_ffi_error__  # ty: ignore[unresolved-attribute]
    assert isinstance(ffi_error, tvm_ffi.core.Error)
    assert ffi_error.kind == "ValueError"
    assert ffi_error.message == "error XYZ"
    assert ffi_error.backtrace.find("TestRaiseError") != -1


def test_exception_deepcopy_roundtrip() -> None:
    """``copy.deepcopy`` goes through ``__reduce_ex__`` and must work too."""
    err = _raise_from_cxx()
    restored = copy.deepcopy(err)

    assert isinstance(restored, ValueError)
    assert restored.__tvm_ffi_error__.kind == "ValueError"  # ty: ignore[unresolved-attribute]
    assert restored.__tvm_ffi_error__.message == "error XYZ"  # ty: ignore[unresolved-attribute]


def test_ffi_error_pickle_roundtrip() -> None:
    """A bare :class:`tvm_ffi.core.Error` round-trips by value."""
    error = tvm_ffi.core.Error("TypeError", "boom", 'File "a.py", line 1, in f\n')
    restored = pickle.loads(pickle.dumps(error))

    assert isinstance(restored, tvm_ffi.core.Error)
    assert restored.kind == "TypeError"
    assert restored.message == "boom"
    assert restored.backtrace == 'File "a.py", line 1, in f\n'
    # a fresh object, not the original handle
    assert not restored.same_as(error)


def test_ffi_error_pickle_null_handle() -> None:
    """An ``Error`` with a NULL handle round-trips without dereferencing it."""
    error = tvm_ffi.core.Error.__new__(tvm_ffi.core.Error)
    assert error.__chandle__() == 0

    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, tvm_ffi.core.Error)
    assert restored.__chandle__() == 0


def test_ffi_error_pickle_drops_extra_context() -> None:
    """``extra_context`` is intentionally not preserved across pickling.

    It may hold arbitrary native payloads with no value representation, so
    dropping it keeps pickling an error independent of where it came from.
    """
    error = tvm_ffi.core.Error("ValueError", "boom", "")
    restored = pickle.loads(pickle.dumps(error))
    assert restored.extra_context is None


def test_restored_exception_can_propagate_through_ffi() -> None:
    """An unpickled exception still works as an FFI error payload.

    ``set_last_ffi_error`` calls ``update_backtrace`` on ``__tvm_ffi_error__``,
    so the attribute must survive as a live ``Error``, not as ``None``.
    """
    restored = pickle.loads(pickle.dumps(_raise_from_cxx()))

    def callback(_: int) -> NoReturn:
        raise restored

    fapply = tvm_ffi.convert(callback)
    with pytest.raises(ValueError, match="error XYZ"):
        fapply(1)
