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
        assert e.__tvm_ffi_error__.kind == "ValueError"  # type: ignore[attr-defined]
        assert e.__tvm_ffi_error__.message == "error XYZ"  # type: ignore[attr-defined]
        assert e.__tvm_ffi_error__.backtrace.find("TestRaiseError") != -1  # type: ignore[attr-defined]

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
            assert e.__tvm_ffi_error__.kind == "ValueError"  # type: ignore[attr-defined]
            assert e.__tvm_ffi_error__.message == "error XYZ"  # type: ignore[attr-defined]
            assert e.__tvm_ffi_error__.backtrace.find("TestRaiseError") != -1  # type: ignore[attr-defined]
            record_object.append(e.__tvm_ffi_error__)  # type: ignore[attr-defined]
            raise e

    try:
        cxx_test_apply(raise_error)
    except ValueError as e:
        backtrace = e.__tvm_ffi_error__.backtrace  # type: ignore[attr-defined]
        assert e.__tvm_ffi_error__.same_as(record_object[0])  # type: ignore[attr-defined]
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
            pytest.xfail("May fail if debug symbols are missing")


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
        assert e.__tvm_ffi_error__.backtrace.find("raise_cxx_error") == -1  # type: ignore[attr-defined]
        ffi_error1 = tvm_ffi.convert(e)
        ffi_error2 = fecho(e)
        assert ffi_error1.backtrace.find("raise_cxx_error") != -1
        assert ffi_error2.backtrace.find("raise_cxx_error") != -1
