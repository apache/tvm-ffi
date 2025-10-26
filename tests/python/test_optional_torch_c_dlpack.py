import pytest
import os
from tvm_ffi._optional_torch_c_dlpack import build_torch_c_dlpack_extension
import tvm_ffi


def test_build_torch_c_dlpack_extension():
    libpath = build_torch_c_dlpack_extension(build_directory="./build_test_dir")
    assert os.path.exists(libpath)

    mod = tvm_ffi.load_module(libpath)
    assert mod is not None

    assert mod.TorchDLPackExchangeAPIPtr() is not None

if __name__ == "__main__":
    pytest.main([__file__])
