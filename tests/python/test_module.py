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

from __future__ import annotations

import gc
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import tvm_ffi.cpp
from tvm_ffi.module import Module


@pytest.fixture(scope="module")
def imports() -> Module:
    return tvm_ffi.cpp.load_inline(
        name="module_lifetime_imports",
        cpp_sources=r"""
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <atomic>
#include <memory>

using namespace tvm::ffi;
std::atomic<int> live_functions{0};
struct Lifetime {
  Lifetime() { ++live_functions; }
  ~Lifetime() { --live_functions; }
};

class ImportedModule : public ModuleObj {
 public:
  const char* kind() const final { return "test_import"; }
  int GetPropertyMask() const final { return Module::kRunnable; }
  Optional<Function> GetFunction(const String& name) final {
    if (name != "value" && name != "late_value") return std::nullopt;
    return Function::FromTyped([lifetime = std::make_shared<Lifetime>()]() { return 42; });
  }
};

class RootModule : public ModuleObj {
 public:
  const char* kind() const final { return "lifetime_root"; }
  Optional<Function> GetFunction(const String& name) final {
    return imports()[0].cast<Module>()->GetFunction(name, true);
  }
  void ImportModule(const Module& module) final {
    imports()[0].cast<Module>()->ImportModule(module);
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("ffi.Module.load_from_bytes.lifetime_root", [](const Bytes&) {
    return Module(make_object<RootModule>());
  });
}

Module MakeImport() { return Module(make_object<ImportedModule>()); }
int LiveFunctions() { return live_functions.load(); }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_import, MakeImport);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(live_functions, LiveFunctions);
""",
    )


@pytest.fixture(params=[False, True], ids=["library-root", "serialized-root"])
def library_path(tmp_path: Path, request: pytest.FixtureRequest, imports: Module) -> str:
    library_bin = ""
    if request.param:
        # A serialized root imports the shared library: root -> _lib.
        def encode(data: bytes) -> bytes:
            return struct.pack("<Q", len(data)) + data

        blob = struct.pack("<4Q", 3, 0, 1, 1) + struct.pack("<2Q", 1, 1)
        blob += encode(b"lifetime_root") + encode(b"0") + encode(b"_lib")
        data = ", ".join(str(v) for v in encode(blob))
        library_bin = (
            'extern "C" { TVM_FFI_DLL_EXPORT unsigned char __tvm_ffi__library_bin[] = {'
            + data
            + "}; }"
        )
    return tvm_ffi.cpp.build_inline(
        name="module_lifetime",
        build_directory=str(tmp_path),
        cpp_sources=r"""
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>

extern "C" {
TVM_FFI_DLL_EXPORT void* __tvm_ffi__library_ctx = nullptr;
}

namespace {
using namespace tvm::ffi;
WeakObjectPtr<FunctionObj> cached_function;
WeakObjectPtr<ModuleObj> context;
void* handle = nullptr;

void TrackContext() {
  context = detail::ObjectUnsafe::ObjectPtrFromUnowned<ModuleObj>(
      static_cast<ModuleObj*>(__tvm_ffi__library_ctx));
}
bool ContextAlive() { return !context.expired(); }
int InitHandle(void** result) {
  int status = TVMFFIEnvModLookupFromImports(__tvm_ffi__library_ctx, "value", result);
  if (status == 0) {
    cached_function = detail::ObjectUnsafe::ObjectPtrFromUnowned<FunctionObj>(
        static_cast<FunctionObj*>(*result));
  }
  return status;
}
int Call() {
  TVM_FFI_CHECK_SAFE_CALL(TVMFFIHandleInitOnce(&handle, InitHandle));
  TVM_FFI_CHECK(!cached_function.expired(), RuntimeError) << "Cached function was freed";
  Any result;
  TVM_FFI_CHECK_SAFE_CALL(
      TVMFFIFunctionCall(handle, nullptr, 0, reinterpret_cast<TVMFFIAny*>(&result)));
  return result.cast<int>();
}
bool CachedFunctionAlive() { return !cached_function.expired(); }
int Constant() { return 42; }
int LateCall() {
  TVM_FFI_CHECK(!context.expired(), RuntimeError) << "Library context was freed";
  void* func;
  TVM_FFI_CHECK_SAFE_CALL(
      TVMFFIEnvModLookupFromImports(__tvm_ffi__library_ctx, "late_value", &func));
  Any result;
  TVM_FFI_CHECK_SAFE_CALL(
      TVMFFIFunctionCall(func, nullptr, 0, reinterpret_cast<TVMFFIAny*>(&result)));
  return result.cast<int>();
}
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(call, Call);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cached_function_alive, CachedFunctionAlive);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(track_context, TrackContext);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(context_alive, ContextAlive);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(late_call, LateCall);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(constant, Constant);
"""
        + library_bin,
    )


@pytest.mark.parametrize("release_first", [True, False])
def test_repeated_load_keeps_imported_function_alive(
    library_path: str, imports: Module, release_first: bool
) -> None:
    first = tvm_ffi.load_module(library_path, keep_module_alive=False)
    first.import_module(imports.make_import())
    assert first.call() == 42

    second = tvm_ffi.load_module(library_path, keep_module_alive=False)
    second.import_module(imports.make_import())
    second.track_context()
    remaining = second if release_first else first
    del first, second
    gc.collect()

    assert remaining.cached_function_alive()
    assert remaining.context_alive()
    assert remaining.call() == 42
    assert remaining.late_call() == 42
    del remaining
    gc.collect()
    assert imports.live_functions() == 0


def test_function_outlives_repeated_load(library_path: str, imports: Module) -> None:
    first = tvm_ffi.load_module(library_path, keep_module_alive=False)
    first.import_module(imports.make_import())
    call = first.call
    assert call() == 42
    del first
    gc.collect()
    second = tvm_ffi.load_module(library_path, keep_module_alive=False)
    del second
    gc.collect()
    assert call() == 42
    del call
    gc.collect()
    assert imports.live_functions() == 0


@pytest.mark.skipif(os.name == "nt", reason="Creating symlinks may require administrator rights")
def test_repeated_load_through_symlink(library_path: str, imports: Module, tmp_path: Path) -> None:
    alias = tmp_path / "alias.so"
    alias.symlink_to(library_path)
    first = tvm_ffi.load_module(library_path, keep_module_alive=False)
    first.import_module(imports.make_import())
    assert first.call() == 42
    second = tvm_ffi.load_module(alias, keep_module_alive=False)
    del first
    gc.collect()
    assert second.cached_function_alive()
    assert second.call() == 42


def test_concurrent_load(library_path: str, imports: Module) -> None:
    def load(_: int) -> Module:
        return tvm_ffi.load_module(library_path, keep_module_alive=False)

    with ThreadPoolExecutor(max_workers=4) as pool:
        modules = list(pool.map(load, range(16)))
    modules[0].import_module(imports.make_import())
    assert modules[0].call() == 42
    remaining = modules[-1]
    del modules
    gc.collect()
    assert remaining.cached_function_alive()
    assert remaining.call() == 42


def test_load_after_final_release(library_path: str, imports: Module) -> None:
    for _ in range(3):
        module = tvm_ffi.load_module(library_path, keep_module_alive=False)
        module.import_module(imports.make_import())
        assert module.call() == 42
        del module
        gc.collect()
        assert imports.live_functions() == 0


def test_concurrent_load_and_release(library_path: str) -> None:
    def load_and_release(_: int) -> None:
        for _ in range(32):
            module = tvm_ffi.load_module(library_path, keep_module_alive=False)
            assert module.get_function("constant")() == 42
            del module

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(load_and_release, range(4)))
