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
"""Build Torch C DLPack Addon."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.torch_version
import torch.utils.cpp_extension

# Important: to avoid cyclic dependency, we avoid import tvm_ffi names at top level here.

IS_WINDOWS = sys.platform == "win32"
IS_DARWIN = sys.platform == "darwin"

cpp_source = """
#include <dlpack/dlpack.h>
#include <ATen/DLConvertor.h>
#include <ATen/Functions.h>
#include <torch/extension.h>

#ifdef BUILD_WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#endif
#ifdef BUILD_WITH_ROCM
#include <c10/hip/HIPStream.h>
#endif

using namespace std;
namespace at {
namespace {

DLDataType getDLDataTypeForDLPackv1(const Tensor& t) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = t.element_size() * 8;
  switch (t.scalar_type()) {
    case ScalarType::UInt1:
    case ScalarType::UInt2:
    case ScalarType::UInt3:
    case ScalarType::UInt4:
    case ScalarType::UInt5:
    case ScalarType::UInt6:
    case ScalarType::UInt7:
    case ScalarType::Byte:
    case ScalarType::UInt16:
    case ScalarType::UInt32:
    case ScalarType::UInt64:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
#if (TORCH_VERSION_MAJOR > 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    case ScalarType::Int1:
    case ScalarType::Int2:
    case ScalarType::Int3:
    case ScalarType::Int4:
    case ScalarType::Int5:
    case ScalarType::Int6:
    case ScalarType::Int7:
#endif
    case ScalarType::Char:
    case ScalarType::Short:
    case ScalarType::Int:
    case ScalarType::Long:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Half:
    case ScalarType::Float:
    case ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Bool:
      dtype.code = DLDataTypeCode::kDLBool;
      break;
    case ScalarType::ComplexHalf:
    case ScalarType::ComplexFloat:
    case ScalarType::ComplexDouble:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case ScalarType::BFloat16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      break;
    case ScalarType::Float8_e5m2:
      dtype.code = DLDataTypeCode::kDLFloat8_e5m2;
      break;
    case ScalarType::Float8_e5m2fnuz:
      dtype.code = DLDataTypeCode::kDLFloat8_e5m2fnuz;
      break;
    case ScalarType::Float8_e4m3fn:
      dtype.code = DLDataTypeCode::kDLFloat8_e4m3fn;
      break;
    case ScalarType::Float8_e4m3fnuz:
      dtype.code = DLDataTypeCode::kDLFloat8_e4m3fnuz;
      break;
#if (TORCH_VERSION_MAJOR > 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
    case ScalarType::Float8_e8m0fnu:
      dtype.code = DLDataTypeCode::kDLFloat8_e8m0fnu;
      break;
#endif
#if (TORCH_VERSION_MAJOR > 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 8)
    case ScalarType::Float4_e2m1fn_x2:
      dtype.code = DLDataTypeCode::kDLFloat4_e2m1fn;
      dtype.lanes = 2;
      dtype.bits = 4;
      break;
#endif
    default:
      TORCH_CHECK(false, "Unsupported scalar type: ");
  }
  return dtype;
}

DLDevice torchDeviceToDLDeviceForDLPackv1(at::Device device) {
  DLDevice ctx;

  ctx.device_id = (device.is_cuda() || device.is_privateuseone())
                      ? static_cast<int32_t>(static_cast<unsigned char>(device.index()))
                      : 0;

  switch (device.type()) {
    case DeviceType::CPU:
      ctx.device_type = DLDeviceType::kDLCPU;
      break;
    case DeviceType::CUDA:
#ifdef USE_ROCM
      ctx.device_type = DLDeviceType::kDLROCM;
#else
      ctx.device_type = DLDeviceType::kDLCUDA;
#endif
      break;
    case DeviceType::OPENCL:
      ctx.device_type = DLDeviceType::kDLOpenCL;
      break;
    case DeviceType::HIP:
      ctx.device_type = DLDeviceType::kDLROCM;
      break;
    case DeviceType::XPU:
      ctx.device_type = DLDeviceType::kDLOneAPI;
      ctx.device_id = at::detail::getXPUHooks().getGlobalIdxFromDevice(device);
      break;
    case DeviceType::MAIA:
      ctx.device_type = DLDeviceType::kDLMAIA;
      break;
    case DeviceType::PrivateUse1:
      ctx.device_type = DLDeviceType::kDLExtDev;
      break;
    case DeviceType::MPS:
      ctx.device_type = DLDeviceType::kDLMetal;
      break;
    default:
      TORCH_CHECK(false, "Cannot pack tensors on " + device.str());
  }

  return ctx;
}

template <class T>
struct ATenDLMTensor {
  Tensor handle;
  T tensor{};
};

template <class T>
void deleter(T* arg) {
  delete static_cast<ATenDLMTensor<T>*>(arg->manager_ctx);
}

// Adds version information for DLManagedTensorVersioned.
// This is a no-op for the other types.
template <class T>
void fillVersion(T* tensor) {}

template <>
void fillVersion<DLManagedTensorVersioned>(DLManagedTensorVersioned* tensor) {
  tensor->flags = 0;
  tensor->version.major = DLPACK_MAJOR_VERSION;
  tensor->version.minor = DLPACK_MINOR_VERSION;
}

// This function returns a shared_ptr to memory managed DLpack tensor
// constructed out of ATen tensor
template <class T>
T* toDLPackImpl(const Tensor& src) {
  ATenDLMTensor<T>* atDLMTensor(new ATenDLMTensor<T>);
  atDLMTensor->handle = src;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.deleter = &deleter<T>;
  atDLMTensor->tensor.dl_tensor.data = src.data_ptr();
  atDLMTensor->tensor.dl_tensor.device = torchDeviceToDLDeviceForDLPackv1(src.device());
  atDLMTensor->tensor.dl_tensor.ndim = static_cast<int32_t>(src.dim());
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataTypeForDLPackv1(src);
  atDLMTensor->tensor.dl_tensor.shape = const_cast<int64_t*>(src.sizes().data());
  atDLMTensor->tensor.dl_tensor.strides = const_cast<int64_t*>(src.strides().data());
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  fillVersion(&atDLMTensor->tensor);
  return &(atDLMTensor->tensor);
}

static Device getATenDeviceForDLPackv1(DLDeviceType type, c10::DeviceIndex index,
                                       void* data = nullptr) {
  switch (type) {
    case DLDeviceType::kDLCPU:
      return at::Device(DeviceType::CPU);
#ifndef USE_ROCM
    // if we are compiled under HIP, we cannot do cuda
    case DLDeviceType::kDLCUDA:
      return at::Device(DeviceType::CUDA, index);
#endif
    case DLDeviceType::kDLOpenCL:
      return at::Device(DeviceType::OPENCL, index);
    case DLDeviceType::kDLROCM:
#ifdef USE_ROCM
      // this looks funny, we need to return CUDA here to masquerade
      return at::Device(DeviceType::CUDA, index);
#else
      return at::Device(DeviceType::HIP, index);
#endif
    case DLDeviceType::kDLOneAPI:
      TORCH_CHECK(data != nullptr, "Can't get ATen device for XPU without XPU data.");
      return at::detail::getXPUHooks().getDeviceFromPtr(data);
    case DLDeviceType::kDLMAIA:
      return at::Device(DeviceType::MAIA, index);
    case DLDeviceType::kDLExtDev:
      return at::Device(DeviceType::PrivateUse1, index);
    case DLDeviceType::kDLMetal:
      return at::Device(DeviceType::MPS, index);
    default:
      TORCH_CHECK(false, "Unsupported device_type: ", std::to_string(type));
  }
}

ScalarType toScalarTypeForDLPackv1(const DLDataType& dtype) {
  ScalarType stype = ScalarType::Undefined;
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 8
  if (dtype.code != DLDataTypeCode::kDLFloat4_e2m1fn) {
    TORCH_CHECK(dtype.lanes == 1, "ATen does not support lanes != 1 for dtype code",
                std::to_string(dtype.code));
  }
#endif
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Byte;
          break;
        case 16:
          stype = ScalarType::UInt16;
          break;
        case 32:
          stype = ScalarType::UInt32;
          break;
        case 64:
          stype = ScalarType::UInt64;
          break;
        default:
          TORCH_CHECK(false, "Unsupported kUInt bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLInt:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Char;
          break;
        case 16:
          stype = ScalarType::Short;
          break;
        case 32:
          stype = ScalarType::Int;
          break;
        case 64:
          stype = ScalarType::Long;
          break;
        default:
          TORCH_CHECK(false, "Unsupported kInt bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat:
      switch (dtype.bits) {
        case 16:
          stype = ScalarType::Half;
          break;
        case 32:
          stype = ScalarType::Float;
          break;
        case 64:
          stype = ScalarType::Double;
          break;
        default:
          TORCH_CHECK(false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLBfloat:
      switch (dtype.bits) {
        case 16:
          stype = ScalarType::BFloat16;
          break;
        default:
          TORCH_CHECK(false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLComplex:
      switch (dtype.bits) {
        case 32:
          stype = ScalarType::ComplexHalf;
          break;
        case 64:
          stype = ScalarType::ComplexFloat;
          break;
        case 128:
          stype = ScalarType::ComplexDouble;
          break;
        default:
          TORCH_CHECK(false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLBool:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Bool;
          break;
        default:
          TORCH_CHECK(false, "Unsupported kDLBool bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e5m2:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Float8_e5m2;
          break;
        default:
          TORCH_CHECK(false, "Unsupported kDLFloat8_e5m2 bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e5m2fnuz:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Float8_e5m2fnuz;
          break;
        default:
          TORCH_CHECK(false, "Unsupported kDLFloat8_e5m2fnuz bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e4m3fn:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Float8_e4m3fn;
          break;
        default:
          TORCH_CHECK(false, "Unsupported kDLFloat8_e4m3fn bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e4m3fnuz:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Float8_e4m3fnuz;
          break;
        default:
          TORCH_CHECK(false, "Unsupported kDLFloat8_e4m3fnuz bits ", std::to_string(dtype.bits));
      }
      break;
#if (TORCH_VERSION_MAJOR > 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
    case DLDataTypeCode::kDLFloat8_e8m0fnu:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Float8_e8m0fnu;
          break;
        default:
          TORCH_CHECK(false, "Unsupported kDLFloat8_e8m0fnu bits ", std::to_string(dtype.bits));
      }
      break;
#endif
#if (TORCH_VERSION_MAJOR > 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 8)
    case DLDataTypeCode::kDLFloat4_e2m1fn:
      switch (dtype.bits) {
        case 4:
          switch (dtype.lanes) {
            case 2:
              stype = ScalarType::Float4_e2m1fn_x2;
              break;
            default:
              TORCH_CHECK(false, "Unsupported kDLFloat4_e2m1fn lanes ",
                          std::to_string(dtype.lanes));
          }
          break;
        default:
          TORCH_CHECK(false, "Unsupported kDLFloat4_e2m1fn bits ", std::to_string(dtype.bits));
      }
      break;
#endif
    default:
      TORCH_CHECK(false, "Unsupported code ", std::to_string(dtype.code));
  }
  return stype;
}

// This function constructs a Tensor from a memory managed DLPack which
// may be represented as either: DLManagedTensor and DLManagedTensorVersioned.
template <class T>
at::Tensor fromDLPackImpl(T* src, std::function<void(void*)> deleter) {
  if (!deleter) {
    deleter = [src](void* self [[maybe_unused]]) {
      if (src->deleter) {
        src->deleter(src);
      }
    };
  }

  DLTensor& dl_tensor = src->dl_tensor;
  Device device = getATenDeviceForDLPackv1(dl_tensor.device.device_type, dl_tensor.device.device_id,
                                           dl_tensor.data);
  ScalarType stype = toScalarTypeForDLPackv1(dl_tensor.dtype);

  if (!dl_tensor.strides) {
    return at::from_blob(dl_tensor.data, IntArrayRef(dl_tensor.shape, dl_tensor.ndim),
                         std::move(deleter), at::device(device).dtype(stype), {device});
  }
  return at::from_blob(dl_tensor.data, IntArrayRef(dl_tensor.shape, dl_tensor.ndim),
                       IntArrayRef(dl_tensor.strides, dl_tensor.ndim), deleter,
                       at::device(device).dtype(stype), {device});
}

void toDLPackNonOwningImpl(const Tensor& tensor, DLTensor& out) {
  // Fill in the pre-allocated DLTensor struct with direct pointers
  // This is a non-owning conversion - the caller owns the tensor
  // and must keep it alive for the duration of DLTensor usage
  out.data = tensor.data_ptr();
  out.device = torchDeviceToDLDeviceForDLPackv1(tensor.device());
  out.ndim = static_cast<int32_t>(tensor.dim());
  out.dtype = getDLDataTypeForDLPackv1(tensor);
  // sizes() and strides() return pointers to TensorImpl's stable storage
  // which remains valid as long as the tensor is alive
  out.shape = const_cast<int64_t*>(tensor.sizes().data());
  out.strides = const_cast<int64_t*>(tensor.strides().data());
  out.byte_offset = 0;
}

}  // namespace
}  // namespace at

struct TorchDLPackExchangeAPI : public DLPackExchangeAPI {
  TorchDLPackExchangeAPI() {
    header.version.major = DLPACK_MAJOR_VERSION;
    header.version.minor = DLPACK_MINOR_VERSION;
    header.prev_api = nullptr;
    managed_tensor_allocator = ManagedTensorAllocator;
    managed_tensor_from_py_object_no_sync = ManagedTensorFromPyObjectNoSync;
    managed_tensor_to_py_object_no_sync = ManagedTensorToPyObjectNoSync;
    dltensor_from_py_object_no_sync = DLTensorFromPyObjectNoSync;
    current_work_stream = CurrentWorkStream;
  }

  static const DLPackExchangeAPI* Global() {
    static TorchDLPackExchangeAPI inst;
    return &inst;
  }

 private:
  static int DLTensorFromPyObjectNoSync(void* py_obj, DLTensor* out) {
    try {
      // Use handle (non-owning) to avoid unnecessary refcount operations
      py::handle handle(static_cast<PyObject*>(py_obj));
      at::Tensor tensor = handle.cast<at::Tensor>();
      at::toDLPackNonOwningImpl(tensor, *out);
      return 0;
    } catch (const std::exception& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      return -1;
    }
  }

  static int ManagedTensorFromPyObjectNoSync(void* py_obj, DLManagedTensorVersioned** out) {
    try {
      py::handle handle(static_cast<PyObject*>(py_obj));
      at::Tensor tensor = handle.cast<at::Tensor>();
      *out = at::toDLPackImpl<DLManagedTensorVersioned>(tensor);
      return 0;
    } catch (const std::exception& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      return -1;
    }
  }

  // Get current CUDA/ROCm work stream
  static int CurrentWorkStream(DLDeviceType device_type, int32_t device_id, void** out_stream) {
    try {
#ifdef BUILD_WITH_CUDA
      if (device_type == kDLCUDA || device_type == kDLROCM) {
        *out_stream = at::cuda::getCurrentCUDAStream(device_id).stream();
        return 0;
      }
#endif
      // For CPU and other devices, return NULL (no stream concept)
      *out_stream = nullptr;
      return 0;
    } catch (const std::exception& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      return -1;
    }
  }

  static int ManagedTensorToPyObjectNoSync(DLManagedTensorVersioned* src, void** py_obj_out) {
    try {
      at::Tensor tensor = at::fromDLPackImpl<DLManagedTensorVersioned>(src, nullptr);
      *py_obj_out = THPVariable_Wrap(tensor);
      return 0;
    } catch (const std::exception& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      return -1;
    }
  }

  static int ManagedTensorAllocator(DLTensor* prototype, DLManagedTensorVersioned** out,
                                    void* error_ctx,
                                    void (*SetError)(void* error_ctx, const char* kind,
                                                     const char* message)) {
    try {
      at::IntArrayRef shape(prototype->shape, prototype->shape + prototype->ndim);
      at::TensorOptions options =
          at::TensorOptions()
              .dtype(at::toScalarType(prototype->dtype))
              .device(at::getATenDeviceForDLPackv1(prototype->device.device_type,
                                                   prototype->device.device_id));
      at::Tensor tensor = at::empty(shape, options);
      *out = at::toDLPackImpl<DLManagedTensorVersioned>(tensor);
      return 0;
    } catch (const std::exception& e) {
      SetError(error_ctx, "MemoryError", e.what());
      return -1;
    }
  }
};

// defien a cross-platgorm macro to export the symbol
#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

extern "C" DLL_EXPORT int64_t TorchDLPackExchangeAPIPtr() {
  return reinterpret_cast<int64_t>(TorchDLPackExchangeAPI::Global());
}
"""


def parse_env_flags(env_var_name: str) -> list[str]:
    env_flags = os.environ.get(env_var_name)
    if env_flags:
        try:
            import shlex  # noqa: PLC0415

            return shlex.split(env_flags)
        except ValueError as e:
            print(
                f"Warning: Could not parse {env_var_name} with shlex: {e}. Falling back to simple split.",
                file=sys.stderr,
            )
            return env_flags.split()
    return []


def _run_build_on_linux_like(
    build_dir: Path,
    libname: str,
    source_path: Path,
    extra_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    extra_include_paths: Sequence[str],
) -> None:
    """Build the module directly by invoking compiler commands (non-Windows only)."""
    from tvm_ffi.libinfo import find_dlpack_include_path  # noqa: PLC0415

    default_cflags = ["-std=c++17", "-fPIC", "-O3", "-fvisibility=hidden"]
    # Platform-specific linker flags
    if IS_DARWIN:
        # macOS uses @loader_path instead of $ORIGIN
        default_ldflags = ["-shared", "-Wl,-rpath,@loader_path"]
    else:
        # Linux uses $ORIGIN
        default_ldflags = ["-shared", "-Wl,-rpath,$ORIGIN"]

    cflags = default_cflags + [flag.strip() for flag in extra_cflags]
    ldflags = default_ldflags + [flag.strip() for flag in extra_ldflags]
    include_paths = [find_dlpack_include_path()] + [
        str(Path(path).resolve()) for path in extra_include_paths
    ]

    # append include paths
    for path in include_paths:
        cflags.extend(["-I", str(path)])

    # Get compiler and build paths
    cxx = os.environ.get("CXX", "c++")
    source_path_resolved = source_path.resolve()
    lib_path = build_dir / libname

    # Build command: compile and link in one step
    build_cmd = [cxx, *cflags, str(source_path_resolved), *ldflags, "-o", str(lib_path)]

    # Run build command
    status = subprocess.run(build_cmd, cwd=str(build_dir), capture_output=True, check=False)
    if status.returncode != 0:
        msg = [f"Build failed with status {status.returncode}"]
        if status.stdout:
            msg.append(f"stdout:\n{status.stdout.decode('utf-8')}")
        if status.stderr:
            msg.append(f"stderr:\n{status.stderr.decode('utf-8')}")
        raise RuntimeError("\n".join(msg))


def _generate_ninja_build_windows(
    build_dir: Path,
    libname: str,
    source_path: Path,
    extra_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    extra_include_paths: Sequence[str],
) -> None:
    """Generate the content of build.ninja for building the module on Windows."""
    from tvm_ffi.libinfo import find_dlpack_include_path  # noqa: PLC0415

    default_cflags = [
        "/std:c++17",
        "/MD",
        "/wd4819",
        "/wd4251",
        "/wd4244",
        "/wd4267",
        "/wd4275",
        "/wd4018",
        "/wd4190",
        "/wd4624",
        "/wd4067",
        "/wd4068",
        "/EHsc",
    ]
    default_ldflags = ["/DLL"]

    cflags = default_cflags + [flag.strip() for flag in extra_cflags]
    ldflags = default_ldflags + [flag.strip() for flag in extra_ldflags]
    include_paths = [find_dlpack_include_path()] + [
        str(Path(path).resolve()) for path in extra_include_paths
    ]

    # append include paths
    for path in include_paths:
        path_str = str(path)
        if " " in path_str:
            path_str = f'"{path_str}"'
        path_str = path_str.replace(":", "$:")
        cflags.append(f"-I{path_str}")

    # flags
    ninja = []
    ninja.append("ninja_required_version = 1.3")
    ninja.append("cxx = {}".format(os.environ.get("CXX", "cl")))
    ninja.append("cflags = {}".format(" ".join(cflags)))
    ninja.append("ldflags = {}".format(" ".join(ldflags)))

    # rules
    ninja.append("")
    ninja.append("rule compile")
    ninja.append("  command = $cxx /showIncludes $cflags -c $in /Fo$out")
    ninja.append("  deps = msvc")
    ninja.append("")

    ninja.append("rule link")
    ninja.append("  command = $cxx $in /link $ldflags /out:$out")
    ninja.append("")

    # build targets
    obj_name = "main.obj"
    ninja.append(
        "build {}: compile {}".format(obj_name, str(source_path.resolve()).replace(":", "$:"))
    )

    # Use appropriate extension based on platform
    ninja.append(f"build {libname}: link {obj_name}")
    ninja.append("")

    # default target
    ninja.append(f"default {libname}")
    ninja.append("")

    with open(build_dir / "build.ninja", "w") as f:  # noqa: PTH123
        f.write("\n".join(ninja))


def get_torch_include_paths(build_with_cuda: bool) -> Sequence[str]:
    """Get the include paths for building with torch."""
    if torch.__version__ >= torch.torch_version.TorchVersion("2.6.0"):
        return torch.utils.cpp_extension.include_paths(
            device_type="cuda" if build_with_cuda else "cpu"
        )
    else:
        return torch.utils.cpp_extension.include_paths(cuda=build_with_cuda)  # type: ignore[call-arg]


def main() -> None:  # noqa: PLR0912, PLR0915
    """Build the torch c dlpack extension."""
    # we need to set the following env to avoid tvm_ffi to build the torch c-dlpack addon during importing
    os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
    from tvm_ffi.utils.lockfile import FileLock  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Build the torch c dlpack extension. After building, a shared library will be placed in the output directory.",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        required=False,
        help="Directory to store the built extension library. If not provided, a temporary directory will be used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default=str(Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")).expanduser()),
        help="Directory to store the built extension library. If not specified, the default cache directory of tvm-ffi will be used.",
    )
    parser.add_argument(
        "--build-with-cuda",
        action="store_true",
        help="Build with CUDA support.",
    )
    parser.add_argument(
        "--build-with-rocm",
        action="store_true",
        help="Build with ROCm support.",
    )
    parser.add_argument(
        "--libname",
        type=str,
        default="auto",
        help="The name of the generated library. It can be a name 'auto' to auto-generate a name following 'libtorch_c_dlpack_addon_torch{version.major}{version.minor}-cpu/cuda.{extension}'.",
    )

    args = parser.parse_args()
    if args.build_with_cuda and args.build_with_rocm:
        raise ValueError("Cannot enable both CUDA and ROCm at the same time.")

    # resolve build directory
    if args.build_dir is None:
        build_dir = Path(tempfile.mkdtemp(prefix="tvm-ffi-torch-c-dlpack-"))
    else:
        build_dir = Path(args.build_dir)
    build_dir = build_dir.resolve()
    if not build_dir.exists():
        build_dir.mkdir(parents=True, exist_ok=True)

    # resolve library name
    if args.libname == "auto":
        major, minor = torch.__version__.split(".")[:2]
        if args.build_with_cuda:
            device = "cuda"
        elif args.build_with_rocm:
            device = "rocm"
        else:
            device = "cpu"
        suffix = ".dll" if IS_WINDOWS else ".so"
        libname = f"libtorch_c_dlpack_addon_torch{major}{minor}-{device}{suffix}"
    else:
        libname = args.libname
    tmp_libname = libname + ".tmp"

    # create output directory is not exists
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(str(output_dir / (libname + ".lock"))):
        if (output_dir / libname).exists():
            # already built
            return

        # write the source
        source_path = build_dir / "addon.cc"
        with open(source_path, "w") as f:  # noqa: PTH123
            f.write(cpp_source)

        # resolve configs
        include_paths = []
        ldflags = []
        cflags = []
        include_paths.append(sysconfig.get_paths()["include"])

        if args.build_with_cuda:
            cflags.append("-DBUILD_WITH_CUDA")
        elif args.build_with_rocm:
            cflags.extend(torch.utils.cpp_extension.COMMON_HIP_FLAGS)
            cflags.append("-DBUILD_WITH_ROCM")
        include_paths.extend(get_torch_include_paths(args.build_with_cuda or args.build_with_rocm))

        # use CXX11 ABI
        if torch.compiled_with_cxx11_abi():
            cflags.append("-D_GLIBCXX_USE_CXX11_ABI=1")
        else:
            cflags.append("-D_GLIBCXX_USE_CXX11_ABI=0")

        for lib_dir in torch.utils.cpp_extension.library_paths():
            if IS_WINDOWS:
                ldflags.append(f"/LIBPATH:{lib_dir}")
            else:
                ldflags.extend(["-L", str(lib_dir)])

        # Add all required PyTorch libraries
        if IS_WINDOWS:
            # On Windows, use .lib format for linking
            ldflags.extend(["c10.lib", "torch.lib", "torch_cpu.lib", "torch_python.lib"])
            if args.build_with_cuda:
                ldflags.extend(["torch_cuda.lib", "c10_cuda.lib"])
        else:
            # On Unix/macOS, use -l format for linking
            ldflags.extend(["-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_python"])
            if args.build_with_cuda:
                ldflags.extend(["-ltorch_cuda", "-lc10_cuda"])

        # Add Python library linking
        if IS_WINDOWS:
            python_lib = f"python{sys.version_info.major}.lib"
            python_libdir_list = [
                sysconfig.get_config_var("LIBDIR"),
                str(Path(sys.base_exec_prefix) / "libs"),
            ]
            if (
                sysconfig.get_path("include") is not None
                and (Path(sysconfig.get_path("include")).parent / "libs").exists()
            ):
                python_libdir_list.append(
                    str((Path(sysconfig.get_path("include")).parent / "libs").resolve())
                )
            for python_libdir in python_libdir_list:
                if python_libdir and (Path(python_libdir) / python_lib).exists():
                    ldflags.append(f"/LIBPATH:{python_libdir.replace(':', '$:')}")
                    break

        if IS_DARWIN:
            python_libdir = sysconfig.get_config_var("LIBDIR")
            if python_libdir:
                py_version = f"python{sysconfig.get_python_version()}"
                ldflags.append(f"-L{python_libdir}")
                ldflags.append(f"-l{py_version}")

        env_ldflags = parse_env_flags("TVM_FFI_JIT_EXTRA_LDFLAGS")
        if env_ldflags:
            ldflags.extend(env_ldflags)

        env_cflags = parse_env_flags("TVM_FFI_JIT_EXTRA_CFLAGS")
        if env_cflags:
            cflags.extend(env_cflags)

        # build the shared library
        if IS_WINDOWS:
            # Use ninja on Windows
            _generate_ninja_build_windows(
                build_dir=build_dir,
                libname=tmp_libname,
                source_path=source_path,
                extra_cflags=cflags,
                extra_ldflags=ldflags,
                extra_include_paths=include_paths,
            )
            from tvm_ffi.cpp.extension import build_ninja  # noqa: PLC0415

            build_ninja(build_dir=str(build_dir))
        else:
            # Use direct command on non-Windows
            _run_build_on_linux_like(
                build_dir=build_dir,
                libname=tmp_libname,
                source_path=source_path,
                extra_cflags=cflags,
                extra_ldflags=ldflags,
                extra_include_paths=include_paths,
            )

        # rename the tmp file to final libname
        shutil.move(str(build_dir / tmp_libname), str(output_dir / libname))


if __name__ == "__main__":
    main()
