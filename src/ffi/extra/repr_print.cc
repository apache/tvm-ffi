/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
 * \file src/ffi/extra/repr_print.cc
 *
 * \brief Reflection-based repr printing with DFS-based cycle/DAG handling.
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>

namespace tvm {
namespace ffi {

namespace {

/*!
 * \brief Convert a DLDeviceType to a short name string.
 */
const char* DeviceTypeName(int device_type) {
  switch (device_type) {
    case kDLCPU:
      return "cpu";
    case kDLCUDA:
      return "cuda";
    case kDLCUDAHost:
      return "cuda_host";
    case kDLOpenCL:
      return "opencl";
    case kDLVulkan:
      return "vulkan";
    case kDLMetal:
      return "metal";
    case kDLVPI:
      return "vpi";
    case kDLROCM:
      return "rocm";
    case kDLROCMHost:
      return "rocm_host";
    case kDLExtDev:
      return "ext_dev";
    case kDLCUDAManaged:
      return "cuda_managed";
    case kDLOneAPI:
      return "oneapi";
    case kDLWebGPU:
      return "webgpu";
    case kDLHexagon:
      return "hexagon";
    default:
      return "unknown";
  }
}

/*!
 * \brief Format a DLDevice as "device_name:device_id".
 */
std::string DeviceToString(DLDevice device) {
  std::ostringstream os;
  os << DeviceTypeName(device.device_type) << ":" << device.device_id;
  return os.str();
}

/*!
 * \brief Format raw bytes as a Python-style bytes literal: b"...".
 */
std::string FormatBytes(const char* data, size_t size) {
  std::ostringstream os;
  os << "b\"";
  for (size_t i = 0; i < size; ++i) {
    unsigned char c = static_cast<unsigned char>(data[i]);
    if (c >= 32 && c < 127 && c != '\"' && c != '\\') {
      os << static_cast<char>(c);
    } else {
      os << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c);
    }
  }
  os << "\"";
  return os.str();
}

/*!
 * \brief Format an object address as a hex string.
 */
std::string AddressStr(const Object* obj) {
  std::ostringstream os;
  os << "0x" << std::hex << reinterpret_cast<uintptr_t>(obj);
  return os.str();
}

/*!
 * \brief Get the type key of an object as a std::string.
 */
std::string GetTypeKeyStr(const Object* obj) {
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(obj->type_index());
  return std::string(type_info->type_key.data, type_info->type_key.size);
}

/*!
 * \brief DFS-based repr printer.
 *
 * Algorithm:
 *   1. Start from the root value and recursively process via DFS.
 *   2. Track each object's state: NotVisited, InProgress, Done.
 *   3. On "Done" objects, return the cached repr string (handles DAGs).
 *   4. On "InProgress" objects, a cycle is detected — return "..." marker.
 *   5. On first visit, mark InProgress, process children, cache result, mark Done.
 *
 * Address display is controlled by the TVM_FFI_REPR_WITH_ADDR environment variable.
 */
class ReprPrinter {
 public:
  String Run(const Any& value) {
    const char* env = std::getenv("TVM_FFI_REPR_WITH_ADDR");
    show_addr_ = env != nullptr && std::string_view(env) == "1";
    return String(ReprOfAny(value));
  }

 private:
  enum class State : int8_t { kNotVisited = 0, kInProgress = 1, kDone = 2 };

  // ---------- Core DFS ----------

  std::string ReprOfAny(const Any& value) {
    int32_t ti = value.type_index();
    switch (ti) {
      case TypeIndex::kTVMFFINone:
        return "None";
      case TypeIndex::kTVMFFIBool:
        return value.cast<bool>() ? "True" : "False";
      case TypeIndex::kTVMFFIInt:
        return std::to_string(value.cast<int64_t>());
      case TypeIndex::kTVMFFIFloat: {
        std::ostringstream os;
        os << value.cast<double>();
        return os.str();
      }
      case TypeIndex::kTVMFFIDataType: {
        String s = DLDataTypeToString(value.cast<DLDataType>());
        return std::string(s.data(), s.size());
      }
      case TypeIndex::kTVMFFIDevice: {
        return DeviceToString(value.cast<DLDevice>());
      }
      default:
        break;
    }
    if (ti == TypeIndex::kTVMFFISmallStr) {
      String s = value.cast<String>();
      return "\"" + std::string(s.data(), s.size()) + "\"";
    }
    if (ti == TypeIndex::kTVMFFISmallBytes) {
      Bytes b = value.cast<Bytes>();
      return FormatBytes(b.data(), b.size());
    }
    if (ti < TypeIndex::kTVMFFIStaticObjectBegin) {
      // Other POD types
      return value.GetTypeKey();
    }
    // Object type — use DFS with state tracking
    const Object* obj = static_cast<const Object*>(value.as<Object>());
    if (obj == nullptr) return "None";
    auto it = state_.find(obj);
    if (it != state_.end()) {
      if (it->second == State::kDone) {
        // DAG: already fully processed, return cached repr
        return repr_cache_[obj];
      }
      if (it->second == State::kInProgress) {
        // Cycle detected
        return show_addr_ ? ("...@" + AddressStr(obj)) : "...";
      }
    }
    // First visit: mark in-progress, process, cache, mark done
    state_[obj] = State::kInProgress;
    std::string repr = ProcessObject(obj);
    repr_cache_[obj] = repr;
    state_[obj] = State::kDone;
    return repr;
  }

  // ---------- Processing ----------

  std::string ProcessObject(const Object* obj) {
    int32_t ti = obj->type_index();
    static reflection::TypeAttrColumn repr_column(reflection::type_attr::kRepr);
    AnyView custom_repr = repr_column[ti];
    std::string result;
    if (custom_repr != nullptr) {
      // Custom __ffi_repr__: call it with fn_repr callback
      Function repr_fn = custom_repr.cast<Function>();
      Function fn_repr = CreateFnRepr();
      String r = repr_fn(obj, fn_repr).cast<String>();
      result = std::string(r.data(), r.size());
    } else {
      // Generic reflection-based repr
      result = GenericRepr(obj);
    }
    // For containers: append address if env var is set
    if (show_addr_ && (ti == TypeIndex::kTVMFFIArray || ti == TypeIndex::kTVMFFIList ||
                       ti == TypeIndex::kTVMFFIMap || ti == TypeIndex::kTVMFFIDict ||
                       ti == TypeIndex::kTVMFFITensor)) {
      result += "@" + AddressStr(obj);
    }
    return result;
  }

  Function CreateFnRepr() {
    return Function::FromTyped(
        [this](AnyView value) -> String { return String(ReprOfAny(Any(value))); });
  }

  // ---------- Generic Repr ----------

  std::string GenericRepr(const Object* obj) {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(obj->type_index());
    std::string type_key = (type_info != nullptr)
                               ? std::string(type_info->type_key.data, type_info->type_key.size)
                               : GetTypeKeyStr(obj);
    std::string header = show_addr_ ? (type_key + "@" + AddressStr(obj)) : type_key;
    if (type_info == nullptr) return header;

    std::ostringstream fields;
    bool first = true;
    bool has_fields = false;
    reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* finfo) {
      if (finfo->flags & kTVMFFIFieldFlagBitMaskReprOff) return;
      has_fields = true;
      if (!first) fields << ", ";
      first = false;
      fields << std::string_view(finfo->name.data, finfo->name.size) << "=";
      reflection::FieldGetter getter(finfo);
      Any fv = getter(obj);
      fields << ReprOfAny(fv);
    });
    if (!has_fields) return header;
    return header + "(" + fields.str() + ")";
  }

  // ---------- Data members ----------
  std::unordered_map<const Object*, State> state_;
  std::unordered_map<const Object*, std::string> repr_cache_;
  bool show_addr_ = false;
};

// ---------- Built-in __ffi_repr__ functions ----------

String ReprString(const details::StringObj* obj, const Function& fn_repr) {
  std::ostringstream os;
  os << "\"" << std::string_view(obj->data, obj->size) << "\"";
  return String(os.str());
}

String ReprBytes(const details::BytesObj* obj, const Function& fn_repr) {
  return String(FormatBytes(obj->data, obj->size));
}

String ReprTensor(const TensorObj* obj, const Function& fn_repr) {
  std::ostringstream os;
  os << DLDataTypeToString(obj->dtype);
  os << "[";
  for (int i = 0; i < obj->ndim; ++i) {
    if (i > 0) os << ", ";
    os << obj->shape[i];
  }
  os << "]@" << DeviceToString(obj->device);
  return String(os.str());
}

String ReprShape(const ShapeObj* obj, const Function& fn_repr) {
  std::ostringstream os;
  os << "Shape(";
  for (size_t i = 0; i < obj->size; ++i) {
    if (i > 0) os << ", ";
    os << obj->data[i];
  }
  os << ")";
  return String(os.str());
}

String ReprArray(const ArrayObj* obj, const Function& fn_repr) {
  std::ostringstream os;
  os << "(";
  size_t count = 0;
  for (const Any& elem : *obj) {
    if (count > 0) os << ", ";
    String s = fn_repr(elem).cast<String>();
    os << std::string_view(s.data(), s.size());
    ++count;
  }
  if (count == 1) os << ",";
  os << ")";
  return String(os.str());
}

String ReprList(const ListObj* obj, const Function& fn_repr) {
  std::ostringstream os;
  os << "[";
  bool first = true;
  for (const Any& elem : *obj) {
    if (!first) os << ", ";
    first = false;
    String s = fn_repr(elem).cast<String>();
    os << std::string_view(s.data(), s.size());
  }
  os << "]";
  return String(os.str());
}

String ReprMapBase(const MapBaseObj* obj, const Function& fn_repr) {
  std::ostringstream os;
  os << "{";
  bool first = true;
  for (const auto& [k, v] : *obj) {
    if (!first) os << ", ";
    first = false;
    String ks = fn_repr(k).cast<String>();
    String vs = fn_repr(v).cast<String>();
    os << std::string_view(ks.data(), ks.size()) << ": " << std::string_view(vs.data(), vs.size());
  }
  os << "}";
  return String(os.str());
}

String ReprDict(const DictObj* obj, const Function& fn_repr) {
  return ReprMapBase(static_cast<const MapBaseObj*>(obj), fn_repr);
}

String ReprMap(const MapObj* obj, const Function& fn_repr) {
  return ReprMapBase(static_cast<const MapBaseObj*>(obj), fn_repr);
}

/*!
 * \brief Register a built-in __ffi_repr__ function for a given type index.
 */
template <typename Func>
void RegisterBuiltinRepr(int32_t type_index, Func&& func) {
  TVMFFIByteArray attr_name = {reflection::type_attr::kRepr,
                               std::char_traits<char>::length(reflection::type_attr::kRepr)};
  Function ffi_func = Function::FromTyped(std::forward<Func>(func), std::string("__ffi_repr__"));
  TVMFFIAny attr_value = AnyView(ffi_func).CopyToTVMFFIAny();
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index, &attr_name, &attr_value));
}

}  // namespace

String ReprPrint(const Any& value) {
  ReprPrinter printer;
  return printer.Run(value);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Ensure type attribute columns exist
  refl::EnsureTypeAttrColumn(refl::type_attr::kRepr);
  // Register built-in repr functions
  RegisterBuiltinRepr(TypeIndex::kTVMFFIStr, ReprString);
  RegisterBuiltinRepr(TypeIndex::kTVMFFIBytes, ReprBytes);
  RegisterBuiltinRepr(TypeIndex::kTVMFFITensor, ReprTensor);
  RegisterBuiltinRepr(TypeIndex::kTVMFFIShape, ReprShape);
  RegisterBuiltinRepr(TypeIndex::kTVMFFIArray, ReprArray);
  RegisterBuiltinRepr(TypeIndex::kTVMFFIList, ReprList);
  RegisterBuiltinRepr(TypeIndex::kTVMFFIMap, ReprMap);
  RegisterBuiltinRepr(TypeIndex::kTVMFFIDict, ReprDict);
  // Register global function
  refl::GlobalDef().def("ffi.ReprPrint",
                        [](const Any& value) -> String { return ReprPrint(value); });
}

}  // namespace ffi
}  // namespace tvm
