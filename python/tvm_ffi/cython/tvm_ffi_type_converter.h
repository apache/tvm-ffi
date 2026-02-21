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
 * \file tvm_ffi_type_converter.h
 * \brief C++ based type converter that validates/converts FFI values to match a target type.
 */
#ifndef TVM_FFI_TYPE_CONVERTER_H_
#define TVM_FFI_TYPE_CONVERTER_H_

#include <tvm/ffi/c_api.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

///--------------------------------------------------------------------------------
/// Converter node kind enum
///--------------------------------------------------------------------------------
enum TVMFFIPyTypeConverterKind {
  kTCKAny = 0,
  kTCKNone = 1,
  kTCKInt = 2,
  kTCKBool = 3,
  kTCKFloat = 4,
  kTCKDataType = 5,
  kTCKDevice = 6,
  kTCKOpaquePtr = 7,
  kTCKString = 8,
  kTCKBytes = 9,
  kTCKTensor = 10,
  kTCKFunction = 11,
  kTCKObject = 12,
  kTCKOpaquePyObject = 13,
  kTCKOptional = 14,
  kTCKUnion = 15,
  kTCKArray = 16,
  kTCKList = 17,
  kTCKMap = 18,
  kTCKTuple = 19,
};

///--------------------------------------------------------------------------------
/// Forward declarations
///--------------------------------------------------------------------------------
struct TVMFFIPyTypeConverterNode;
struct TVMFFIPyTypeConverter;

///--------------------------------------------------------------------------------
/// Helper: get a human-readable type name from a TVMFFIAny value
///--------------------------------------------------------------------------------
static inline const char* TVMFFIPyGetTypeName(const TVMFFIAny* input) {
  switch (input->type_index) {
    case kTVMFFINone:
      return "None";
    case kTVMFFIInt:
      return "int";
    case kTVMFFIBool:
      return "bool";
    case kTVMFFIFloat:
      return "float";
    case kTVMFFIOpaquePtr:
      return "ctypes.c_void_p";
    case kTVMFFIDataType:
      return "dtype";
    case kTVMFFIDevice:
      return "Device";
    case kTVMFFIDLTensorPtr:
      return "Tensor";
    case kTVMFFISmallStr:
      return "str";
    case kTVMFFISmallBytes:
      return "bytes";
    case kTVMFFIObjectRValueRef:
      return "ObjectRValueRef";
    default:
      break;
  }
  if (input->type_index >= kTVMFFIStaticObjectBegin) {
    switch (input->type_index) {
      case kTVMFFIStr:
        return "str";
      case kTVMFFIBytes:
        return "bytes";
      case kTVMFFIError:
        return "Error";
      case kTVMFFIFunction:
        return "Callable";
      case kTVMFFIShape:
        return "Shape";
      case kTVMFFITensor:
        return "Tensor";
      case kTVMFFIArray:
        return "list";
      case kTVMFFIMap:
        return "dict";
      case kTVMFFIModule:
        return "Module";
      case kTVMFFIOpaquePyObject:
        return "OpaquePyObject";
      case kTVMFFIList:
        return "list";
      default: {
        // try to get type key from type info
        const TVMFFITypeInfo* tinfo = TVMFFIGetTypeInfo(input->type_index);
        if (tinfo != nullptr) {
          // NOTE: this is a pointer to static storage, safe to return
          return tinfo->type_key.data;
        }
        return "<unknown>";
      }
    }
  }
  return "<unknown>";
}

///--------------------------------------------------------------------------------
/// Helper: Set a TypeError via FFI error mechanism (no GIL needed)
///--------------------------------------------------------------------------------
static inline int TVMFFIPySetTypeError(const char* message, size_t message_len) {
  TVMFFIByteArray kind_ba;
  kind_ba.data = "TypeError";
  kind_ba.size = 9;
  TVMFFIByteArray msg_ba;
  msg_ba.data = message;
  msg_ba.size = message_len;
  TVMFFIByteArray bt_ba;
  bt_ba.data = "";
  bt_ba.size = 0;
  TVMFFIObjectHandle err_obj = nullptr;
  if (TVMFFIErrorCreate(&kind_ba, &msg_ba, &bt_ba, &err_obj) != 0) {
    return -1;
  }
  TVMFFIErrorSetRaised(err_obj);
  return -1;
}

///--------------------------------------------------------------------------------
/// Helper: copy an Any with incref for objects
///--------------------------------------------------------------------------------
static inline void TVMFFIPyAnyCopyWithIncRef(const TVMFFIAny* src, TVMFFIAny* dst) {
  dst->type_index = src->type_index;
  dst->zero_padding = src->zero_padding;
  dst->v_int64 = src->v_int64;
  if (src->type_index >= kTVMFFIStaticObjectBegin && src->v_ptr != nullptr) {
    TVMFFIObjectIncRef(src->v_ptr);
  }
}

///--------------------------------------------------------------------------------
/// Helper: Check if actual_index is a subtype of target_index using type ancestors
///--------------------------------------------------------------------------------
static inline int TVMFFIPyCheckTypeHierarchy(int32_t actual_index, int32_t target_index) {
  if (actual_index == target_index) return 1;
  const TVMFFITypeInfo* tinfo = TVMFFIGetTypeInfo(actual_index);
  if (tinfo == nullptr) return 0;
  // Walk ancestors: type_ancestors[0..type_depth-1] are the ancestors from root to parent
  for (int32_t i = 0; i < tinfo->type_depth; ++i) {
    if (tinfo->type_ancestors[i]->type_index == target_index) {
      return 1;
    }
  }
  return 0;
}

///--------------------------------------------------------------------------------
/// TVMFFIPyTypeConverterNode: a node in the type converter tree
///--------------------------------------------------------------------------------
struct TVMFFIPyTypeConverterNode {
  TVMFFIPyTypeConverterKind kind;
  int32_t type_index;  // for kTCKObject: required type index
  std::string type_repr;
  std::vector<TVMFFIPyTypeConverterNode*> children;

  ~TVMFFIPyTypeConverterNode() {
    for (auto* child : children) {
      delete child;
    }
  }

  /*!
   * \brief Check if the input value strictly matches this type (zero-copy eligible).
   * \param input The input value to check
   * \return 1 if exact match, 0 otherwise
   */
  int CheckStrict(const TVMFFIAny* input) const {
    switch (kind) {
      case kTCKAny:
        return 1;
      case kTCKNone:
        return input->type_index == kTVMFFINone;
      case kTCKInt:
        return input->type_index == kTVMFFIInt;
      case kTCKBool:
        return input->type_index == kTVMFFIBool;
      case kTCKFloat:
        return input->type_index == kTVMFFIFloat;
      case kTCKDataType:
        return input->type_index == kTVMFFIDataType;
      case kTCKDevice:
        return input->type_index == kTVMFFIDevice;
      case kTCKOpaquePtr:
        return input->type_index == kTVMFFIOpaquePtr;
      case kTCKString:
        return input->type_index == kTVMFFIStr || input->type_index == kTVMFFISmallStr;
      case kTCKBytes:
        return input->type_index == kTVMFFIBytes || input->type_index == kTVMFFISmallBytes;
      case kTCKTensor:
        return input->type_index == kTVMFFITensor;
      case kTCKFunction:
        return input->type_index == kTVMFFIFunction;
      case kTCKOpaquePyObject:
        return input->type_index == kTVMFFIOpaquePyObject;
      case kTCKObject:
        if (input->type_index < kTVMFFIStaticObjectBegin) return 0;
        return TVMFFIPyCheckTypeHierarchy(input->type_index, type_index);
      case kTCKOptional:
        if (input->type_index == kTVMFFINone) return 1;
        return children[0]->CheckStrict(input);
      case kTCKUnion:
        for (size_t i = 0; i < children.size(); ++i) {
          if (children[i]->CheckStrict(input)) return 1;
        }
        return 0;
      case kTCKArray:
        return CheckStrictSeq(input, kTVMFFIArray);
      case kTCKList:
        return CheckStrictSeq(input, kTVMFFIList);
      case kTCKMap:
        return CheckStrictMap(input);
      case kTCKTuple:
        return CheckStrictTuple(input);
      default:
        return 0;
    }
  }

  /*!
   * \brief Convert the input value to match this type.
   * \param input The input value to convert
   * \param output The output value (caller must decref if object)
   * \return 0 on success, -1 on error (FFI error set)
   */
  int Convert(const TVMFFIAny* input, TVMFFIAny* output,
              const TVMFFIPyTypeConverter* converter) const;

 private:
  int CheckStrictSeq(const TVMFFIAny* input, int32_t expected_type_index) const {
    if (input->type_index != expected_type_index) return 0;
    if (children.empty()) return 1;
    // Access elements via known memory layout
    TVMFFIObjectHandle handle = static_cast<TVMFFIObjectHandle>(input->v_ptr);
    TVMFFISeqCell* cell =
        reinterpret_cast<TVMFFISeqCell*>(reinterpret_cast<char*>(handle) + sizeof(TVMFFIObject));
    TVMFFIAny* elements = static_cast<TVMFFIAny*>(cell->data);
    int64_t size = cell->size;
    const TVMFFIPyTypeConverterNode* elem_checker = children[0];
    for (int64_t i = 0; i < size; ++i) {
      if (!elem_checker->CheckStrict(&elements[i])) return 0;
    }
    return 1;
  }

  int CheckStrictMap(const TVMFFIAny* input) const {
    if (input->type_index != kTVMFFIMap) return 0;
    // Map iteration requires calling functions, so we cannot do it purely in C++
    // without the GIL. For strict check, we return 0 to force slow-path.
    // However, we can try: if children are both kTCKAny, return 1.
    if (children.size() == 2 && children[0]->kind == kTCKAny && children[1]->kind == kTCKAny) {
      return 1;
    }
    // Otherwise, we need to iterate. We'll use the converter's map iteration.
    return 0;
  }

  int CheckStrictTuple(const TVMFFIAny* input) const {
    if (input->type_index != kTVMFFIArray) return 0;
    TVMFFIObjectHandle handle = static_cast<TVMFFIObjectHandle>(input->v_ptr);
    TVMFFISeqCell* cell =
        reinterpret_cast<TVMFFISeqCell*>(reinterpret_cast<char*>(handle) + sizeof(TVMFFIObject));
    TVMFFIAny* elements = static_cast<TVMFFIAny*>(cell->data);
    int64_t size = cell->size;
    // Check size matches number of children
    if (size != static_cast<int64_t>(children.size())) return 0;
    for (int64_t i = 0; i < size; ++i) {
      if (!children[i]->CheckStrict(&elements[i])) return 0;
    }
    return 1;
  }

  int ConvertPOD(const TVMFFIAny* input, TVMFFIAny* output) const;
  int ConvertString(const TVMFFIAny* input, TVMFFIAny* output) const;
  int ConvertBytes(const TVMFFIAny* input, TVMFFIAny* output) const;
  int ConvertTensor(const TVMFFIAny* input, TVMFFIAny* output) const;
  int ConvertObject(const TVMFFIAny* input, TVMFFIAny* output) const;
  int ConvertOptional(const TVMFFIAny* input, TVMFFIAny* output,
                      const TVMFFIPyTypeConverter* converter) const;
  int ConvertUnion(const TVMFFIAny* input, TVMFFIAny* output,
                   const TVMFFIPyTypeConverter* converter) const;
  int ConvertSeq(const TVMFFIAny* input, TVMFFIAny* output, const TVMFFIPyTypeConverter* converter,
                 int32_t seq_type_index) const;
  int ConvertMap(const TVMFFIAny* input, TVMFFIAny* output,
                 const TVMFFIPyTypeConverter* converter) const;
  int ConvertTuple(const TVMFFIAny* input, TVMFFIAny* output,
                   const TVMFFIPyTypeConverter* converter) const;

  int SetTypeError(const TVMFFIAny* input) const {
    std::string msg = "expected " + type_repr + ", got " + TVMFFIPyGetTypeName(input);
    return TVMFFIPySetTypeError(msg.c_str(), msg.size());
  }

  int SetTypeErrorWithContext(const TVMFFIAny* input, const char* context) const {
    std::string msg =
        "expected " + type_repr + ", but " + context + " has type " + TVMFFIPyGetTypeName(input);
    return TVMFFIPySetTypeError(msg.c_str(), msg.size());
  }
};

///--------------------------------------------------------------------------------
/// TVMFFIPyTypeConverter: top-level converter owning the tree and cached handles
///--------------------------------------------------------------------------------
struct TVMFFIPyTypeConverter {
  TVMFFIPyTypeConverterNode* root;
  TVMFFIObjectHandle array_ctor;
  TVMFFIObjectHandle list_ctor;
  TVMFFIObjectHandle map_ctor;
  TVMFFIObjectHandle map_iter_func;
  TVMFFIObjectHandle map_size_func;

  ~TVMFFIPyTypeConverter() {
    delete root;
    if (array_ctor) TVMFFIObjectDecRef(array_ctor);
    if (list_ctor) TVMFFIObjectDecRef(list_ctor);
    if (map_ctor) TVMFFIObjectDecRef(map_ctor);
    if (map_iter_func) TVMFFIObjectDecRef(map_iter_func);
    if (map_size_func) TVMFFIObjectDecRef(map_size_func);
  }
};

///--------------------------------------------------------------------------------
/// Implementation of Convert methods
///--------------------------------------------------------------------------------
inline int TVMFFIPyTypeConverterNode::ConvertPOD(const TVMFFIAny* input, TVMFFIAny* output) const {
  switch (kind) {
    case kTCKInt:
      if (input->type_index == kTVMFFIInt) {
        *output = *input;
        return 0;
      }
      if (input->type_index == kTVMFFIBool) {
        output->type_index = kTVMFFIInt;
        output->zero_padding = 0;
        output->v_int64 = input->v_int64;
        return 0;
      }
      return SetTypeError(input);
    case kTCKFloat:
      if (input->type_index == kTVMFFIFloat) {
        *output = *input;
        return 0;
      }
      if (input->type_index == kTVMFFIInt) {
        output->type_index = kTVMFFIFloat;
        output->zero_padding = 0;
        output->v_float64 = static_cast<double>(input->v_int64);
        return 0;
      }
      if (input->type_index == kTVMFFIBool) {
        output->type_index = kTVMFFIFloat;
        output->zero_padding = 0;
        output->v_float64 = static_cast<double>(input->v_int64);
        return 0;
      }
      return SetTypeError(input);
    case kTCKBool:
      if (input->type_index == kTVMFFIBool) {
        *output = *input;
        return 0;
      }
      return SetTypeError(input);
    case kTCKNone:
      if (input->type_index == kTVMFFINone) {
        *output = *input;
        return 0;
      }
      return SetTypeError(input);
    case kTCKDataType:
      if (input->type_index == kTVMFFIDataType) {
        *output = *input;
        return 0;
      }
      return SetTypeError(input);
    case kTCKDevice:
      if (input->type_index == kTVMFFIDevice) {
        *output = *input;
        return 0;
      }
      return SetTypeError(input);
    case kTCKOpaquePtr:
      if (input->type_index == kTVMFFIOpaquePtr) {
        *output = *input;
        return 0;
      }
      return SetTypeError(input);
    default:
      return SetTypeError(input);
  }
}

inline int TVMFFIPyTypeConverterNode::ConvertString(const TVMFFIAny* input,
                                                    TVMFFIAny* output) const {
  if (input->type_index == kTVMFFIStr || input->type_index == kTVMFFISmallStr) {
    TVMFFIPyAnyCopyWithIncRef(input, output);
    return 0;
  }
  return SetTypeError(input);
}

inline int TVMFFIPyTypeConverterNode::ConvertBytes(const TVMFFIAny* input,
                                                   TVMFFIAny* output) const {
  if (input->type_index == kTVMFFIBytes || input->type_index == kTVMFFISmallBytes) {
    TVMFFIPyAnyCopyWithIncRef(input, output);
    return 0;
  }
  return SetTypeError(input);
}

inline int TVMFFIPyTypeConverterNode::ConvertTensor(const TVMFFIAny* input,
                                                    TVMFFIAny* output) const {
  if (input->type_index == kTVMFFITensor || input->type_index == kTVMFFIDLTensorPtr) {
    TVMFFIPyAnyCopyWithIncRef(input, output);
    return 0;
  }
  return SetTypeError(input);
}

inline int TVMFFIPyTypeConverterNode::ConvertObject(const TVMFFIAny* input,
                                                    TVMFFIAny* output) const {
  if (input->type_index < kTVMFFIStaticObjectBegin) {
    return SetTypeError(input);
  }
  if (TVMFFIPyCheckTypeHierarchy(input->type_index, type_index)) {
    TVMFFIPyAnyCopyWithIncRef(input, output);
    return 0;
  }
  return SetTypeError(input);
}

inline int TVMFFIPyTypeConverterNode::ConvertOptional(
    const TVMFFIAny* input, TVMFFIAny* output, const TVMFFIPyTypeConverter* converter) const {
  if (input->type_index == kTVMFFINone) {
    output->type_index = kTVMFFINone;
    output->zero_padding = 0;
    output->v_int64 = 0;
    return 0;
  }
  return children[0]->Convert(input, output, converter);
}

inline int TVMFFIPyTypeConverterNode::ConvertUnion(const TVMFFIAny* input, TVMFFIAny* output,
                                                   const TVMFFIPyTypeConverter* converter) const {
  for (size_t i = 0; i < children.size(); ++i) {
    // First try strict check
    if (children[i]->CheckStrict(input)) {
      return children[i]->Convert(input, output, converter);
    }
  }
  // Then try conversion (some types allow coercion)
  for (size_t i = 0; i < children.size(); ++i) {
    // Try to convert; if it fails, clear the error and try next
    int ret = children[i]->Convert(input, output, converter);
    if (ret == 0) return 0;
    // Clear the error that was set by the failed Convert
    TVMFFIObjectHandle err_obj = nullptr;
    TVMFFIErrorMoveFromRaised(&err_obj);
    if (err_obj != nullptr) {
      TVMFFIObjectDecRef(err_obj);
    }
  }
  // None of the union alternatives matched
  std::string msg = "expected " + type_repr + ", got " + TVMFFIPyGetTypeName(input);
  return TVMFFIPySetTypeError(msg.c_str(), msg.size());
}

inline int TVMFFIPyTypeConverterNode::ConvertSeq(const TVMFFIAny* input, TVMFFIAny* output,
                                                 const TVMFFIPyTypeConverter* converter,
                                                 int32_t seq_type_index) const {
  int32_t actual_type = input->type_index;
  // Accept both Array and List as input for both Array and List target types
  if (actual_type != kTVMFFIArray && actual_type != kTVMFFIList) {
    return SetTypeError(input);
  }
  // Fast path: check if all elements already match
  if (actual_type == seq_type_index && CheckStrict(input)) {
    TVMFFIPyAnyCopyWithIncRef(input, output);
    return 0;
  }
  // Slow path: convert element by element
  TVMFFIObjectHandle handle = static_cast<TVMFFIObjectHandle>(input->v_ptr);
  TVMFFISeqCell* cell =
      reinterpret_cast<TVMFFISeqCell*>(reinterpret_cast<char*>(handle) + sizeof(TVMFFIObject));
  TVMFFIAny* elements = static_cast<TVMFFIAny*>(cell->data);
  int64_t size = cell->size;

  // Build converted elements array
  std::vector<TVMFFIAny> converted(size);
  const TVMFFIPyTypeConverterNode* elem_converter = children.empty() ? nullptr : children[0];

  for (int64_t i = 0; i < size; ++i) {
    if (elem_converter == nullptr || elem_converter->kind == kTCKAny) {
      TVMFFIPyAnyCopyWithIncRef(&elements[i], &converted[i]);
    } else {
      int ret = elem_converter->Convert(&elements[i], &converted[i], converter);
      if (ret != 0) {
        // Clean up already converted elements
        for (int64_t j = 0; j < i; ++j) {
          if (converted[j].type_index >= kTVMFFIStaticObjectBegin && converted[j].v_ptr) {
            TVMFFIObjectDecRef(converted[j].v_ptr);
          }
        }
        // Enhance the error message with element index context
        TVMFFIObjectHandle err_obj = nullptr;
        TVMFFIErrorMoveFromRaised(&err_obj);
        if (err_obj != nullptr) {
          TVMFFIObjectDecRef(err_obj);
        }
        char idx_buf[64];
        snprintf(idx_buf, sizeof(idx_buf), "element at index %lld", (long long)i);
        return children[0]->SetTypeErrorWithContext(&elements[i], idx_buf);
      }
    }
  }

  // Call the constructor to create a new container
  TVMFFIObjectHandle ctor =
      (seq_type_index == kTVMFFIArray) ? converter->array_ctor : converter->list_ctor;
  int ret = TVMFFIFunctionCall(ctor, converted.data(), static_cast<int32_t>(size), output);

  // Clean up converted elements
  for (int64_t i = 0; i < size; ++i) {
    if (converted[i].type_index >= kTVMFFIStaticObjectBegin && converted[i].v_ptr) {
      TVMFFIObjectDecRef(converted[i].v_ptr);
    }
  }
  return ret;
}

inline int TVMFFIPyTypeConverterNode::ConvertMap(const TVMFFIAny* input, TVMFFIAny* output,
                                                 const TVMFFIPyTypeConverter* converter) const {
  if (input->type_index != kTVMFFIMap) {
    return SetTypeError(input);
  }

  // Get map size
  TVMFFIAny size_result;
  size_result.type_index = kTVMFFINone;
  size_result.v_int64 = 0;
  TVMFFIAny map_arg;
  map_arg.type_index = input->type_index;
  map_arg.zero_padding = 0;
  map_arg.v_ptr = input->v_ptr;
  if (TVMFFIFunctionCall(converter->map_size_func, &map_arg, 1, &size_result) != 0) {
    return -1;
  }
  int64_t map_size = size_result.v_int64;

  // Get iterator functor
  TVMFFIAny iter_result;
  iter_result.type_index = kTVMFFINone;
  iter_result.v_int64 = 0;
  if (TVMFFIFunctionCall(converter->map_iter_func, &map_arg, 1, &iter_result) != 0) {
    return -1;
  }
  TVMFFIObjectHandle iter_func_handle = static_cast<TVMFFIObjectHandle>(iter_result.v_ptr);

  const TVMFFIPyTypeConverterNode* key_converter = (children.size() >= 1) ? children[0] : nullptr;
  const TVMFFIPyTypeConverterNode* val_converter = (children.size() >= 2) ? children[1] : nullptr;

  // Fast path: if input is already strictly matching, just return with incref
  bool all_strict = true;
  if (key_converter && key_converter->kind != kTCKAny && val_converter &&
      val_converter->kind != kTCKAny) {
    // We need to iterate to check. We'll do it in the conversion loop.
    all_strict = false;
  } else if ((key_converter == nullptr || key_converter->kind == kTCKAny) &&
             (val_converter == nullptr || val_converter->kind == kTCKAny)) {
    // Both are Any, fast path
    TVMFFIObjectDecRef(iter_func_handle);
    TVMFFIPyAnyCopyWithIncRef(input, output);
    return 0;
  } else {
    all_strict = false;
  }

  // Build key-value pairs for constructor
  std::vector<TVMFFIAny> kv_pairs(map_size * 2);
  int64_t kv_idx = 0;

  for (int64_t i = 0; i < map_size; ++i) {
    // Get current key (command=0)
    TVMFFIAny cmd_arg;
    cmd_arg.type_index = kTVMFFIInt;
    cmd_arg.zero_padding = 0;
    cmd_arg.v_int64 = 0;
    TVMFFIAny key_result;
    key_result.type_index = kTVMFFINone;
    key_result.v_int64 = 0;
    if (TVMFFIFunctionCall(iter_func_handle, &cmd_arg, 1, &key_result) != 0) {
      goto cleanup_error;
    }

    // Get current value (command=1)
    cmd_arg.v_int64 = 1;
    TVMFFIAny val_result;
    val_result.type_index = kTVMFFINone;
    val_result.v_int64 = 0;
    if (TVMFFIFunctionCall(iter_func_handle, &cmd_arg, 1, &val_result) != 0) {
      // decref key_result if needed
      if (key_result.type_index >= kTVMFFIStaticObjectBegin && key_result.v_ptr) {
        TVMFFIObjectDecRef(key_result.v_ptr);
      }
      goto cleanup_error;
    }

    // Convert key
    if (key_converter && key_converter->kind != kTCKAny) {
      int ret = key_converter->Convert(&key_result, &kv_pairs[kv_idx], converter);
      // decref owned key_result
      if (key_result.type_index >= kTVMFFIStaticObjectBegin && key_result.v_ptr) {
        TVMFFIObjectDecRef(key_result.v_ptr);
      }
      if (ret != 0) {
        // decref owned val_result
        if (val_result.type_index >= kTVMFFIStaticObjectBegin && val_result.v_ptr) {
          TVMFFIObjectDecRef(val_result.v_ptr);
        }
        // Improve error with key context
        TVMFFIObjectHandle err_obj = nullptr;
        TVMFFIErrorMoveFromRaised(&err_obj);
        if (err_obj) TVMFFIObjectDecRef(err_obj);
        char ctx_buf[128];
        snprintf(ctx_buf, sizeof(ctx_buf), "key at index %lld", (long long)i);
        key_converter->SetTypeErrorWithContext(&key_result, ctx_buf);
        goto cleanup_error;
      }
    } else {
      // key_result is already owned (returned from FunctionCall)
      kv_pairs[kv_idx] = key_result;
    }
    kv_idx++;

    // Convert value
    if (val_converter && val_converter->kind != kTCKAny) {
      int ret = val_converter->Convert(&val_result, &kv_pairs[kv_idx], converter);
      // decref owned val_result
      if (val_result.type_index >= kTVMFFIStaticObjectBegin && val_result.v_ptr) {
        TVMFFIObjectDecRef(val_result.v_ptr);
      }
      if (ret != 0) {
        // Improve error with value context
        TVMFFIObjectHandle err_obj = nullptr;
        TVMFFIErrorMoveFromRaised(&err_obj);
        if (err_obj) TVMFFIObjectDecRef(err_obj);
        // Use the key for context - we already converted it
        char ctx_buf[128];
        snprintf(ctx_buf, sizeof(ctx_buf), "value at index %lld", (long long)i);
        val_converter->SetTypeErrorWithContext(&val_result, ctx_buf);
        // Decref the converted key we just added
        if (kv_pairs[kv_idx - 1].type_index >= kTVMFFIStaticObjectBegin &&
            kv_pairs[kv_idx - 1].v_ptr) {
          TVMFFIObjectDecRef(kv_pairs[kv_idx - 1].v_ptr);
        }
        kv_idx--;  // undo the key increment
        goto cleanup_error;
      }
    } else {
      // val_result is already owned
      kv_pairs[kv_idx] = val_result;
    }
    kv_idx++;

    // Advance iterator (command=2)
    if (i < map_size - 1) {
      cmd_arg.v_int64 = 2;
      TVMFFIAny advance_result;
      advance_result.type_index = kTVMFFINone;
      advance_result.v_int64 = 0;
      if (TVMFFIFunctionCall(iter_func_handle, &cmd_arg, 1, &advance_result) != 0) {
        goto cleanup_error;
      }
    }
  }

  {
    // Call map constructor
    int ret = TVMFFIFunctionCall(converter->map_ctor, kv_pairs.data(), static_cast<int32_t>(kv_idx),
                                 output);
    // Cleanup kv_pairs
    for (int64_t j = 0; j < kv_idx; ++j) {
      if (kv_pairs[j].type_index >= kTVMFFIStaticObjectBegin && kv_pairs[j].v_ptr) {
        TVMFFIObjectDecRef(kv_pairs[j].v_ptr);
      }
    }
    TVMFFIObjectDecRef(iter_func_handle);
    return ret;
  }

cleanup_error:
  // Cleanup already added kv_pairs
  for (int64_t j = 0; j < kv_idx; ++j) {
    if (kv_pairs[j].type_index >= kTVMFFIStaticObjectBegin && kv_pairs[j].v_ptr) {
      TVMFFIObjectDecRef(kv_pairs[j].v_ptr);
    }
  }
  TVMFFIObjectDecRef(iter_func_handle);
  return -1;
}

inline int TVMFFIPyTypeConverterNode::ConvertTuple(const TVMFFIAny* input, TVMFFIAny* output,
                                                   const TVMFFIPyTypeConverter* converter) const {
  if (input->type_index != kTVMFFIArray) {
    return SetTypeError(input);
  }
  TVMFFIObjectHandle handle = static_cast<TVMFFIObjectHandle>(input->v_ptr);
  TVMFFISeqCell* cell =
      reinterpret_cast<TVMFFISeqCell*>(reinterpret_cast<char*>(handle) + sizeof(TVMFFIObject));
  TVMFFIAny* elements = static_cast<TVMFFIAny*>(cell->data);
  int64_t size = cell->size;

  // Check size match
  int64_t expected_size = static_cast<int64_t>(children.size());
  if (size != expected_size) {
    char msg_buf[256];
    snprintf(msg_buf, sizeof(msg_buf), "expected tuple of size %lld, got tuple of size %lld",
             (long long)expected_size, (long long)size);
    return TVMFFIPySetTypeError(msg_buf, strlen(msg_buf));
  }

  // Fast path: check strict
  if (CheckStrict(input)) {
    TVMFFIPyAnyCopyWithIncRef(input, output);
    return 0;
  }

  // Slow path: convert per-element
  std::vector<TVMFFIAny> converted(size);
  for (int64_t i = 0; i < size; ++i) {
    int ret = children[i]->Convert(&elements[i], &converted[i], converter);
    if (ret != 0) {
      for (int64_t j = 0; j < i; ++j) {
        if (converted[j].type_index >= kTVMFFIStaticObjectBegin && converted[j].v_ptr) {
          TVMFFIObjectDecRef(converted[j].v_ptr);
        }
      }
      // Enhance error with index
      TVMFFIObjectHandle err_obj = nullptr;
      TVMFFIErrorMoveFromRaised(&err_obj);
      if (err_obj) TVMFFIObjectDecRef(err_obj);
      char idx_buf[64];
      snprintf(idx_buf, sizeof(idx_buf), "element at index %lld", (long long)i);
      return children[i]->SetTypeErrorWithContext(&elements[i], idx_buf);
    }
  }

  // Build Array
  int ret = TVMFFIFunctionCall(converter->array_ctor, converted.data(), static_cast<int32_t>(size),
                               output);
  for (int64_t i = 0; i < size; ++i) {
    if (converted[i].type_index >= kTVMFFIStaticObjectBegin && converted[i].v_ptr) {
      TVMFFIObjectDecRef(converted[i].v_ptr);
    }
  }
  return ret;
}

inline int TVMFFIPyTypeConverterNode::Convert(const TVMFFIAny* input, TVMFFIAny* output,
                                              const TVMFFIPyTypeConverter* converter) const {
  switch (kind) {
    case kTCKAny:
      TVMFFIPyAnyCopyWithIncRef(input, output);
      return 0;
    case kTCKNone:
    case kTCKInt:
    case kTCKBool:
    case kTCKFloat:
    case kTCKDataType:
    case kTCKDevice:
    case kTCKOpaquePtr:
      return ConvertPOD(input, output);
    case kTCKString:
      return ConvertString(input, output);
    case kTCKBytes:
      return ConvertBytes(input, output);
    case kTCKTensor:
      return ConvertTensor(input, output);
    case kTCKFunction:
      if (input->type_index == kTVMFFIFunction) {
        TVMFFIPyAnyCopyWithIncRef(input, output);
        return 0;
      }
      return SetTypeError(input);
    case kTCKOpaquePyObject:
      if (input->type_index == kTVMFFIOpaquePyObject) {
        TVMFFIPyAnyCopyWithIncRef(input, output);
        return 0;
      }
      return SetTypeError(input);
    case kTCKObject:
      return ConvertObject(input, output);
    case kTCKOptional:
      return ConvertOptional(input, output, converter);
    case kTCKUnion:
      return ConvertUnion(input, output, converter);
    case kTCKArray:
      return ConvertSeq(input, output, converter, kTVMFFIArray);
    case kTCKList:
      return ConvertSeq(input, output, converter, kTVMFFIList);
    case kTCKMap:
      return ConvertMap(input, output, converter);
    case kTCKTuple:
      return ConvertTuple(input, output, converter);
    default:
      return SetTypeError(input);
  }
}

///--------------------------------------------------------------------------------
/// C-style factory functions (for Cython to call)
///--------------------------------------------------------------------------------

/*!
 * \brief Create a converter node.
 * \param kind The kind of the node
 * \param type_index The type index (for kTCKObject)
 * \param type_repr String representation for error messages
 * \param type_repr_len Length of type_repr
 * \param children Array of child node pointers (ownership transferred)
 * \param num_children Number of children
 * \return The created node (caller owns)
 */
inline TVMFFIPyTypeConverterNode* TVMFFIPyTypeConverterNodeCreate(
    int kind, int32_t type_index, const char* type_repr, int type_repr_len,
    TVMFFIPyTypeConverterNode** children, int num_children) {
  auto* node = new TVMFFIPyTypeConverterNode();
  node->kind = static_cast<TVMFFIPyTypeConverterKind>(kind);
  node->type_index = type_index;
  node->type_repr = std::string(type_repr, type_repr_len);
  node->children.assign(children, children + num_children);
  return node;
}

/*!
 * \brief Delete a converter node (for error cleanup before converter is created).
 */
inline void TVMFFIPyTypeConverterNodeDelete(TVMFFIPyTypeConverterNode* node) { delete node; }

/*!
 * \brief Create the top-level converter.
 * \param root Root converter node (ownership transferred)
 * \param array_ctor Handle to ffi.Array constructor (must be incref'd by caller)
 * \param list_ctor Handle to ffi.List constructor (must be incref'd by caller)
 * \param map_ctor Handle to ffi.Map constructor (must be incref'd by caller)
 * \param map_iter_func Handle to ffi.MapForwardIterFunctor (must be incref'd by caller)
 * \param map_size_func Handle to ffi.MapSize (must be incref'd by caller)
 * \return The converter (caller owns via Deleter)
 */
inline TVMFFIPyTypeConverter* TVMFFIPyTypeConverterCreate(TVMFFIPyTypeConverterNode* root,
                                                          TVMFFIObjectHandle array_ctor,
                                                          TVMFFIObjectHandle list_ctor,
                                                          TVMFFIObjectHandle map_ctor,
                                                          TVMFFIObjectHandle map_iter_func,
                                                          TVMFFIObjectHandle map_size_func) {
  auto* conv = new TVMFFIPyTypeConverter();
  conv->root = root;
  conv->array_ctor = array_ctor;
  conv->list_ctor = list_ctor;
  conv->map_ctor = map_ctor;
  conv->map_iter_func = map_iter_func;
  conv->map_size_func = map_size_func;
  return conv;
}

/*!
 * \brief SafeCall implementation for the converter as a Function.
 * Validates num_args == 1, converts input, returns owned result.
 */
inline int TVMFFIPyTypeConverterSafeCall(void* self, const TVMFFIAny* args, int32_t num_args,
                                         TVMFFIAny* rv) noexcept {
  if (num_args != 1) {
    std::string msg = "type converter expects exactly 1 argument, got " + std::to_string(num_args);
    TVMFFIPySetTypeError(msg.c_str(), msg.size());
    return -1;
  }
  TVMFFIPyTypeConverter* converter = static_cast<TVMFFIPyTypeConverter*>(self);
  rv->type_index = kTVMFFINone;
  rv->zero_padding = 0;
  rv->v_int64 = 0;
  return converter->root->Convert(&args[0], rv, converter);
}

/*!
 * \brief Deleter for the converter.
 */
inline void TVMFFIPyTypeConverterDeleter(void* self) noexcept {
  delete static_cast<TVMFFIPyTypeConverter*>(self);
}

#endif  // TVM_FFI_TYPE_CONVERTER_H_
