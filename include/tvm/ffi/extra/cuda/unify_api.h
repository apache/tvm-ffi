
#ifndef TVM_FFI_EXTRA_CUDA_UNIFY_API_H_
#define TVM_FFI_EXTRA_CUDA_UNIFY_API_H_

#include <cuda.h>
#include <cuda_runtime.h>

#if CUDART_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
#endif