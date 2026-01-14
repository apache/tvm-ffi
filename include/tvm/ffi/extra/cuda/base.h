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
/*!
 * \file tvm/ffi/extra/cuda/base.h
 * \brief CUDA base utilities.
 */
#ifndef TVM_FFI_EXTRA_CUDA_BASE_H_
#define TVM_FFI_EXTRA_CUDA_BASE_H_

namespace tvm {
namespace ffi {

/*!
 * \brief A simple 3D dimension type for CUDA kernel launch configuration.
 *
 * This struct mimics the behavior of dim3 from CUDA Runtime API and provides
 * a compatible interface for kernel launch configuration. It can be constructed
 * from 1, 2, or 3 dimensions.
 */
struct dim3 {
  /*! \brief X dimension (number of blocks in x-direction or threads in x-direction) */
  unsigned int x;
  /*! \brief Y dimension (number of blocks in y-direction or threads in y-direction) */
  unsigned int y;
  /*! \brief Z dimension (number of blocks in z-direction or threads in z-direction) */
  unsigned int z;

  /*! \brief Default constructor initializes to (1, 1, 1) */
  dim3() : x(1), y(1), z(1) {}

  /*! \brief Construct with x dimension, y and z default to 1 */
  explicit dim3(unsigned int x_) : x(x_), y(1), z(1) {}

  /*! \brief Construct with x and y dimensions, z defaults to 1 */
  dim3(unsigned int x_, unsigned int y_) : x(x_), y(y_), z(1) {}

  /*! \brief Construct with all three dimensions */
  dim3(unsigned int x_, unsigned int y_, unsigned int z_) : x(x_), y(y_), z(z_) {}
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_CUDA_BASE_H_
