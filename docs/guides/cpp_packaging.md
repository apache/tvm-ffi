<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->
# C++ Packaging and Distribution

This guide explains how to package and distribute C++ libraries that use tvm-ffi, with a focus on ABI compatibility and cross-platform distribution.

## Distribution and ABI Compatibility

When distributing kernels or libraries that use tvm-ffi, it's important to understand the ABI compatibility challenges that arise from glibc versioning. This section provides guidance for kernel authors and library distributors.

### Understanding the ABI Challenge

While tvm-ffi uses a C ABI at the interface level (through DLTensor, TVMFFISafeCallType, etc.), the tvm-ffi library itself is written in C++ and depends on specific versions of glibc and the C++ standard library. This creates potential compatibility issues from two perspectives:

**Consumer Perspective:**
Applications that link against `libtvm_ffi.so` must use a compatible glibc version. If the glibc version mismatches, STL and glibc function symbols may be incompatible, leading to runtime errors or undefined behavior.

**Producer/Kernel Distributor Perspective:**
Even when kernel authors expose their functionality through the tvm-ffi interface (which solves cross-framework ABI issues like tensor representation), if their compiled `kernel.so` shared library contains ANY glibc or tvm_ffi symbols, consumers with different glibc versions may encounter undefined symbol errors at load time.

### The manylinux Solution

The recommended solution is to use the [manylinux](https://github.com/pypa/manylinux) approach, which is the standard way Python packages handle cross-platform binary distribution. The key principle is to build on an old glibc version and run on newer versions.

Since glibc maintains forward compatibility (mostly), libraries built against an older glibc version will work on systems with newer glibc versions. The `apache-tvm-ffi` Python wheel is already built using manylinux-compatible environments.

### Practical Guidance for Kernel Distributors

#### For Pure C++ Library Distribution

If you're distributing C++ libraries or CUDA kernels:

1. **Use a Docker image with an old glibc version** for building:

   ```bash
   # See CONTRIBUTING.md for pre-built Docker images
   # Or use manylinux Docker images as a base
   docker pull quay.io/pypa/manylinux2014_x86_64
   ```

2. **For CUDA kernels**, ensure both your host launching code and the kernel are built in this environment:

   ```bash
   # Inside the container
   nvcc -shared -Xcompiler -fPIC your_cuda_kernel.cu -o kernel.so \
       $(tvm-ffi-config --cxxflags) \
       $(tvm-ffi-config --ldflags) \
       $(tvm-ffi-config --libs)
   ```

3. **Link against manylinux-compatible tvm_ffi.so**: Use the tvm-ffi library from the `apache-tvm-ffi` wheel, which is already manylinux-compatible.

#### Build System Considerations

- **Containerized builds**: Many C++ projects use containerized build systems. Adapt your existing Docker setup to use manylinux base images or images with older glibc versions.
- **CI/CD pipelines**: Configure your continuous integration to build in manylinux environments. GitHub Actions and other CI services support Docker-based builds.
- **Testing**: Always test your distributed binaries on multiple Linux distributions to verify compatibility.

### Verification

To check the glibc version your binary depends on:

```bash
objdump -T your_kernel.so | grep GLIBC_
```

This shows the minimum glibc version required. Ensure it's compatible with your target deployment environments.

### Summary

- **Build on old glibc** (via manylinux or old Linux distributions)
- **Run on new glibc** (forward compatibility guaranteed)
- **Use containerized builds** for reproducible environments
- **Test across distributions** to verify compatibility

For more details on setting up development environments, see `CONTRIBUTING.md`.
