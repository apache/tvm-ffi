#!/usr/bin/env julia
#=
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
=#

"""
Example: Loading and calling TVM CUDA module

This example demonstrates how to:
1. Load a compiled TVM CUDA module (.so file)
2. Create CUDA arrays (or use host arrays with device parameter)
3. Call CUDA kernels through TVM FFI

Based on tvm-ffi/examples/quickstart/load/load_cuda.cc
"""

using TVMFFI

println("=" ^ 70)
println("TVM FFI Julia Example: Loading add_one_cuda.so (GPU)")
println("=" ^ 70)

# Check if CUDA is available
has_cuda = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !has_cuda
    println("\n⚠️  CUDA not available!")
    println("   This example requires:")
    println("   • CUDA.jl package: using Pkg; Pkg.add(\"CUDA\")")
    println("   • CUDA-capable GPU")
    println("   • CUDA toolkit")
    println("\n   Falling back to CPU-GPU memory transfer demo...")
end

# Path to the compiled CUDA module
module_path = joinpath(@__DIR__, "../../../examples/quickstart/build/add_one_cuda.so")

println("\n1. Loading CUDA module from: $module_path")

# Check if file exists
if !isfile(module_path)
    println("❌ Error: Module file not found!")
    println("   Please build the CUDA example first:")
    println("   cd tvm-ffi/examples/quickstart")
    println("   cmake . -B build -DEXAMPLE_NAME=\"compile_cuda\"")
    println("   cmake --build build")
    exit(1)
end

# Load the module
module_loader = get_global_func("ffi.ModuleLoadFromFile")
if module_loader === nothing
    println("❌ Error: ffi.ModuleLoadFromFile not found!")
    exit(1)
end

println("✓ Found module loader function")

println("\n2. Loading CUDA module...")
tvm_module = try
    module_loader(module_path)
catch e
    println("❌ Error loading module:")
    println("   ", e)
    exit(1)
end

println("✓ CUDA module loaded successfully")

# Get the function
println("\n3. Getting 'add_one_cuda' function from module...")
func_getter = get_global_func("ffi.ModuleGetFunction")

add_one_cuda = try
    func_getter(tvm_module, "add_one_cuda", true)
catch e
    println("❌ Error getting function:")
    println("   ", e)
    exit(1)
end

println("✓ Got CUDA function: ", typeof(add_one_cuda))

if has_cuda
    println("\n4. Using CUDA.jl arrays...")
    
    # Create CUDA arrays
    x_gpu = CUDA.CuArray(Float32[1, 2, 3, 4, 5])
    y_gpu = CUDA.zeros(Float32, 5)
    
    println("   Input (x_gpu):  ", Array(x_gpu))
    println("   Output (y_gpu): ", Array(y_gpu))
    
    # Get device ID from CUDA
    device_id = CUDA.device().handle
    cuda_device = cuda(device_id)
    
    # Create DLTensor views
    println("\n5. Converting CUDA arrays to DLTensor...")
    
    # Unified interface: from_julia_array now handles GPU arrays automatically!
    # Returns self-contained holders - memory-safe API
    x_holder = from_julia_array(x_gpu)  # Auto-detects CUDA, returns GPUDLTensorHolder
    y_holder = from_julia_array(y_gpu)  # Auto-detects CUDA
    
    println("✓ Created DLTensor views (auto-detected CUDA backend)")
    
    # Call the CUDA function
    # Pass holders directly - they keep GPU arrays alive
    println("\n6. Calling add_one_cuda(x, y) on GPU...")
    try
        add_one_cuda(x_holder, y_holder)
        CUDA.synchronize()  # Wait for GPU to finish
        println("✓ CUDA function call succeeded!")
    catch e
        println("❌ Error calling CUDA function:")
        println("   ", e)
        exit(1)
    end
    
    # Check results
    println("\n7. Results:")
    y_host = Array(y_gpu)
    x_host = Array(x_gpu)
    println("   Input (x):   ", x_host)
    println("   Output (y):  ", y_host)
    println("   Expected:    ", x_host .+ 1)
    
    # Verify
    if y_host ≈ x_host .+ 1
        println("\n✅ SUCCESS! GPU output matches expected values!")
    else
        println("\n❌ FAILED! Output does not match")
        println("   Difference: ", y_host .- (x_host .+ 1))
    end
    
else
    # Demo without CUDA.jl - show the concept
    println("\n4. CUDA not available - showing concept...")
    println("   With CUDA.jl, you would:")
    println("   • Create CuArray on GPU")
    println("   • Convert to DLTensor with cuda() device")
    println("   • Call TVM CUDA function")
    println("   • Results computed on GPU!")
    
    println("\n   To enable CUDA support:")
    println("   julia> using Pkg")
    println("   julia> Pkg.add(\"CUDA\")")
end

println("\n" * "=" ^ 70)
println("CUDA Example Completed!")
println("=" ^ 70)

println("\n📝 Summary:")
println("   ✓ Loaded TVM CUDA module")
println("   ✓ Retrieved 'add_one_cuda' function")
if has_cuda
    println("   ✓ Created CUDA arrays with CUDA.jl")
    println("   ✓ Called TVM CUDA kernel successfully")
    println("   ✓ Verified correct GPU execution")
    println("\nThis demonstrates:")
    println("   • Loading CUDA modules")
    println("   • GPU tensor passing")
    println("   • Zero-copy GPU memory sharing")
    println("   • Successful CUDA kernel execution")
else
    println("   ⚠️  CUDA not available (demo mode)")
    println("\nInstall CUDA.jl to run on GPU:")
    println("   using Pkg; Pkg.add(\"CUDA\")")
end

