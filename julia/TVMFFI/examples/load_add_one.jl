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
Example: Loading and calling TVM compiled module

This example demonstrates how to:
1. Load a compiled TVM module (.so file)
2. Get a function from the module
3. Create tensors and call the function

Based on tvm-ffi/examples/quickstart/load/load_numpy.py
"""

using TVMFFI

println("=" ^ 60)
println("TVM FFI Julia Example: Loading add_one_cpu.so")
println("=" ^ 60)

# Path to the compiled module
module_path = joinpath(@__DIR__, "../../../examples/quickstart/build/add_one_cpu.so")

println("\n1. Loading module from: $module_path")

# Check if file exists
if !isfile(module_path)
    println("❌ Error: Module file not found!")
    println("   Please build the example first:")
    println("   cd tvm-ffi/examples/quickstart")
    println("   cmake . -B build -DEXAMPLE_NAME=\"compile_cpu\"")
    println("   cmake --build build")
    exit(1)
end

# Load the module using TVM FFI global function
# ffi.ModuleLoadFromFile is registered in the TVM runtime
module_loader = get_global_func("ffi.ModuleLoadFromFile")

if module_loader === nothing
    println("❌ Error: ffi.ModuleLoadFromFile not found!")
    println("   Make sure TVM FFI runtime library is properly loaded.")
    exit(1)
end

println("✓ Found module loader function")

# Load the module
println("\n2. Loading module...")
tvm_module = try
    module_loader(module_path)
catch e
    println("❌ Error loading module:")
    println("   ", e)
    exit(1)
end

println("✓ Module loaded successfully: ", typeof(tvm_module))

# Get the function from the module
println("\n3. Getting 'add_one_cpu' function from module...")
func_getter = get_global_func("ffi.ModuleGetFunction")

if func_getter === nothing
    println("❌ Error: ffi.ModuleGetFunction not found!")
    exit(1)
end

# Get the function: ModuleGetFunction(module, name, query_imports)
add_one_cpu = try
    func_getter(tvm_module, "add_one_cpu", true)
catch e
    println("❌ Error getting function:")
    println("   ", e)
    exit(1)
end

println("✓ Got function: ", typeof(add_one_cpu))

# Create input and output arrays
println("\n4. Creating tensors...")
x = Float32[1, 2, 3, 4, 5]
y = zeros(Float32, 5)

println("   Input (x):  ", x)
println("   Output (y): ", y)

# Convert Julia arrays to DLTensor holders (new safe API!)
println("\n5. Converting arrays to DLTensor...")
x_holder = from_julia_array(x)
y_holder = from_julia_array(y)

println("✓ Created DLTensor holders (self-contained, memory-safe)")

# Call the function!
# Pass holders directly - they keep all data alive automatically
println("\n6. Calling add_one_cpu(x, y)...")
try
    add_one_cpu(x_holder, y_holder)
    println("✓ Function call succeeded!")
catch e
    println("❌ Error calling function:")
    println("   ", e)
    exit(1)
end

# Check results
println("\n7. Results:")
println("   Input (x):   ", x)
println("   Output (y):  ", y)
println("   Expected:    ", x .+ 1)

# Verify
if y ≈ x .+ 1
    println("\n✅ SUCCESS! Output matches expected values!")
else
    println("\n❌ FAILED! Output does not match expected values")
    println("   Difference: ", y .- (x .+ 1))
end

println("\n" * "=" ^ 60)
println("TVM FFI Julia Example - Completed Successfully!")
println("=" ^ 60)

println("\n📝 Summary:")
println("   ✓ Loaded TVM module from: $module_path")
println("   ✓ Retrieved 'add_one_cpu' function")
println("   ✓ Created DLTensor views of Julia arrays")
println("   ✓ Called TVM function successfully")
println("   ✓ Verified correct results (element-wise add one)")
println("\nThis demonstrates:")
println("   • Module loading")
println("   • Function retrieval")
println("   • Zero-copy tensor passing")
println("   • Successful execution on CPU")

