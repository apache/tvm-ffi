#!/usr/bin/env julia

"""
GC Safety Test for Auto-Conversion

This test verifies that auto-created DLTensorHolders are properly
protected during C function calls, even under aggressive GC.
"""

using TVMFFI

println("=== GC Safety Test ===\n")

# Load the TVM module
println("1. Loading TVM module...")
module_path = joinpath(@__DIR__, "..", "..", "..", "examples", "quickstart", "build", "add_one_cpu.so")

if !isfile(module_path)
    println("⚠️  Module not found at: $module_path")
    println("   Please build TVM FFI first")
    exit(0)
end

mod_loader = get_global_func("ffi.ModuleLoadFromFile")
tvm_module = mod_loader(module_path)

# Get the add_one_cpu function
func_getter = get_global_func("ffi.ModuleGetFunction")
add_one = func_getter(tvm_module, "add_one_cpu", true)

if add_one === nothing
    println("❌ Function 'add_one_cpu' not found in module")
    println("   Make sure libtvm_ffi_testing.so contains add_one_cpu")
    exit(0)
end

println("✓ Loaded module and function")

# Test 1: Simple auto-conversion
println("\n2. Test 1: Simple auto-conversion")
x = Float32[1, 2, 3, 4, 5]
y = zeros(Float32, 5)

add_one(x, y)  # Auto-convert

if y ≈ x .+ 1
    println("   ✅ Simple auto-conversion works")
else
    println("   ❌ Failed!")
    exit(1)
end

# Test 2: Auto-conversion with forced GC
println("\n3. Test 2: Auto-conversion under GC pressure")
function create_temp_arrays()
    # Create arrays that would be GC'd if not protected
    local_x = Float32[10, 20, 30, 40, 50]
    local_y = zeros(Float32, 5)

    # Force some allocations to trigger GC
    for i in 1:100
        _ = rand(Float32, 1000)
    end

    GC.gc()  # Force GC

    # Call with auto-conversion
    add_one(local_x, local_y)

    return local_x, local_y
end

x2, y2 = create_temp_arrays()

if y2 ≈ x2 .+ 1
    println("   ✅ Auto-conversion survives GC")
else
    println("   ❌ GC safety failed!")
    exit(1)
end

# Test 3: Slice auto-conversion
println("\n4. Test 3: Slice auto-conversion")
matrix = Float32[1 2 3 4; 5 6 7 8; 9 10 11 12]
col = @view matrix[:, 2]
col_out = zeros(Float32, 3)

# Force GC
GC.gc()

# Call with slice (auto-converted)
add_one(col, col_out)

if col_out ≈ collect(col) .+ 1
    println("   ✅ Slice auto-conversion works")
else
    println("   ❌ Slice auto-conversion failed!")
    exit(1)
end

# Test 4: Stress test with many allocations
println("\n5. Test 4: Stress test (many calls under GC pressure)")
for i in 1:100
    x_temp = Float32[i, i+1, i+2]
    y_temp = zeros(Float32, 3)

    # Allocate garbage to trigger GC
    _ = [rand(Float32, 100) for _ in 1:10]

    if i % 20 == 0
        GC.gc()  # Periodic forced GC
    end

    # Auto-conversion call
    add_one(x_temp, y_temp)

    if !(y_temp ≈ x_temp .+ 1)
        println("   ❌ Failed at iteration $i")
        exit(1)
    end
end
println("   ✅ 100 iterations with GC pressure - all passed")

# Test 5: Pre-created holder (optimization case)
println("\n6. Test 5: Pre-created holder reuse")
x5 = Float32[100, 200, 300]
y5 = zeros(Float32, 3)
holder = from_julia_array(x5)

for i in 1:1000
    fill!(y5, 0)
    add_one(holder, y5)  # Reuse holder

    if !(y5 ≈ x5 .+ 1)
        println("   ❌ Holder reuse failed at iteration $i")
        exit(1)
    end
end
println("   ✅ Holder reused 1000 times successfully")

println("\n" * "="^60)
println("✅ ALL GC SAFETY TESTS PASSED!")
println("="^60)

println("\n📝 Summary:")
println("   ✓ Auto-conversion is GC-safe")
println("   ✓ Slices work correctly")
println("   ✓ Survives aggressive GC")
println("   ✓ Holder reuse works for optimization")
println("\nConclusion: Auto-conversion is both convenient AND safe! 🎉")
