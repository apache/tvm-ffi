# Basic usage example for TVMFFI.jl

using TVMFFI

println("=== TVMFFI.jl Basic Usage Examples ===\n")

# 1. Device creation
println("1. Creating devices:")
cpu_device = cpu(0)
println("  CPU device: ", cpu_device)

# If you have CUDA available:
# cuda_device = cuda(0)
# println("  CUDA device: ", cuda_device)

# 2. Data type creation
println("\n2. Creating data types:")
int32_type = DLDataType(Int32)
println("  Int32 dtype: ", string(int32_type))

float64_type = DLDataType(Float64)
println("  Float64 dtype: ", string(float64_type))

# From string
float32_type = DLDataType("float32")
println("  Float32 dtype: ", string(float32_type))

# 3. String handling
println("\n3. String handling:")
tvm_str = TVMString("Hello from Julia!")
println("  TVM String: ", tvm_str)
println("  Converted to Julia: ", String(tvm_str))
println("  Length: ", length(tvm_str))

# 4. Bytes handling
println("\n4. Bytes handling:")
data = UInt8[0x48, 0x65, 0x6c, 0x6c, 0x6f]  # "Hello" in ASCII
tvm_bytes = TVMBytes(data)
println("  TVM Bytes: ", tvm_bytes)
println("  Converted to Julia: ", Vector{UInt8}(tvm_bytes))

# 5. Error handling
println("\n5. Error handling:")
try
    # Try to get a non-existent function
    func = get_global_func("this_function_does_not_exist_12345")
    if func !== nothing
        println("  Unexpected: function found!")
    else
        println("  Function not found (returned nothing)")
    end
catch e
    if e isa TVMError
        println("  Caught TVMError:")
        println("    Kind: ", e.kind)
        println("    Message: ", e.message)
    else
        println("  Caught other error: ", typeof(e))
    end
end

# 6. Creating custom errors
println("\n6. Creating custom errors:")
try
    # Create and throw a custom TVM error
    err = TVMError(ValueError, "This is a test error", "test_backtrace")
    throw(err)
catch e
    if e isa TVMError
        println("  Caught custom error:")
        println("    Kind: ", e.kind)
        println("    Message: ", e.message)
    end
end

println("\n=== Examples completed successfully ===")
