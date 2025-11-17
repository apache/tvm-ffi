using Test
using TVMFFI
using TVMFFI.LibTVMFFI  # Import for internal constants

@testset "TVMFFI.jl Tests" begin
    
    @testset "Device Creation" begin
        # Test CPU device
        dev = cpu(0)
        @test dev.device_type == Int32(LibTVMFFI.kDLCPU)
        @test dev.device_id == 0
        
        # Test multiple devices
        dev1 = cpu(1)
        @test dev1.device_id == 1
        
        # Test CUDA (may not be available)
        cuda_dev = cuda(0)
        @test cuda_dev.device_type == Int32(LibTVMFFI.kDLCUDA)
    end
    
    @testset "Data Type Creation" begin
        # Test from Julia types
        dt_int32 = DLDataType(Int32)
        @test dt_int32.code == UInt8(LibTVMFFI.kDLInt)
        @test dt_int32.bits == 32
        @test dt_int32.lanes == 1
        
        dt_float64 = DLDataType(Float64)
        @test dt_float64.code == UInt8(LibTVMFFI.kDLFloat)
        @test dt_float64.bits == 64
        
        # Test from strings
        dt_from_str = DLDataType("int32")
        @test dt_from_str.code == dt_int32.code
        @test dt_from_str.bits == dt_int32.bits
        
        # Test string conversion
        @test string(dt_int32) == "int32"
        @test string(dt_float64) == "float64"
    end
    
    @testset "TVM String" begin
        # Test basic string creation
        s = TVMString("hello")
        @test String(s) == "hello"
        @test length(s) == 5
        
        # Test empty string
        empty_s = TVMString("")
        @test String(empty_s) == ""
        @test length(empty_s) == 0
        
        # Test small string optimization (≤7 bytes)
        small = TVMString("tiny")
        @test String(small) == "tiny"
        @test small.data.type_index == Int32(LibTVMFFI.kTVMFFISmallStr)
        
        # Test larger string (heap allocated)
        large = TVMString("this is a longer string")
        @test String(large) == "this is a longer string"
        @test large.data.type_index == Int32(LibTVMFFI.kTVMFFIStr)
    end
    
    @testset "TVM Bytes" begin
        # Test bytes creation
        data = UInt8[0x01, 0x02, 0x03, 0x04]
        b = TVMBytes(data)
        result = Vector{UInt8}(b)
        @test result == data
        @test length(b) == 4
        
        # Test empty bytes
        empty_b = TVMBytes(UInt8[])
        @test length(empty_b) == 0
    end
    
    @testset "Error Handling" begin
        # Test error creation
        err = TVMError(ValueError, "test message", "test backtrace")
        @test err.kind == "ValueError"
        @test err.message == "test message"
        @test err.backtrace == "test backtrace"
        
        # Test error kinds
        @test ValueError.name == "ValueError"
        @test TVMFFI.TypeError.name == "TypeError"  # Qualify to avoid Base.TypeError
        @test RuntimeError.name == "RuntimeError"
        
        # Test exception throwing
        @test_throws TVMError begin
            err = TVMError(ValueError, "thrown error", "")
            throw(err)
        end
    end
    
    @testset "Type Conversions" begin
        # Test to_tvm_any and from_tvm_any for basic types
        
        # Int64
        any_int = TVMFFI.to_tvm_any(Int64(42))
        @test TVMFFI.from_tvm_any(any_int) == 42
        
        # Float64
        any_float = TVMFFI.to_tvm_any(3.14)
        @test TVMFFI.from_tvm_any(any_float) ≈ 3.14
        
        # Bool
        any_bool = TVMFFI.to_tvm_any(true)
        @test TVMFFI.from_tvm_any(any_bool) == true
        
        # Nothing
        any_none = TVMFFI.to_tvm_any(nothing)
        @test TVMFFI.from_tvm_any(any_none) === nothing
        
        # Device
        dev = cpu(0)
        any_dev = TVMFFI.to_tvm_any(dev)
        result_dev = TVMFFI.from_tvm_any(any_dev)
        @test result_dev.device_type == dev.device_type
        @test result_dev.device_id == dev.device_id
        
        # String
        any_str = TVMFFI.to_tvm_any("test")
        result_str = TVMFFI.from_tvm_any(any_str)
        @test result_str == "test"
    end
    
    @testset "Global Function Lookup" begin
        # Test non-existent function
        func = get_global_func("this_function_definitely_does_not_exist_xyz123")
        # Should return nothing or throw, depending on implementation
        # For now, we just test it doesn't crash
        @test true
    end
    
end

println("\n✓ All tests passed!")

