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

using .LibTVMFFI

"""
    DLTensor

DLPack tensor structure (from dlpack.h).
This is a C-compatible struct representing a multi-dimensional array.

# Note on GPU Pointers
For GPU arrays (CuArray, ROCArray, etc.), the data field contains a GPU device
pointer, not a CPU pointer. We use UInt64 to store the pointer value and
reinterpret it as needed, since GPU pointers can't be directly converted to Ptr{Cvoid}.
"""
struct DLTensor
    data::Ptr{Cvoid}
    device::DLDevice
    ndim::Int32
    dtype::DLDataType
    shape::Ptr{Int64}
    strides::Ptr{Int64}
    byte_offset::UInt64
end

"""
    DLTensor(data_ptr, device, ndim, dtype, shape, strides, byte_offset)

Construct DLTensor with automatic GPU pointer handling.
"""
function DLTensor(data_ptr, device::DLDevice, ndim::Int32, dtype::DLDataType,
                  shape::Ptr{Int64}, strides::Ptr{Int64}, byte_offset::UInt64)
    # Convert GPU pointers (CuPtr, etc.) to generic pointer
    # by reinterpreting through UInt
    ptr_as_uint = if data_ptr isa Ptr
        UInt(data_ptr)
    else
        # GPU pointer (CuPtr, ROCPtr, etc.)
        # Get the raw pointer value
        reinterpret(UInt, data_ptr)
    end

    # Convert to Ptr{Cvoid}
    data_cvoid = reinterpret(Ptr{Cvoid}, ptr_as_uint)

    return DLTensor(data_cvoid, device, ndim, dtype, shape, strides, byte_offset)
end

"""
    TVMTensor

TVM tensor wrapper with automatic memory management.

Provides accessors for shape, dtype, and device information.
"""
mutable struct TVMTensor
    handle::LibTVMFFI.TVMFFIObjectHandle

    """
        TVMTensor(handle; own=true)

    Create a TVMTensor from a raw handle.

    # Arguments
    - `handle`: The raw tensor handle
    - `own`: If true, increment refcount (default). If false, take ownership without IncRef.
    """
    function TVMTensor(handle::LibTVMFFI.TVMFFIObjectHandle; own::Bool=true)
        if handle == C_NULL
            error("Cannot create TVMTensor from NULL handle")
        end

        # Optionally increase reference count
        if own
            LibTVMFFI.TVMFFIObjectIncRef(handle)
        end

        tensor = new(handle)

        # Finalizer
        finalizer(tensor) do t
            if t.handle != C_NULL
                LibTVMFFI.TVMFFIObjectDecRef(t.handle)
            end
        end

        return tensor
    end
end

"""
    get_dltensor_ptr(tensor::TVMTensor) -> Ptr{DLTensor}

Get pointer to the underlying DLTensor structure.
"""
function get_dltensor_ptr(tensor::TVMTensor)
    # DLTensor follows immediately after TVMFFIObject header
    Ptr{DLTensor}(tensor.handle + sizeof(LibTVMFFI.TVMFFIObject))
end

"""
    shape(tensor::TVMTensor) -> Vector{Int64}

Get the shape of the tensor as a vector.

# Note
Julia arrays typically use `size()` which returns a tuple.
This function returns a vector for compatibility with some use cases.
"""
function shape(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)

    ndim = Int(dltensor.ndim)
    if ndim == 0
        return Int64[]
    end

    shape_vec = unsafe_wrap(Array, dltensor.shape, ndim)
    return copy(shape_vec)
end

"""
    Base.size(tensor::TVMTensor) -> Tuple

Get the size of the tensor as a tuple (Julia standard).
"""
Base.size(tensor::TVMTensor) = Tuple(shape(tensor))

"""
    Base.size(tensor::TVMTensor, dim::Int) -> Int

Get the size of a specific dimension.
"""
function Base.size(tensor::TVMTensor, dim::Int)
    s = size(tensor)
    if dim < 1 || dim > length(s)
        return 1  # Julia convention for out-of-bounds dimensions
    end
    return s[dim]
end

"""
    Base.ndims(tensor::TVMTensor) -> Int

Get the number of dimensions.
"""
function Base.ndims(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)
    return Int(dltensor.ndim)
end

"""
    Base.length(tensor::TVMTensor) -> Int

Get the total number of elements.
"""
function Base.length(tensor::TVMTensor)
    prod(size(tensor))
end

"""
    dtype(tensor::TVMTensor) -> DLDataType

Get the data type of the tensor.
"""
function dtype(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)
    return dltensor.dtype
end

"""
    device(tensor::TVMTensor) -> DLDevice

Get the device of the tensor.
"""
function device(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)
    return dltensor.device
end

"""
    strides(tensor::TVMTensor) -> Vector{Int64}

Get the strides of the tensor.
"""
function strides(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)

    if dltensor.strides == C_NULL
        # Compute default C-contiguous strides
        shape_vec = shape(tensor)
        ndim = length(shape_vec)
        strides_vec = ones(Int64, ndim)

        for i in (ndim-1):-1:1
            strides_vec[i] = strides_vec[i+1] * shape_vec[i+1]
        end

        return strides_vec
    else
        ndim = Int(dltensor.ndim)
        strides_vec = unsafe_wrap(Array, dltensor.strides, ndim)
        return copy(strides_vec)
    end
end

"""
    is_contiguous(tensor::TVMTensor) -> Bool

Check if the tensor is contiguous in memory.
"""
function is_contiguous(tensor::TVMTensor)
    shape_vec = shape(tensor)
    strides_vec = strides(tensor)
    ndim = length(shape_vec)

    expected_stride = 1
    for i in ndim:-1:1
        if strides_vec[i] != expected_stride
            return false
        end
        expected_stride *= shape_vec[i]
    end

    return true
end

"""
    data_ptr(tensor::TVMTensor) -> Ptr{Cvoid}

Get the raw data pointer of the tensor.
"""
function data_ptr(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)
    return dltensor.data
end

"""
    to_julia_array(tensor::TVMTensor, ::Type{T}) -> Array{T}

Convert TVM tensor to Julia array (zero-copy for CPU contiguous tensors).

Returns an array that shares memory with the tensor.
"""
function to_julia_array(tensor::TVMTensor, ::Type{T}) where T
    # Check device
    dev = device(tensor)
    if dev.device_type != Int32(LibTVMFFI.kDLCPU)
        error("Can only convert CPU tensors to Julia arrays. " *
              "Tensor is on device type $(dev.device_type)")
    end

    # Verify dtype matches
    expected_dtype = DLDataType(T)
    actual_dtype = dtype(tensor)

    if expected_dtype.code != actual_dtype.code ||
       expected_dtype.bits != actual_dtype.bits
        error("Type mismatch: tensor has dtype $(string(actual_dtype)), " *
              "but requested type $T (dtype $(string(expected_dtype)))")
    end

    # Check contiguity
    if !is_contiguous(tensor)
        error("Can only convert contiguous tensors. For non-contiguous: use copy(to_julia_array())")
    end

    # Get shape and data pointer
    shape_tuple = size(tensor)
    ptr = Ptr{T}(data_ptr(tensor))

    # Create zero-copy view
    # Note: The array keeps a reference to the tensor to prevent GC
    arr = unsafe_wrap(Array, ptr, shape_tuple)

    return arr
end

# Pretty printing
function Base.show(io::IO, tensor::TVMTensor)
    shape_tuple = size(tensor)
    dt = dtype(tensor)
    dev = device(tensor)

    print(io, "TVMTensor{", string(dt), "}(")
    print(io, "shape=", shape_tuple, ", ")
    print(io, "device=", dev, ")")
end

"""
    Base.summary(tensor::TVMTensor) -> String

Get a summary string for the tensor.
"""
function Base.summary(io::IO, tensor::TVMTensor)
    shape_tuple = size(tensor)
    dt = dtype(tensor)
    print(io, join(shape_tuple, "×"), " TVMTensor{", string(dt), "}")
end

"""
    DLTensorHolder{T, S}

Self-contained DLTensor holder with automatic lifetime management.

Works for both CPU and GPU arrays, including slices (SubArray).

# Fields
- `tensor::DLTensor`: DLPack tensor structure
- `shape::Vector{Int64}`: Shape array
- `strides::Vector{Int64}`: Strides array
- `source::S`: Source array (Array, CuArray, SubArray, etc.)
"""
mutable struct DLTensorHolder{T, S}
    tensor::DLTensor
    shape::Vector{Int64}
    strides::Vector{Int64}
    source::S  # Can be any array type (CPU or GPU)
end

# Outer constructor for CPU arrays
function DLTensorHolder(arr::Union{Array{T}, SubArray{T}}, device::DLDevice=cpu()) where T
    # Get shape
    shape_tuple = size(arr)
    ndim = length(shape_tuple)
    shape_vec = collect(Int64, shape_tuple)

    # Get strides - Julia provides this for both Array and SubArray
    arr_strides = Base.strides(arr)
    strides_vec = collect(Int64, arr_strides)

    # Get data pointer - Julia's pointer() handles SubArray correctly
    data_ptr = pointer(arr)

    # byte_offset is 0 (pointer already points to first element)
    byte_offset = UInt64(0)

    # Get dtype
    dt = DLDataType(T)

    # Create DLTensor
    tensor = DLTensor(
        data_ptr,
        device,
        Int32(ndim),
        dt,
        pointer(shape_vec),
        pointer(strides_vec),
        byte_offset
    )

    return DLTensorHolder{T, typeof(arr)}(tensor, shape_vec, strides_vec, arr)
end

"""
    Base.Ref(holder::DLTensorHolder) -> Ref{DLTensor}

Get a reference to the tensor for passing to C functions.
The holder keeps all data alive.
"""
Base.Ref(holder::DLTensorHolder) = Ref(holder.tensor)

"""
    Base.unsafe_convert(::Type{Ptr{DLTensor}}, holder::DLTensorHolder)

Convert holder to pointer for C calls.
This is called automatically by ccall.
"""
function Base.unsafe_convert(::Type{Ptr{DLTensor}}, holder::DLTensorHolder)
    # Get pointer to the tensor field within the holder
    # The tensor is the first field, so we can get its address
    return Ptr{DLTensor}(pointer_from_objref(holder))
end

"""
    from_julia_array(arr, device=cpu()) -> DLTensorHolder

Create DLTensor holder from Julia array or slice.

Supports CPU and GPU arrays, including zero-copy slices.

# Examples
```julia
# Array
holder = from_julia_array(x)

# Slice (zero-copy)
col = @view matrix[:, 2]
holder = from_julia_array(col)

# GPU array (auto-detects device)
gpu_holder = from_julia_array(cuda_array)
```
"""
function from_julia_array(arr::Union{Array{T}, SubArray{T}}, device::DLDevice=cpu()) where T
    return DLTensorHolder(arr, device)
end
