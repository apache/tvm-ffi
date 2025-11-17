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

# Cache global functions at module initialization
# This avoids repeated lookups
const _module_loader = Ref{Union{TVMFunction, Nothing}}(nothing)
const _function_getter = Ref{Union{TVMFunction, Nothing}}(nothing)

"""
Initialize cached global functions.
Called from __init__() in main module.
"""
function _init_module_api()
    _module_loader[] = get_global_func("ffi.ModuleLoadFromFile")
    _function_getter[] = get_global_func("ffi.ModuleGetFunction")
    
    if _module_loader[] === nothing
        @warn "ffi.ModuleLoadFromFile not found - module loading unavailable"
    end
    if _function_getter[] === nothing
        @warn "ffi.ModuleGetFunction not found - function querying unavailable"
    end
end

"""
    TVMModule

Wrapper for TVM module objects.
"""
struct TVMModule
    handle::TVMObject
    
    TVMModule(obj::TVMObject) = new(obj)
end

"""
    load_module(path::AbstractString) -> TVMModule

Load a compiled TVM module from file.

# Example
```julia
mod = load_module("build/add_one_cpu.so")
```
"""
function load_module(path::AbstractString)
    loader = _module_loader[]
    if loader === nothing
        error("Module API not initialized - did __init__() run?")
    end
    
    mod_obj = loader(path)
    return TVMModule(mod_obj)
end

"""
    get_function(mod::TVMModule, name, query_imports=true) -> Union{TVMFunction, Nothing}

Get a function from the module by name.

Returns `nothing` if function not found.

# Example
```julia
mod = load_module("build/add_one_cpu.so")
add_one = get_function(mod, "add_one_cpu")
```
"""
function get_function(mod::TVMModule, name::AbstractString, query_imports::Bool=true)
    getter = _function_getter[]
    if getter === nothing
        error("Module API not initialized - did __init__() run?")
    end
    
    return getter(mod.handle, name, query_imports)
end

"""
    Base.getindex(mod::TVMModule, name::AbstractString) -> TVMFunction

Get a function from module using bracket notation (Python-style).

# Examples
```julia
mod = load_module("build/add_one_cpu.so")
add_one = mod["add_one_cpu"]  # Cleaner!
add_one(x, y)
```
"""
function Base.getindex(mod::TVMModule, name::AbstractString)
    func = get_function(mod, name, true)
    if func === nothing
        error("Function '$name' not found in module")
    end
    return func
end

Base.show(io::IO, mod::TVMModule) = print(io, "TVMModule(@", repr(UInt(mod.handle.handle)), ")")
