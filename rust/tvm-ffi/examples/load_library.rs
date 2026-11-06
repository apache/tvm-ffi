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
use tvm_ffi::{Module, Result, Tensor};

fn main() {
    // lib_path is build using scripts/generate_example_lib.py [out_path]
    // invoked automatically through build.rs
    // you can replace it to any other ffi compatible library
    let lib_path = concat!(env!("OUT_DIR"), "/add_one_cpu.so");
    let lib: tvm_ffi::Module = Module::load_from_file(lib_path).unwrap();
    // get the function from the library
    let add_one_cpu: tvm_ffi::Function = lib.get_function("add_one_cpu").unwrap();
    // load the function from the library
    // and cast into a typed version of the function
    let typed_add_one = tvm_ffi::into_typed_fn!(
        add_one_cpu,
        Fn(&Tensor, &Tensor) -> Result<()>
    );
    // run the function
    let x_data: &[f32] = &[0.0, 1.0, 2.0, 3.0];
    let x = Tensor::from_slice(x_data, &[4]).unwrap();
    let y = Tensor::from_slice(&[0.0f32; 4], &[4]).unwrap();
    println!("x: {:?}", x.data_as_slice::<f32>().unwrap());
    typed_add_one(&x, &y).unwrap();
    println!("y: {:?}", y.data_as_slice::<f32>().unwrap());
}
