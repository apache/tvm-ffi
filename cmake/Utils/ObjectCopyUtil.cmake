# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# We need this to simulate `CUDA_{CUBIN,FATBIN}_COMPILATION` in `add_tvm_ffi_{cubin,fatbin}`, to
# copy `a.cu.o` to `a.cubin`/`a.fatbin`.

# Usage: cmake -DOBJECTS=<input_object_file1>;...;<input_object_fileN> -DOUT_DIR=<output_directory>
# -DEXT=<extension> -P <this_script>

# Parameter: OBJECTS: semicolon-separated list of input object files; OUT_DIR: output directory,
# empty for the same directory as the object file EXT: extension to rename to

string(REPLACE "\"" "" ext_strip "${EXT}")
string(REPLACE "\"" "" out_dir_strip "${OUT_DIR}")
foreach (obj_raw ${OBJECTS})
  string(REPLACE "\"" "" obj "${obj_raw}")

  # Extract filename: /path/to/kernel.cu.o -> kernel Note: CMake objects are usually named
  # source.cu.o, so we strip extensions twice.
  get_filename_component(fname ${obj} NAME_WE)
  get_filename_component(fname ${fname} NAME_WE)

  # If OUT_DIR is provided, use it. Otherwise, use the object's directory.
  if (NOT out_dir_strip STREQUAL "")
    set(FINAL_DIR "${out_dir_strip}")
  else ()
    get_filename_component(FINAL_DIR ${obj} DIRECTORY)
  endif ()

  message("Copying ${obj} -> ${FINAL_DIR}/${fname}.${ext_strip}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${obj}" "${FINAL_DIR}/${fname}.${ext_strip}"
  )
endforeach ()
