#!/bin/bash
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
set -ex

if command -v ninja >/dev/null 2>&1; then
	generator="Ninja"
else
	echo "Ninja not found, falling back to Unix Makefiles" >&2
	generator="Unix Makefiles"
fi

rm -rf build/CMakeCache.txt
cmake -G "$generator" -B build -S .
cmake --build build --parallel

# running python example
python run_example.py

# running c++ example
./build/run_example

if [ -x ./build/run_example_cuda ]; then
	./build/run_example_cuda
fi
