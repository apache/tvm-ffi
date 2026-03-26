#!/usr/bin/env bash
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

# Run ast_roundtrip_check.py across multiple Python versions.
# Each version gets its own isolated venv with tvm-ffi built from source.
#
# Usage: ./run_multi_python.sh <directory> <method> [extra-args...]
# Example: ./run_multi_python.sh ../../tests/python tvm_ffi.text._roundtrip
set -euo pipefail

VERSIONS=(3.9 3.10 3.11 3.12 3.13 3.14)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/ast_roundtrip_check.py"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <directory> <method> [extra-args...]" >&2
    exit 1
fi

pass=0
fail=0

for v in "${VERSIONS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Python $v"
    echo "========================================"
    if uv run --python "$v" "$SCRIPT" "$@"; then
        ((pass++))
    else
        ((fail++))
    fi
done

echo ""
echo "========================================"
echo "  ${#VERSIONS[@]} versions: $pass passed, $fail failed"
echo "========================================"
exit $((fail > 0))
