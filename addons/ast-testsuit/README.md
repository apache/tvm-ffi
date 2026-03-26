<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# ast-testsuit

AST roundtrip test harness: parse Python files, run them through a transform,
re-parse the output, and diff the two ASTs field by field.

## Usage

```bash
uv run ast_roundtrip_check.py <directory> <method>
```

**Arguments:**

| Argument | Description |
| --- | --- |
| `directory` | Directory to walk recursively for `.py` files |
| `method` | Dotted callable (`ast.AST -> str`), e.g. `tvm_ffi.text._roundtrip` |
| `--include-positions` | Also compare `lineno`/`col_offset` fields (skipped by default) |

## Example

```bash
# roundtrip all Python files under tests/python/testdata/
uv run ast_roundtrip_check.py ../../tests/python/testdata/ tvm_ffi.text._roundtrip
```

## How it works

For each `.py` file found:

1. **Parse** the source into a Python AST (`a`).
2. **Transform** by calling `method(a)` to get a string `b`.
3. **Re-parse** `b` into a second AST (`b'`).
4. **Compare** `a` and `b'` recursively, field by field.

Positional fields (`lineno`, `col_offset`, `end_lineno`, `end_col_offset`,
`type_comment`) are skipped by default since printers rarely preserve them.

## Output

```text
OK tests/testdata/simple.py

MISMATCH tests/testdata/complex.py (2 diff(s))
  body[0].name: 'foo' != 'bar'
  body[1].value.args: length: 2 vs 3

ERROR tests/testdata/broken.py
Traceback ...

--- 3 files: 1 ok, 1 mismatched, 1 errors ---
```

Exit code is 0 when all files match, 1 otherwise.

## Multi-version testing

The script declares `apache-tvm-ffi` as an inline dependency with a path source,
so `uv run --python <version>` builds and installs tvm-ffi into an isolated
ephemeral venv automatically:

```bash
# single version
uv run --python 3.12 ast_roundtrip_check.py <directory> <method>

# all versions (3.9–3.14)
./run_multi_python.sh <directory> <method>
```

uv downloads any missing Python interpreter on the fly.
