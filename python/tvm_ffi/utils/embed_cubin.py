#!/usr/bin/env python3
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
"""Utility script to embed CUBIN data into existing object files.

This script takes an object file containing C++ code that uses TVM_FFI_EMBED_CUBIN
and embeds the CUBIN binary data directly into it, creating a new combined object file.

Usage:
    python -m tvm_ffi.utils.embed_cubin \
        --output-obj new_obj.o \
        --input-obj old_obj.o \
        --cubin my-cubin.cubin \
        --name my_module

The output object file will contain both:
- The original C++ code from input-obj
- The embedded CUBIN data with symbols:
    __tvm_ffi__cubin_<name>
    __tvm_ffi__cubin_<name>_end

These symbols are accessed in C++ code using:
    TVM_FFI_EMBED_CUBIN(my_module);
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


def embed_cubin(  # noqa: PLR0912, PLR0915
    cubin_path: Path,
    input_obj_path: Path,
    output_obj_path: Path,
    name: str,
    verbose: bool = False,
) -> None:
    """Embed a CUBIN file into an existing object file.

    This function takes a CUBIN binary file and merges it into an existing object
    file that contains C++ code using TVM_FFI_EMBED_CUBIN. The result is a new
    object file containing both the original code and the embedded CUBIN data.

    The process involves:
    1. Creating an intermediate CUBIN object file using `ld -r -b binary`
    2. Adding `.note.GNU-stack` section to CUBIN object for security
    3. Using `objcopy` to rename symbols to the format expected by TVM-FFI:
       - _binary_<filename>_start → __tvm_ffi__cubin_<name>
       - _binary_<filename>_end → __tvm_ffi__cubin_<name>_end
    4. Using `ld -r` (relocatable link) to merge the CUBIN object with the input object
    5. Localizing the symbols to prevent conflicts when multiple object files use the same name

    Parameters
    ----------
    cubin_path : Path
        Path to the input CUBIN file.
    input_obj_path : Path
        Path to the existing object file containing C++ code with TVM_FFI_EMBED_CUBIN.
    output_obj_path : Path
        Path to the output object file (will contain both code and CUBIN data).
    name : str
        Name used in the TVM_FFI_EMBED_CUBIN macro (e.g., "my_module").
        This name must match the name used in the C++ code.
    verbose : bool, optional
        If True, print detailed command output. Default is False.

    Raises
    ------
    FileNotFoundError
        If the input CUBIN file or input object file does not exist.
    RuntimeError
        If `ld` or `objcopy` commands fail.

    Examples
    --------
    In Python,

    ```python
    embed_cubin(
        Path("kernel.cubin"),
        Path("old.o"),
        Path("new.o"),
        "my_kernels",
    )
    ```

    Then in C++ code,

    ```C++
    TVM_FFI_EMBED_CUBIN(my_kernels);
    auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_kernels, "kernel_name");
    ```

    (in the source that was compiled to old.o).

    """
    if not cubin_path.exists():
        raise FileNotFoundError(f"CUBIN file not found: {cubin_path}")

    if not input_obj_path.exists():
        raise FileNotFoundError(f"Input object file not found: {input_obj_path}")

    if verbose:
        print(f"Input CUBIN: {cubin_path} ({cubin_path.stat().st_size} bytes)")
        print(f"Input object: {input_obj_path} ({input_obj_path.stat().st_size} bytes)")
        print(f"Output object: {output_obj_path}")
        print(f"Symbol name: {name}")

    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Copy CUBIN to temp directory with a predictable name for symbol generation
        # The filename affects the auto-generated symbol names from ld
        temp_cubin = tmp_path / f"embedded_{name}.cubin"
        temp_cubin.write_bytes(cubin_path.read_bytes())

        # Step 1: Create intermediate CUBIN object file with ld -r -b binary
        cubin_obj = tmp_path / f"cubin_{name}.o"
        ld_binary_cmd = ["ld", "-r", "-b", "binary", "-o", str(cubin_obj), str(temp_cubin.name)]

        if verbose:
            print("\n[Step 1/4] Creating CUBIN object file with ld")
            print(f"Command: {' '.join(ld_binary_cmd)}")
            print(f"Working directory: {tmp_path}")

        result = subprocess.run(ld_binary_cmd, cwd=str(tmp_path), capture_output=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8")
            stdout = result.stdout.decode("utf-8")
            raise RuntimeError(
                f"ld (binary) failed with status {result.returncode}\n"
                f"Command: {' '.join(ld_binary_cmd)}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

        if verbose and result.stdout:
            print(f"stdout: {result.stdout.decode('utf-8')}")
        if verbose and result.stderr:
            print(f"stderr: {result.stderr.decode('utf-8')}")

        # Step 1.5: Add .note.GNU-stack section to CUBIN object for security
        # This marks the stack as non-executable and prevents linker warnings
        # We do this before renaming so the section is preserved in the final output
        objcopy_stack_cmd = [
            "objcopy",
            "--add-section",
            ".note.GNU-stack=/dev/null",
            "--set-section-flags",
            ".note.GNU-stack=noload,readonly",
            str(cubin_obj),
        ]

        if verbose:
            print("\n[Step 1.5/4] Adding .note.GNU-stack section to CUBIN object")
            print(f"Command: {' '.join(objcopy_stack_cmd)}")

        result = subprocess.run(objcopy_stack_cmd, capture_output=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8")
            stdout = result.stdout.decode("utf-8")
            raise RuntimeError(
                f"objcopy (add stack section) failed with status {result.returncode}\n"
                f"Command: {' '.join(objcopy_stack_cmd)}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

        if verbose and result.stdout:
            print(f"stdout: {result.stdout.decode('utf-8')}")
        if verbose and result.stderr:
            print(f"stderr: {result.stderr.decode('utf-8')}")

        # Step 2: Rename symbols with objcopy
        # The ld command creates symbols like:
        #   _binary_embedded_<name>_cubin_start
        #   _binary_embedded_<name>_cubin_end
        # We rename them to:
        #   __tvm_ffi__cubin_<name>
        #   __tvm_ffi__cubin_<name>_end
        # Note: We don't localize yet - that happens after merging
        cubin_obj_renamed = tmp_path / f"cubin_{name}_renamed.o"
        objcopy_cmd = [
            "objcopy",
            "--rename-section",
            ".data=.rodata,alloc,load,readonly,data,contents",
            "--redefine-sym",
            f"_binary_embedded_{name}_cubin_start=__tvm_ffi__cubin_{name}",
            "--redefine-sym",
            f"_binary_embedded_{name}_cubin_end=__tvm_ffi__cubin_{name}_end",
            str(cubin_obj),
            str(cubin_obj_renamed),
        ]

        if verbose:
            print("\n[Step 2/4] Renaming CUBIN symbols with objcopy")
            print(f"Command: {' '.join(objcopy_cmd)}")

        result = subprocess.run(objcopy_cmd, capture_output=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8")
            stdout = result.stdout.decode("utf-8")
            raise RuntimeError(
                f"objcopy failed with status {result.returncode}\n"
                f"Command: {' '.join(objcopy_cmd)}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

        if verbose and result.stdout:
            print(f"stdout: {result.stdout.decode('utf-8')}")
        if verbose and result.stderr:
            print(f"stderr: {result.stderr.decode('utf-8')}")

        # Step 3: Merge input object file with CUBIN object using ld -r (relocatable link)
        ld_merge_cmd = [
            "ld",
            "-r",
            "-o",
            str(output_obj_path),
            str(input_obj_path),
            str(cubin_obj_renamed),
        ]

        if verbose:
            print("\n[Step 3/4] Merging objects with ld (relocatable link)")
            print(f"Command: {' '.join(ld_merge_cmd)}")

        result = subprocess.run(ld_merge_cmd, capture_output=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8")
            stdout = result.stdout.decode("utf-8")
            raise RuntimeError(
                f"ld (merge) failed with status {result.returncode}\n"
                f"Command: {' '.join(ld_merge_cmd)}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

        if verbose and result.stdout:
            print(f"stdout: {result.stdout.decode('utf-8')}")
        if verbose and result.stderr:
            print(f"stderr: {result.stderr.decode('utf-8')}")

        # Step 4: Localize CUBIN symbols to prevent conflicts across object files
        # We do this after merging so the C++ code can reference the symbols during the link
        objcopy_localize_cmd = [
            "objcopy",
            "--localize-symbol",
            f"__tvm_ffi__cubin_{name}",
            "--localize-symbol",
            f"__tvm_ffi__cubin_{name}_end",
            str(output_obj_path),
        ]

        if verbose:
            print("\n[Step 4/4] Localizing CUBIN symbols")
            print(f"Command: {' '.join(objcopy_localize_cmd)}")

        result = subprocess.run(objcopy_localize_cmd, capture_output=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8")
            stdout = result.stdout.decode("utf-8")
            raise RuntimeError(
                f"objcopy (localize symbols) failed with status {result.returncode}\n"
                f"Command: {' '.join(objcopy_localize_cmd)}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

        if verbose and result.stdout:
            print(f"stdout: {result.stdout.decode('utf-8')}")
        if verbose and result.stderr:
            print(f"stderr: {result.stderr.decode('utf-8')}")

    if verbose:
        print(f"\n✓ Successfully created combined object file: {output_obj_path}")
        print("  Contains:")
        print(f"    - Original code from {input_obj_path.name}")
        print("    - Embedded CUBIN data with local symbols:")
        print(f"        __tvm_ffi__cubin_{name} (local)")
        print(f"        __tvm_ffi__cubin_{name}_end (local)")
        print("    - .note.GNU-stack section (non-executable stack)")


def main() -> int:
    """Main entry point for the embed_cubin utility."""  # noqa: D401
    parser = argparse.ArgumentParser(
        prog="python -m tvm_ffi.utils.embed_cubin",
        description="Embed CUBIN data into existing object files that use TVM_FFI_EMBED_CUBIN macro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m tvm_ffi.utils.embed_cubin \\
      --output-obj new.o \\
      --input-obj old.o \\
      --cubin kernel.cubin \\
      --name my_kernels

  # With verbose output
  python -m tvm_ffi.utils.embed_cubin \\
      --output-obj new.o \\
      --input-obj old.o \\
      --cubin kernel.cubin \\
      --name my_kernels \\
      --verbose

Workflow:
  1. Compile C++ code that uses TVM_FFI_EMBED_CUBIN to create old.o
  2. Compile CUDA kernel to CUBIN (e.g., using nvcc or NVRTC)
  3. Use this tool to merge them into new.o
  4. Link new.o into your final shared library

Usage in C++ code (source compiled to old.o):
  TVM_FFI_EMBED_CUBIN(my_kernels);
  auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_kernels, "kernel_name");

Requirements:
  - GNU binutils (ld and objcopy) must be available in PATH
  - Linux/Unix platform (Windows uses different embedding mechanisms)
        """,
    )

    parser.add_argument(
        "--output-obj",
        type=str,
        required=True,
        metavar="PATH",
        help="Output object file path (e.g., new.o)",
    )

    parser.add_argument(
        "--input-obj",
        type=str,
        required=True,
        metavar="PATH",
        help="Input object file path containing TVM_FFI_EMBED_CUBIN usage (e.g., old.o)",
    )

    parser.add_argument(
        "--cubin",
        type=str,
        required=True,
        metavar="PATH",
        help="Input CUBIN file path (e.g., kernel.cubin)",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        metavar="NAME",
        help="Name used in TVM_FFI_EMBED_CUBIN macro (e.g., my_kernels)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed command output",
    )

    args = parser.parse_args()

    try:
        cubin_path = Path(args.cubin).resolve()
        input_obj_path = Path(args.input_obj).resolve()
        output_obj_path = Path(args.output_obj).resolve()

        # Create output directory if it doesn't exist
        output_obj_path.parent.mkdir(parents=True, exist_ok=True)

        embed_cubin(cubin_path, input_obj_path, output_obj_path, args.name, args.verbose)

        if not args.verbose:
            print(f"✓ Created {output_obj_path}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
