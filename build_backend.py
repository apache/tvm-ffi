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
"""Custom build backend that extends scikit-build-core with torch extension support."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from scikit_build_core import build as _orig

# Re-export the standard hooks (except those we override below)
build_sdist = _orig.build_sdist
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
build_editable = _orig.build_editable


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    """Get build requirements for wheel, conditionally including torch."""
    requires = _orig.get_requires_for_build_wheel(config_settings)

    # Only require torch if we're building the torch extension
    build_torch_ext = os.environ.get("TVM_FFI_BUILD_TORCH_EXT", "").lower()
    if build_torch_ext in ("1", "true", "yes", "on"):
        if "torch" not in requires:
            requires.append("torch")

    return requires


def get_requires_for_build_editable(config_settings: dict[str, Any] | None = None) -> list[str]:
    """Get build requirements for editable install, conditionally including torch."""
    requires = _orig.get_requires_for_build_editable(config_settings)

    # Only require torch if we're building the torch extension
    build_torch_ext = os.environ.get("TVM_FFI_BUILD_TORCH_EXT", "").lower()
    if build_torch_ext in ("1", "true", "yes", "on"):
        if "torch" not in requires:
            requires.append("torch")

    return requires


def build_torch_extension(wheel_directory: str, config_settings: dict[str, Any] | None = None) -> None:
    """Build the torch C++ extension at compile time.

    This function compiles the torch_c_dlpack_ext.cc extension using
    torch.utils.cpp_extension, similar to the JIT compilation but done
    at build time instead.

    The extension will only be built if the environment variable
    TVM_FFI_BUILD_TORCH_EXT is set to a truthy value (e.g., "1", "true", "yes").

    Parameters
    ----------
    wheel_directory
        The directory where the wheel will be built.
    config_settings
        Optional configuration settings.
    """
    # Check if the environment variable is set
    build_torch_ext = os.environ.get("TVM_FFI_BUILD_TORCH_EXT", "").lower()
    if build_torch_ext not in ("1", "true", "yes", "on"):
        print("=" * 80, file=sys.stderr)
        print("TVM_FFI_BUILD_TORCH_EXT not set - skipping torch extension build", file=sys.stderr)
        print("Set TVM_FFI_BUILD_TORCH_EXT=1 to enable torch extension compilation", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return

    print("=" * 80, file=sys.stderr)
    print("Attempting to build torch C++ extension...", file=sys.stderr)
    try:
        import torch
        from torch.utils import cpp_extension
        print(f"PyTorch {torch.__version__} found in build environment", file=sys.stderr)
    except ImportError:
        # Torch is optional, skip building the extension
        print("PyTorch not available in build environment - skipping torch extension build", file=sys.stderr)
        print("The extension will be compiled via JIT when first imported with torch", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return

    # Check if torch already has the API we're patching
    if hasattr(torch.Tensor, "__c_dlpack_exchange_api__"):
        # No need to build the extension
        return

    # Find the source file
    root_dir = Path(__file__).parent
    source_file = root_dir / "src" / "ffi" / "torch_c_dlpack_ext.cc"

    if not source_file.exists():
        print(f"Warning: torch extension source not found at {source_file}", file=sys.stderr)
        return

    # Find include paths - we need dlpack headers
    include_paths = []

    # Check for dlpack in 3rdparty
    dlpack_include = root_dir / "3rdparty" / "dlpack" / "include"
    if dlpack_include.is_dir():
        include_paths.append(str(dlpack_include))

    # Check for installed include path
    tvm_ffi_include = root_dir / "include"
    if tvm_ffi_include.is_dir():
        include_paths.append(str(tvm_ffi_include))

    # Setup output directory for the compiled extension
    build_dir = Path(wheel_directory).parent / "torch_ext_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    extra_cflags = ["-O3", "-std=c++17"]
    extra_ldflags = []

    # Add CUDA support if available
    if torch.cuda.is_available():
        try:
            include_paths += cpp_extension.include_paths("cuda")
            extra_cflags += ["-DBUILD_WITH_CUDA"]
        except Exception:
            # CUDA include paths might not be available in all environments
            pass

    # Compile the extension
    try:
        # Build the extension
        cpp_extension.load(
            name="torch_c_dlpack_ext",
            sources=[str(source_file)],
            build_directory=str(build_dir),
            extra_cflags=extra_cflags,
            extra_include_paths=include_paths,
            verbose=True,
        )

        print("Successfully built torch_c_dlpack_ext extension", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to build torch extension: {e}", file=sys.stderr)
        print("The package will still work, but torch DLPack optimization will use JIT compilation", file=sys.stderr)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build a wheel with torch extension support.

    This extends the standard scikit-build-core build_wheel to also
    compile the torch C++ extension at build time.

    Parameters
    ----------
    wheel_directory
        The directory where the wheel will be built.
    config_settings
        Optional configuration settings.
    metadata_directory
        Optional metadata directory.

    Returns
    -------
    wheel_name
        The name of the built wheel.
    """
    print("=" * 80, file=sys.stderr)
    print("CUSTOM BUILD BACKEND: build_wheel is being called", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # First, run the standard scikit-build-core build
    wheel_name = _orig.build_wheel(wheel_directory, config_settings, metadata_directory)

    # Then, try to build the torch extension and add it to the wheel
    try:
        import glob
        import shutil
        import zipfile
        import tempfile

        # Build the torch extension
        build_torch_extension(wheel_directory, config_settings)

        # Find the compiled .so file
        root_dir = Path(__file__).parent
        build_dir = Path(wheel_directory).parent / "torch_ext_build"
        so_pattern = str(build_dir / "torch_c_dlpack_ext*.so")
        so_files = glob.glob(so_pattern)

        if so_files:
            # Add the .so file to the wheel
            wheel_path = Path(wheel_directory) / wheel_name
            print(f"Adding torch extension to wheel: {wheel_path}", file=sys.stderr)

            # Create a temporary directory to extract and repack the wheel
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Extract the wheel
                with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir_path)

                # Copy the .so file to the package directory in the wheel
                for so_file in so_files:
                    # Use simple .so extension without platform tags
                    new_name = "torch_c_dlpack_ext.so"
                    dest = tmpdir_path / "tvm_ffi" / new_name
                    shutil.copy2(so_file, dest)
                    print(f"Added {so_file} to wheel as tvm_ffi/{new_name}", file=sys.stderr)

                # Repack the wheel
                wheel_path.unlink()  # Remove old wheel
                with zipfile.ZipFile(wheel_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                    for file in tmpdir_path.rglob('*'):
                        if file.is_file():
                            arcname = file.relative_to(tmpdir_path)
                            zip_out.write(file, arcname)

                print(f"Successfully added torch extension to wheel", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to add torch extension to wheel: {e}", file=sys.stderr)
        print("The package will still work, but torch DLPack optimization will use JIT compilation", file=sys.stderr)

    # Rename wheel and update metadata with local version if specified
    local_version = os.environ.get("TVM_FFI_LOCAL_VERSION", "")
    if local_version:
        try:
            import zipfile
            import tempfile

            wheel_path = Path(wheel_directory) / wheel_name

            # Parse wheel name: apache_tvm_ffi-0.1.0-cp312-abi3-linux_x86_64.whl
            # Transform to: apache_tvm_ffi-0.1.0+torch2.5.0-cp312-abi3-linux_x86_64.whl
            name_parts = wheel_name.split('-')
            if len(name_parts) >= 5:
                # Insert local version after the version number
                name_parts[1] = f"{name_parts[1]}+{local_version}"
                new_wheel_name = '-'.join(name_parts)

                # Update METADATA and RECORD files in the wheel
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)

                    # Extract wheel
                    with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir_path)

                    # Find and update METADATA file
                    dist_info_dirs = list(tmpdir_path.glob("*.dist-info"))
                    if dist_info_dirs:
                        old_dist_info = dist_info_dirs[0]
                        new_dist_info_name = old_dist_info.name.replace(name_parts[1].split('+')[0], name_parts[1])
                        new_dist_info = tmpdir_path / new_dist_info_name

                        # Update METADATA
                        metadata_file = old_dist_info / "METADATA"
                        if metadata_file.exists():
                            content = metadata_file.read_text()
                            # Update Version line
                            content = content.replace(
                                f"Version: {name_parts[1].split('+')[0]}",
                                f"Version: {name_parts[1]}"
                            )
                            metadata_file.write_text(content)

                        # Rename dist-info directory
                        old_dist_info.rename(new_dist_info)

                    # Repack wheel with new name
                    new_wheel_path = Path(wheel_directory) / new_wheel_name
                    with zipfile.ZipFile(new_wheel_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                        for file in tmpdir_path.rglob('*'):
                            if file.is_file():
                                arcname = file.relative_to(tmpdir_path)
                                zip_out.write(file, arcname)

                    # Remove old wheel
                    wheel_path.unlink()

                print(f"Renamed wheel to include local version: {new_wheel_name}", file=sys.stderr)
                return new_wheel_name
        except Exception as e:
            print(f"Warning: Failed to rename wheel with local version: {e}", file=sys.stderr)

    return wheel_name
