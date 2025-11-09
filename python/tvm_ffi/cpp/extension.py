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
"""Build and load C++/CUDA sources into a tvm_ffi Module using Ninja."""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path, find_libtvm_ffi
from tvm_ffi.module import Module, load_module
from tvm_ffi.utils import FileLock

IS_WINDOWS = sys.platform == "win32"


def _hash_sources(
    cpp_source: str | None,
    cuda_source: str | None,
    cpp_files: Sequence[str] | None,
    cuda_files: Sequence[str] | None,
    functions: Sequence[str] | Mapping[str, str],
    extra_cflags: Sequence[str],
    extra_cuda_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    extra_include_paths: Sequence[str],
) -> str:
    """Generate a unique hash for the given sources and functions."""
    m = hashlib.sha256()

    def _maybe_hash_string(source: str | None) -> None:
        if source is not None:
            m.update(source.encode("utf-8"))

    def _hash_sequence(seq: Sequence[str]) -> None:
        for item in seq:
            m.update(item.encode("utf-8"))

    def _hash_mapping(mapping: Mapping[str, str]) -> None:
        for key in sorted(mapping):
            m.update(key.encode("utf-8"))
            m.update(mapping[key].encode("utf-8"))

    _maybe_hash_string(cpp_source)
    _maybe_hash_string(cuda_source)
    _hash_sequence(sorted(cpp_files or []))
    _hash_sequence(sorted(cuda_files or []))
    if isinstance(functions, Mapping):
        _hash_mapping(functions)
    else:
        _hash_sequence(sorted(functions))
    _hash_sequence(extra_cflags)
    _hash_sequence(extra_cuda_cflags)
    _hash_sequence(extra_ldflags)
    _hash_sequence(extra_include_paths)
    return m.hexdigest()[:16]


def _maybe_write(path: str, content: str) -> None:
    """Write content to path if it does not already exist with the same content."""
    p = Path(path)
    if p.exists():
        with p.open() as f:
            existing_content = f.read()
        if existing_content == content:
            return
    with p.open("w") as f:
        f.write(content)


@functools.lru_cache
def _find_cuda_home() -> str:
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = str(Path(nvcc_path).parent.parent)
        else:
            # Guess #3
            if IS_WINDOWS:
                cuda_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
                cuda_homes = list(cuda_root.glob("v*.*"))
                if len(cuda_homes) == 0:
                    raise RuntimeError(
                        "Could not find CUDA installation. Please set CUDA_HOME environment variable."
                    )
                cuda_home = str(cuda_homes[0])
            else:
                cuda_home = "/usr/local/cuda"
            if not Path(cuda_home).exists():
                raise RuntimeError(
                    "Could not find CUDA installation. Please set CUDA_HOME environment variable."
                )
    return cuda_home


def _get_cuda_target() -> str:
    """Get the CUDA target architecture flag."""
    if "TVM_FFI_CUDA_ARCH_LIST" in os.environ:
        arch_list = os.environ["TVM_FFI_CUDA_ARCH_LIST"].split()  # e.g., "8.9 9.0a"
        flags = []
        for arch in arch_list:
            if len(arch.split(".")) != 2:
                raise ValueError(f"Invalid CUDA architecture: {arch}")
            major, minor = arch.split(".")
            flags.append(f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}")
        return " ".join(flags)
    else:
        try:
            status = subprocess.run(
                args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                check=True,
            )
            compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
            major, minor = compute_cap.split(".")
            return f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
        except Exception:
            # fallback to a reasonable default
            return "-gencode=arch=compute_70,code=sm_70"


def _run_command_in_dev_prompt(
    args: list[str],
    cwd: str | os.PathLike[str],
    capture_output: bool,
) -> subprocess.CompletedProcess:
    """Locates the Developer Command Prompt and runs a command within its environment."""
    try:
        # Path to vswhere.exe
        vswhere_path = str(
            Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))
            / "Microsoft Visual Studio"
            / "Installer"
            / "vswhere.exe"
        )

        if not Path(vswhere_path).exists():
            raise FileNotFoundError("vswhere.exe not found.")

        # Find the Visual Studio installation path
        vs_install_path = subprocess.run(
            [
                vswhere_path,
                "-latest",
                "-prerelease",
                "-products",
                "*",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        if not vs_install_path:
            raise FileNotFoundError("No Visual Studio installation found.")

        # Construct the path to the VsDevCmd.bat file
        vsdevcmd_path = str(Path(vs_install_path) / "Common7" / "Tools" / "VsDevCmd.bat")

        if not Path(vsdevcmd_path).exists():
            raise FileNotFoundError(f"VsDevCmd.bat not found at: {vsdevcmd_path}")

        # Use cmd.exe to run the batch file and then your command.
        # The /k flag keeps the command prompt open after the batch file runs.
        # The "&" symbol chains the commands.
        cmd_command = '"{vsdevcmd_path}" -arch=x64 & {command}'.format(
            vsdevcmd_path=vsdevcmd_path, command=" ".join(args)
        )

        # Execute the command in a new shell
        return subprocess.run(
            cmd_command, check=False, cwd=cwd, capture_output=capture_output, shell=True
        )

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "Failed to run the following command in MSVC developer environment: {}".format(
                " ".join(args)
            )
        ) from e


def _generate_ninja_build(  # noqa: PLR0915
    name: str,
    with_cuda: bool,
    extra_cflags: Sequence[str],
    extra_cuda_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    extra_include_paths: Sequence[str],
    cpp_files: Sequence[str],
    cuda_files: Sequence[str],
) -> str:
    """Generate the content of build.ninja for building the module."""
    default_include_paths = [find_include_path(), find_dlpack_include_path()]

    tvm_ffi_lib = find_libtvm_ffi()
    tvm_ffi_lib_path = str(Path(tvm_ffi_lib).parent)
    tvm_ffi_lib_name = Path(tvm_ffi_lib).stem
    if IS_WINDOWS:
        default_cflags = [
            "/std:c++17",
            "/MD",
            "/wd4819",
            "/wd4251",
            "/wd4244",
            "/wd4267",
            "/wd4275",
            "/wd4018",
            "/wd4190",
            "/wd4624",
            "/wd4067",
            "/wd4068",
            "/EHsc",
        ]
        default_cuda_cflags = ["-Xcompiler", "/std:c++17", "/O2"]
        default_ldflags = [
            "/DLL",
            f"/LIBPATH:{tvm_ffi_lib_path}",
            f"{tvm_ffi_lib_name}.lib",
        ]
    else:
        default_cflags = ["-std=c++17", "-fPIC", "-O2"]
        default_cuda_cflags = ["-Xcompiler", "-fPIC", "-std=c++17", "-O2"]
        default_ldflags = ["-shared", f"-L{tvm_ffi_lib_path}", "-ltvm_ffi"]

        if with_cuda:
            # determine the compute capability of the current GPU
            default_cuda_cflags += [_get_cuda_target()]
            default_ldflags += [
                "-L{}".format(str(Path(_find_cuda_home()) / "lib64")),
                "-lcudart",
            ]

    cflags = default_cflags + [flag.strip() for flag in extra_cflags]
    cuda_cflags = default_cuda_cflags + [flag.strip() for flag in extra_cuda_cflags]
    ldflags = default_ldflags + [flag.strip() for flag in extra_ldflags]
    include_paths = default_include_paths + [
        str(Path(path).resolve()) for path in extra_include_paths
    ]

    # append include paths
    for path in include_paths:
        cflags.append("-I{}".format(path.replace(":", "$:")))
        cuda_cflags.append("-I{}".format(path.replace(":", "$:")))

    # flags
    ninja: list[str] = []
    ninja.append("ninja_required_version = 1.3")
    ninja.append("cxx = {}".format(os.environ.get("CXX", "cl" if IS_WINDOWS else "c++")))
    ninja.append("cflags = {}".format(" ".join(cflags)))
    if with_cuda:
        ninja.append("nvcc = {}".format(str(Path(_find_cuda_home()) / "bin" / "nvcc")))
        ninja.append("cuda_cflags = {}".format(" ".join(cuda_cflags)))
    ninja.append("ldflags = {}".format(" ".join(ldflags)))

    # rules
    ninja.append("")
    ninja.append("rule compile")
    if IS_WINDOWS:
        ninja.append("  command = $cxx /showIncludes $cflags -c $in /Fo$out")
        ninja.append("  deps = msvc")
    else:
        ninja.append("  depfile = $out.d")
        ninja.append("  deps = gcc")
        ninja.append("  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out")
    ninja.append("")

    if with_cuda:
        ninja.append("rule compile_cuda")
        ninja.append("  depfile = $out.d")
        ninja.append("  deps = gcc")
        ninja.append(
            "  command = $nvcc --generate-dependencies-with-compile --dependency-output $out.d $cuda_cflags -c $in -o $out"
        )
        ninja.append("")

    ninja.append("rule link")
    if IS_WINDOWS:
        ninja.append("  command = $cxx $in /link $ldflags /out:$out")
    else:
        ninja.append("  command = $cxx $in $ldflags -o $out")
    ninja.append("")

    # build targets
    link_files: list[str] = []
    for i, cpp_path in enumerate(sorted(cpp_files)):
        obj_name = f"cpp_{i}.o"
        ninja.append("build {}: compile {}".format(obj_name, cpp_path.replace(":", "$:")))
        link_files.append(obj_name)

    for i, cuda_path in enumerate(sorted(cuda_files)):
        obj_name = f"cuda_{i}.o"
        ninja.append("build {}: compile_cuda {}".format(obj_name, cuda_path.replace(":", "$:")))
        link_files.append(obj_name)

    # Use appropriate extension based on platform
    ext = ".dll" if IS_WINDOWS else ".so"
    link_name = " ".join(link_files)
    ninja.append(f"build {name}{ext}: link {link_name}")
    ninja.append("")

    # default target
    ninja.append(f"default {name}{ext}")
    ninja.append("")
    return "\n".join(ninja)


def build_ninja(build_dir: str) -> None:
    """Build the module in the given build directory using ninja."""
    command = ["ninja", "-v"]
    num_workers = os.environ.get("MAX_JOBS", None)
    if num_workers is not None:
        command += ["-j", num_workers]
    if IS_WINDOWS:
        status = _run_command_in_dev_prompt(args=command, cwd=build_dir, capture_output=True)
    else:
        status = subprocess.run(check=False, args=command, cwd=build_dir, capture_output=True)
    if status.returncode != 0:
        msg = [f"ninja exited with status {status.returncode}"]
        encoding = "oem" if IS_WINDOWS else "utf-8"
        if status.stdout:
            msg.append(f"stdout:\n{status.stdout.decode(encoding)}")
        if status.stderr:
            msg.append(f"stderr:\n{status.stderr.decode(encoding)}")

        raise RuntimeError("\n".join(msg))


def _decorate_with_tvm_ffi(source: str, functions: Mapping[str, str]) -> str:
    """Decorate the given source code with TVM FFI export macros."""
    sources = [
        "#include <tvm/ffi/container/tensor.h>",
        "#include <tvm/ffi/dtype.h>",
        "#include <tvm/ffi/error.h>",
        "#include <tvm/ffi/extra/c_env_api.h>",
        "#include <tvm/ffi/function.h>",
        "",
        source,
    ]

    for func_name, func_doc in functions.items():
        sources.append(f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({func_name}, {func_name});")
        _ = func_doc  # todo: add support to embed function docstring to the tvm ffi functions.

    sources.append("")

    return "\n".join(sources)


def _str_seq2list(seq: Sequence[str] | str | None) -> list[str]:
    if seq is None:
        return []
    elif isinstance(seq, str):
        return [seq]
    else:
        return list(seq)


def _build_impl(
    name: str,
    cpp_files: Sequence[str] | str | None,
    cuda_files: Sequence[str] | str | None,
    extra_cflags: Sequence[str] | None,
    extra_cuda_cflags: Sequence[str] | None,
    extra_ldflags: Sequence[str] | None,
    extra_include_paths: Sequence[str] | None,
    build_directory: str | None,
    need_lock: bool = True,
) -> str:
    """Real implementation of build function."""
    # need to resolve the path to make it unique
    cpp_path_list = [str(Path(p).resolve()) for p in _str_seq2list(cpp_files)]
    cuda_path_list = [str(Path(p).resolve()) for p in _str_seq2list(cuda_files)]
    with_cpp = bool(cpp_path_list)
    with_cuda = bool(cuda_path_list)
    assert with_cpp or with_cuda, "Either cpp_files or cuda_files must be provided."

    extra_ldflags_list = list(extra_ldflags) if extra_ldflags is not None else []
    extra_cflags_list = list(extra_cflags) if extra_cflags is not None else []
    extra_cuda_cflags_list = list(extra_cuda_cflags) if extra_cuda_cflags is not None else []
    extra_include_paths_list = list(extra_include_paths) if extra_include_paths is not None else []

    build_dir: Path
    if build_directory is None:
        cache_dir = os.environ.get("TVM_FFI_CACHE_DIR", str(Path("~/.cache/tvm-ffi").expanduser()))
        source_hash: str = _hash_sources(
            None,
            None,
            cpp_path_list,
            cuda_path_list,
            {},
            extra_cflags_list,
            extra_cuda_cflags_list,
            extra_ldflags_list,
            extra_include_paths_list,
        )
        build_dir = Path(cache_dir).expanduser() / f"{name}_{source_hash}"
    else:
        build_dir = Path(build_directory).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    # generate build.ninja
    ninja_source = _generate_ninja_build(
        name=name,
        with_cuda=with_cuda,
        extra_cflags=extra_cflags_list,
        extra_cuda_cflags=extra_cuda_cflags_list,
        extra_ldflags=extra_ldflags_list,
        extra_include_paths=extra_include_paths_list,
        cpp_files=cpp_path_list,
        cuda_files=cuda_path_list,
    )

    # may not hold lock when build_directory is specified, prevent deadlock
    with FileLock(str(build_dir / "lock")) if need_lock else nullcontext():
        # write build.ninja if it does not already exist
        _maybe_write(str(build_dir / "build.ninja"), ninja_source)
        # build the module
        build_ninja(str(build_dir))
        # Use appropriate extension based on platform
        ext = ".dll" if IS_WINDOWS else ".so"
        return str((build_dir / f"{name}{ext}").resolve())


def build_inline(
    name: str,
    *,
    cpp_sources: Sequence[str] | str | None = None,
    cuda_sources: Sequence[str] | str | None = None,
    functions: Mapping[str, str] | Sequence[str] | str | None = None,
    extra_cflags: Sequence[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    extra_ldflags: Sequence[str] | None = None,
    extra_include_paths: Sequence[str] | None = None,
    build_directory: str | None = None,
) -> str:
    """Compile and build a C++/CUDA module from inline source code.

    This function compiles the given C++ and/or CUDA source code into a shared library. Both ``cpp_sources`` and
    ``cuda_sources`` are compiled to an object file, and then linked together into a shared library. It's possible to only
    provide cpp_sources or cuda_sources. The path to the compiled shared library is returned.

    The ``functions`` parameter is used to specify which functions in the source code should be exported to the tvm ffi
    module. It can be a mapping, a sequence, or a single string. When a mapping is given, the keys are the names of the
    exported functions, and the values are docstrings for the functions. When a sequence of string is given, they are
    the function names needed to be exported, and the docstrings are set to empty strings. A single function name can
    also be given as a string, indicating that only one function is to be exported.

    Extra compiler and linker flags can be provided via the ``extra_cflags``, ``extra_cuda_cflags``, and ``extra_ldflags``
    parameters. The default flags are generally sufficient for most use cases, but you may need to provide additional
    flags for your specific use case.

    The include dir of tvm ffi and dlpack are used by default for the compiler to find the headers. Thus, you can
    include any header from tvm ffi in your source code. You can also provide additional include paths via the
    ``extra_include_paths`` parameter and include custom headers in your source code.

    The compiled shared library is cached in a cache directory to avoid recompilation. The `build_directory` parameter
    is provided to specify the build directory. If not specified, a default tvm ffi cache directory will be used.
    The default cache directory can be specified via the `TVM_FFI_CACHE_DIR` environment variable. If not specified,
    the default cache directory is ``~/.cache/tvm-ffi``.

    Parameters
    ----------
    name
        The name of the tvm ffi module.
    cpp_sources
        The C++ source code. It can be a list of sources or a single source.
    cuda_sources
        The CUDA source code. It can be a list of sources or a single source.
    functions
        The functions in cpp_sources or cuda_source that will be exported to the tvm ffi module. When a mapping is
        given, the keys are the names of the exported functions, and the values are docstrings for the functions. When
        a sequence or a single string is given, they are the functions needed to be exported, and the docstrings are set
        to empty strings. A single function name can also be given as a string. When cpp_sources is given, the functions
        must be declared (not necessarily defined) in the cpp_sources. When cpp_sources is not given, the functions
        must be defined in the cuda_sources. If not specified, no function will be exported.
    extra_cflags
        The extra compiler flags for C++ compilation.
        The default flags are:

        - On Linux/macOS: ['-std=c++17', '-fPIC', '-O2']
        - On Windows: ['/std:c++17', '/O2']

    extra_cuda_cflags
        The extra compiler flags for CUDA compilation.

    extra_ldflags
        The extra linker flags.
        The default flags are:

        - On Linux/macOS: ['-shared']
        - On Windows: ['/DLL']

    extra_include_paths
        The extra include paths.

    build_directory
        The build directory. If not specified, a default tvm ffi cache directory will be used. By default, the
        cache directory is ``~/.cache/tvm-ffi``. You can also set the ``TVM_FFI_CACHE_DIR`` environment variable to
        specify the cache directory.

    Returns
    -------
    lib_path: str
        The path to the built shared library.

    Example
    -------

    .. code-block:: python

        import torch
        from tvm_ffi import Module
        import tvm_ffi.cpp

        # define the cpp source code
        cpp_source = '''
             void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
               // implementation of a library function
               TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
               DLDataType f32_dtype{kDLFloat, 32, 1};
               TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
               TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
               TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
               TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
               for (int i = 0; i < x.size(0); ++i) {
                 static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
               }
             }
        '''

        # compile the cpp source code and load the module
        lib_path: str = tvm_ffi.cpp.build_inline(
            name='hello',
            cpp_sources=cpp_source,
            functions='add_one_cpu'
        )

        # load the module
        mod: Module = tvm_ffi.load_module(lib_path)

        # use the function from the loaded module to perform
        x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        y = torch.empty_like(x)
        mod.add_one_cpu(x, y)
        torch.testing.assert_close(x + 1, y)

    """
    cpp_source_list = _str_seq2list(cpp_sources)
    cpp_source = "\n".join(cpp_source_list)
    with_cpp = bool(cpp_source_list)
    del cpp_source_list

    cuda_source_list = _str_seq2list(cuda_sources)
    cuda_source = "\n".join(cuda_source_list)
    with_cuda = bool(cuda_source_list)
    del cuda_source_list

    extra_ldflags_list = list(extra_ldflags) if extra_ldflags is not None else []
    extra_cflags_list = list(extra_cflags) if extra_cflags is not None else []
    extra_cuda_cflags_list = list(extra_cuda_cflags) if extra_cuda_cflags is not None else []
    extra_include_paths_list = list(extra_include_paths) if extra_include_paths is not None else []

    # add function registration code to sources
    if functions is None:
        function_map: dict[str, str] = {}
    elif isinstance(functions, str):
        function_map = {functions: ""}
    elif isinstance(functions, Mapping):
        function_map = dict(functions)
    else:
        function_map = {name: "" for name in functions}

    if with_cpp:
        cpp_source = _decorate_with_tvm_ffi(cpp_source, function_map)
        cuda_source = _decorate_with_tvm_ffi(cuda_source, {})
    else:
        cpp_source = _decorate_with_tvm_ffi(cpp_source, {})
        cuda_source = _decorate_with_tvm_ffi(cuda_source, function_map)

    # determine the cache dir for the built module
    build_dir: Path
    if build_directory is None:
        cache_dir = os.environ.get("TVM_FFI_CACHE_DIR", str(Path("~/.cache/tvm-ffi").expanduser()))
        source_hash: str = _hash_sources(
            cpp_source,
            cuda_source,
            None,
            None,
            function_map,
            extra_cflags_list,
            extra_cuda_cflags_list,
            extra_ldflags_list,
            extra_include_paths_list,
        )
        build_dir = Path(cache_dir).expanduser() / f"{name}_{source_hash}"
    else:
        build_dir = Path(build_directory).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    cpp_file = str((build_dir / "main.cpp").resolve())
    cuda_file = str((build_dir / "cuda.cu").resolve())

    with FileLock(str(build_dir / "lock")):
        # write source files if they do not already exist
        _maybe_write(cpp_file, cpp_source)
        if with_cuda:
            _maybe_write(cuda_file, cuda_source)

        return _build_impl(
            name=name,
            cpp_files=[cpp_file] if with_cpp else [],
            cuda_files=[cuda_file] if with_cuda else [],
            extra_cflags=extra_cflags_list,
            extra_cuda_cflags=extra_cuda_cflags_list,
            extra_ldflags=extra_ldflags_list,
            extra_include_paths=extra_include_paths_list,
            build_directory=str(build_dir),
            need_lock=False,  # already hold the lock
        )


def load_inline(
    name: str,
    *,
    cpp_sources: Sequence[str] | str | None = None,
    cuda_sources: Sequence[str] | str | None = None,
    functions: Mapping[str, str] | Sequence[str] | str | None = None,
    extra_cflags: Sequence[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    extra_ldflags: Sequence[str] | None = None,
    extra_include_paths: Sequence[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    """Compile, build and load a C++/CUDA module from inline source code.

    This function compiles the given C++ and/or CUDA source code into a shared library. Both ``cpp_sources`` and
    ``cuda_sources`` are compiled to an object file, and then linked together into a shared library. It's possible to only
    provide cpp_sources or cuda_sources.

    The ``functions`` parameter is used to specify which functions in the source code should be exported to the tvm ffi
    module. It can be a mapping, a sequence, or a single string. When a mapping is given, the keys are the names of the
    exported functions, and the values are docstrings for the functions. When a sequence of string is given, they are
    the function names needed to be exported, and the docstrings are set to empty strings. A single function name can
    also be given as a string, indicating that only one function is to be exported.

    Extra compiler and linker flags can be provided via the ``extra_cflags``, ``extra_cuda_cflags``, and ``extra_ldflags``
    parameters. The default flags are generally sufficient for most use cases, but you may need to provide additional
    flags for your specific use case.

    The include dir of tvm ffi and dlpack are used by default for the compiler to find the headers. Thus, you can
    include any header from tvm ffi in your source code. You can also provide additional include paths via the
    ``extra_include_paths`` parameter and include custom headers in your source code.

    The compiled shared library is cached in a cache directory to avoid recompilation. The `build_directory` parameter
    is provided to specify the build directory. If not specified, a default tvm ffi cache directory will be used.
    The default cache directory can be specified via the `TVM_FFI_CACHE_DIR` environment variable. If not specified,
    the default cache directory is ``~/.cache/tvm-ffi``.

    Parameters
    ----------
    name: str
        The name of the tvm ffi module.
    cpp_sources: Sequence[str] | str, optional
        The C++ source code. It can be a list of sources or a single source.
    cuda_sources: Sequence[str] | str, optional
        The CUDA source code. It can be a list of sources or a single source.
    functions: Mapping[str, str] | Sequence[str] | str, optional
        The functions in cpp_sources or cuda_source that will be exported to the tvm ffi module. When a mapping is
        given, the keys are the names of the exported functions, and the values are docstrings for the functions. When
        a sequence or a single string is given, they are the functions needed to be exported, and the docstrings are set
        to empty strings. A single function name can also be given as a string. When cpp_sources is given, the functions
        must be declared (not necessarily defined) in the cpp_sources. When cpp_sources is not given, the functions
        must be defined in the cuda_sources. If not specified, no function will be exported.
    extra_cflags: Sequence[str], optional
        The extra compiler flags for C++ compilation.
        The default flags are:

        - On Linux/macOS: ['-std=c++17', '-fPIC', '-O2']
        - On Windows: ['/std:c++17', '/O2']

    extra_cuda_cflags: Sequence[str], optional
        The extra compiler flags for CUDA compilation.

    extra_ldflags: Sequence[str], optional
        The extra linker flags.
        The default flags are:

        - On Linux/macOS: ['-shared']
        - On Windows: ['/DLL']

    extra_include_paths: Sequence[str], optional
        The extra include paths.

    build_directory: str, optional
        The build directory. If not specified, a default tvm ffi cache directory will be used. By default, the
        cache directory is ``~/.cache/tvm-ffi``. You can also set the ``TVM_FFI_CACHE_DIR`` environment variable to
        specify the cache directory.

    Returns
    -------
    mod: Module
        The loaded tvm ffi module.


    Example
    -------

    .. code-block:: python

        import torch
        from tvm_ffi import Module
        import tvm_ffi.cpp

        # define the cpp source code
        cpp_source = '''
             void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
               // implementation of a library function
               TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
               DLDataType f32_dtype{kDLFloat, 32, 1};
               TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
               TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
               TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
               TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
               for (int i = 0; i < x.size(0); ++i) {
                 static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
               }
             }
        '''

        # compile the cpp source code and load the module
        mod: Module = tvm_ffi.cpp.load_inline(
            name='hello',
            cpp_sources=cpp_source,
            functions='add_one_cpu'
        )

        # use the function from the loaded module to perform
        x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        y = torch.empty_like(x)
        mod.add_one_cpu(x, y)
        torch.testing.assert_close(x + 1, y)

    """
    return load_module(
        build_inline(
            name=name,
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            functions=functions,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=extra_include_paths,
            build_directory=build_directory,
        )
    )


def build(
    name: str,
    *,
    cpp_files: Sequence[str] | str | None = None,
    cuda_files: Sequence[str] | str | None = None,
    extra_cflags: Sequence[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    extra_ldflags: Sequence[str] | None = None,
    extra_include_paths: Sequence[str] | None = None,
    build_directory: str | None = None,
) -> str:
    """Compile and build a C++/CUDA module from source files.

    This function compiles the given C++ and/or CUDA source files into a shared library. Both ``cpp_files`` and
    ``cuda_files`` are compiled to object files, and then linked together into a shared library. It's possible to only
    provide cpp_files or cuda_files. The path to the compiled shared library is returned.

    Note that this function does not automatically export functions to the tvm ffi module. You need to
    manually use the TVM FFI export macros (e.g., ``TVM_FFI_DLL_EXPORT_TYPED_FUNC``) in your source files to export
    functions. This gives you more control over which functions are exported and how they are exported.

    Extra compiler and linker flags can be provided via the ``extra_cflags``, ``extra_cuda_cflags``, and ``extra_ldflags``
    parameters. The default flags are generally sufficient for most use cases, but you may need to provide additional
    flags for your specific use case.

    The include dir of tvm ffi and dlpack are used by default for the compiler to find the headers. Thus, you can
    include any header from tvm ffi in your source files. You can also provide additional include paths via the
    ``extra_include_paths`` parameter and include custom headers in your source code.

    The compiled shared library is cached in a cache directory to avoid recompilation. The `build_directory` parameter
    is provided to specify the build directory. If not specified, a default tvm ffi cache directory will be used.
    The default cache directory can be specified via the `TVM_FFI_CACHE_DIR` environment variable. If not specified,
    the default cache directory is ``~/.cache/tvm-ffi``.

    Parameters
    ----------
    name
        The name of the tvm ffi module.
    cpp_files
        The C++ source files to compile. It can be a list of file paths or a single file path. Both absolute and
        relative paths are supported.
    cuda_files
        The CUDA source files to compile. It can be a list of file paths or a single file path. Both absolute and
        relative paths are supported.
    extra_cflags
        The extra compiler flags for C++ compilation.
        The default flags are:

        - On Linux/macOS: ['-std=c++17', '-fPIC', '-O2']
        - On Windows: ['/std:c++17', '/MD', '/O2']

    extra_cuda_cflags
        The extra compiler flags for CUDA compilation.
        The default flags are:

        - ['-Xcompiler', '-fPIC', '-std=c++17', '-O2'] (Linux/macOS)
        - ['-Xcompiler', '/std:c++17', '/O2'] (Windows)

    extra_ldflags
        The extra linker flags.
        The default flags are:

        - On Linux/macOS: ['-shared', '-L<tvm_ffi_lib_path>', '-ltvm_ffi']
        - On Windows: ['/DLL', '/LIBPATH:<tvm_ffi_lib_path>', '<tvm_ffi_lib_name>.lib']

    extra_include_paths
        The extra include paths for header files. Both absolute and relative paths are supported.

    build_directory
        The build directory. If not specified, a default tvm ffi cache directory will be used. By default, the
        cache directory is ``~/.cache/tvm-ffi``. You can also set the ``TVM_FFI_CACHE_DIR`` environment variable to
        specify the cache directory.

    Returns
    -------
    lib_path: str
        The path to the built shared library.

    Example
    -------

    .. code-block:: python

        import torch
        from tvm_ffi import Module
        import tvm_ffi.cpp

        # Assume we have a C++ source file "my_ops.cpp" with the following content:
        # ```cpp
        # #include <tvm/ffi/container/tensor.h>
        # #include <tvm/ffi/dtype.h>
        # #include <tvm/ffi/error.h>
        # #include <tvm/ffi/extra/c_env_api.h>
        # #include <tvm/ffi/function.h>
        #
        # void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
        #   TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
        #   DLDataType f32_dtype{kDLFloat, 32, 1};
        #   TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
        #   TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
        #   TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
        #   TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
        #   for (int i = 0; i < x.size(0); ++i) {
        #     static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
        #   }
        # }
        #
        # TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, add_one_cpu);
        # ```

        # compile the cpp source file and get the library path
        lib_path: str = tvm_ffi.cpp.build(
            name='my_ops',
            cpp_files='my_ops.cpp'
        )

        # load the module
        mod: Module = tvm_ffi.load_module(lib_path)

        # use the function from the loaded module
        x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        y = torch.empty_like(x)
        mod.add_one_cpu(x, y)
        torch.testing.assert_close(x + 1, y)

    """
    return _build_impl(
        name=name,
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        need_lock=True,
    )


def load(
    name: str,
    *,
    cpp_files: Sequence[str] | str | None = None,
    cuda_files: Sequence[str] | str | None = None,
    extra_cflags: Sequence[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    extra_ldflags: Sequence[str] | None = None,
    extra_include_paths: Sequence[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    """Compile, build and load a C++/CUDA module from source files.

    This function compiles the given C++ and/or CUDA source files into a shared library and loads it as a tvm ffi
    module. Both ``cpp_files`` and ``cuda_files`` are compiled to object files, and then linked together into a shared
    library. It's possible to only provide cpp_files or cuda_files.

    Note that this function does not automatically export functions to the tvm ffi module. You need to
    manually use the TVM FFI export macros (e.g., :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC`) in your source files to export
    functions. This gives you more control over which functions are exported and how they are exported.

    Extra compiler and linker flags can be provided via the ``extra_cflags``, ``extra_cuda_cflags``, and ``extra_ldflags``
    parameters. The default flags are generally sufficient for most use cases, but you may need to provide additional
    flags for your specific use case.

    The include dir of tvm ffi and dlpack are used by default for the compiler to find the headers. Thus, you can
    include any header from tvm ffi in your source files. You can also provide additional include paths via the
    ``extra_include_paths`` parameter and include custom headers in your source code.

    The compiled shared library is cached in a cache directory to avoid recompilation. The `build_directory` parameter
    is provided to specify the build directory. If not specified, a default tvm ffi cache directory will be used.
    The default cache directory can be specified via the `TVM_FFI_CACHE_DIR` environment variable. If not specified,
    the default cache directory is ``~/.cache/tvm-ffi``.

    Parameters
    ----------
    name: str
        The name of the tvm ffi module.
    cpp_files: Sequence[str] | str, optional
        The C++ source files to compile. It can be a list of file paths or a single file path. Both absolute and
        relative paths are supported.
    cuda_files: Sequence[str] | str, optional
        The CUDA source files to compile. It can be a list of file paths or a single file path. Both absolute and
        relative paths are supported.
    extra_cflags: Sequence[str], optional
        The extra compiler flags for C++ compilation.
        The default flags are:

        - On Linux/macOS: ['-std=c++17', '-fPIC', '-O2']
        - On Windows: ['/std:c++17', '/MD', '/O2']

    extra_cuda_cflags: Sequence[str], optional
        The extra compiler flags for CUDA compilation.
        The default flags are:

        - ['-Xcompiler', '-fPIC', '-std=c++17', '-O2'] (Linux/macOS)
        - ['-Xcompiler', '/std:c++17', '/O2'] (Windows)

    extra_ldflags: Sequence[str], optional
        The extra linker flags.
        The default flags are:

        - On Linux/macOS: ['-shared', '-L<tvm_ffi_lib_path>', '-ltvm_ffi']
        - On Windows: ['/DLL', '/LIBPATH:<tvm_ffi_lib_path>', '<tvm_ffi_lib_name>.lib']

    extra_include_paths: Sequence[str], optional
        The extra include paths for header files. Both absolute and relative paths are supported.

    build_directory: str, optional
        The build directory. If not specified, a default tvm ffi cache directory will be used. By default, the
        cache directory is ``~/.cache/tvm-ffi``. You can also set the ``TVM_FFI_CACHE_DIR`` environment variable to
        specify the cache directory.

    Returns
    -------
    mod: Module
        The loaded tvm ffi module.


    Example
    -------

    .. code-block:: python

        import torch
        from tvm_ffi import Module
        import tvm_ffi.cpp

        # Assume we have a C++ source file "my_ops.cpp" with the following content:
        # ```cpp
        # #include <tvm/ffi/container/tensor.h>
        # #include <tvm/ffi/dtype.h>
        # #include <tvm/ffi/error.h>
        # #include <tvm/ffi/extra/c_env_api.h>
        # #include <tvm/ffi/function.h>
        #
        # void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
        #   TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
        #   DLDataType f32_dtype{kDLFloat, 32, 1};
        #   TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
        #   TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
        #   TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
        #   TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
        #   for (int i = 0; i < x.size(0); ++i) {
        #     static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
        #   }
        # }
        #
        # TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, add_one_cpu);
        # ```

        # compile the cpp source file and load the module
        mod: Module = tvm_ffi.cpp.load(
            name='my_ops',
            cpp_files='my_ops.cpp'
        )

        # use the function from the loaded module
        x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        y = torch.empty_like(x)
        mod.add_one_cpu(x, y)
        torch.testing.assert_close(x + 1, y)

    """
    return load_module(
        build(
            name=name,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=extra_include_paths,
            build_directory=build_directory,
        )
    )
