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
"""Unit tests for GPU backend (CUDA vs HIP) auto-detection.

These tests isolate the detection logic from the host machine by faking the
PyTorch build signal and the set of compilers visible on ``PATH``. They do not
compile anything, so they run anywhere regardless of the local toolchain.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Iterator

import pytest
from tvm_ffi.cpp import extension


@pytest.fixture(autouse=True)
def _clear_backend_cache() -> Iterator[None]:
    """Reset the lru_cache so each test observes a fresh detection."""
    extension._detect_gpu_backend.cache_clear()
    yield
    extension._detect_gpu_backend.cache_clear()


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch, *, cuda: str | None, hip: str | None
) -> None:
    """Register a fake ``torch`` module exposing the given version signals."""
    fake = types.ModuleType("torch")
    fake.version = types.SimpleNamespace(cuda=cuda, hip=hip)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)


def _hide_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import torch`` raise ImportError inside the detector."""
    # Mapping a module name to None in sys.modules forces an ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)


def _fake_which(available: set[str]) -> Callable[..., str | None]:
    """Return a shutil.which replacement that only finds names in ``available``."""

    def _which(cmd: str, *args: object, **kwargs: object) -> str | None:
        if cmd in available:
            return f"/usr/bin/{cmd}"
        return None

    return _which


@pytest.fixture
def _no_env_override(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("TVM_FFI_GPU_BACKEND", raising=False)
    yield


# --- Explicit environment / argument overrides ------------------------------


@pytest.mark.parametrize("value", ["cuda", "hip"])
def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TVM_FFI_GPU_BACKEND", value)
    # Even with a conflicting torch signal, the explicit override is honored.
    _install_fake_torch(monkeypatch, cuda="12.4" if value == "hip" else None, hip=None)
    assert extension._detect_gpu_backend() == value


def test_env_override_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TVM_FFI_GPU_BACKEND", "HIP")
    assert extension._detect_gpu_backend() == "hip"


def test_resolve_explicit_supported_backend() -> None:
    assert extension._resolve_gpu_backend("cuda") == "cuda"
    assert extension._resolve_gpu_backend("hip") == "hip"


def test_resolve_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="Invalid backend"):
        extension._resolve_gpu_backend("rocm")


# --- PyTorch build signals --------------------------------------------------


def test_torch_cuda_with_stray_rocm(
    monkeypatch: pytest.MonkeyPatch, _no_env_override: None
) -> None:
    """CUDA PyTorch must win even when a stray /opt/rocm-style HIP toolchain exists.

    This is the core regression for the misclassification bug: the presence of a
    ROCm install must not override an explicit CUDA build signal from PyTorch.
    """
    _install_fake_torch(monkeypatch, cuda="12.4", hip=None)
    # hipcc is present on PATH, mimicking a box that also carries ROCm bits.
    monkeypatch.setattr(extension.shutil, "which", _fake_which({"hipcc", "nvcc"}))
    assert extension._detect_gpu_backend() == "cuda"


def test_torch_hip_signal(monkeypatch: pytest.MonkeyPatch, _no_env_override: None) -> None:
    _install_fake_torch(monkeypatch, cuda=None, hip="6.0.32830")
    monkeypatch.setattr(extension.shutil, "which", _fake_which({"nvcc"}))
    assert extension._detect_gpu_backend() == "hip"


def test_torch_inconclusive_falls_back_to_nvcc(
    monkeypatch: pytest.MonkeyPatch, _no_env_override: None
) -> None:
    """A CPU-only PyTorch build reports neither signal; probe compilers instead."""
    _install_fake_torch(monkeypatch, cuda=None, hip=None)
    monkeypatch.setattr(extension.shutil, "which", _fake_which({"nvcc"}))
    assert extension._detect_gpu_backend() == "cuda"


def test_torch_helper_returns_none_when_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, cuda=None, hip=None)
    assert extension._detect_gpu_backend_from_torch() is None


def test_torch_helper_returns_none_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_torch(monkeypatch)
    assert extension._detect_gpu_backend_from_torch() is None


# --- Compiler fallback order (no conclusive torch) --------------------------


def test_nvcc_fallback_without_torch(
    monkeypatch: pytest.MonkeyPatch, _no_env_override: None
) -> None:
    _hide_torch(monkeypatch)
    monkeypatch.setattr(extension.shutil, "which", _fake_which({"nvcc"}))
    assert extension._detect_gpu_backend() == "cuda"


def test_hipcc_fallback_without_torch(
    monkeypatch: pytest.MonkeyPatch, _no_env_override: None
) -> None:
    _hide_torch(monkeypatch)
    monkeypatch.setattr(extension.shutil, "which", _fake_which({"hipcc"}))
    assert extension._detect_gpu_backend() == "hip"


def test_nvcc_precedes_hipcc_when_both_present(
    monkeypatch: pytest.MonkeyPatch, _no_env_override: None
) -> None:
    _hide_torch(monkeypatch)
    monkeypatch.setattr(extension.shutil, "which", _fake_which({"nvcc", "hipcc"}))
    assert extension._detect_gpu_backend() == "cuda"


def test_stray_rocm_alone_does_not_select_hip(
    monkeypatch: pytest.MonkeyPatch, _no_env_override: None
) -> None:
    """Regression for #651: only /opt/rocm present, but nvcc is the real compiler.

    With no torch signal and only nvcc available, detection must resolve to CUDA
    even though a ROCm directory exists on disk. Detection no longer consults the
    ROCm home directory, so the stray install cannot force a HIP selection.
    """
    _hide_torch(monkeypatch)
    monkeypatch.setattr(extension.shutil, "which", _fake_which({"nvcc"}))
    assert extension._detect_gpu_backend() == "cuda"


def test_no_backend_resolves_raises(
    monkeypatch: pytest.MonkeyPatch, _no_env_override: None
) -> None:
    _hide_torch(monkeypatch)
    monkeypatch.setattr(extension.shutil, "which", _fake_which(set()))
    with pytest.raises(RuntimeError, match="Could not determine the GPU backend"):
        extension._detect_gpu_backend()
