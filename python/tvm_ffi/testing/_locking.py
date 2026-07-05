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
"""Helpers for tests that use exclusive machine-local resources.

Parallel test runners (for example ``pytest -n auto``) spread tests across
several worker processes.  Tests that reach for a shared, machine-local
resource such as a single GPU must serialize with each other so that
concurrent access does not exhaust device memory or corrupt device state.
:func:`run_with_gpu_lock` provides that serialization through an advisory
file lock shared by every cooperating worker on the machine.
"""

from __future__ import annotations

import getpass
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from ..utils import FileLock

_LOCK_DIR_ENV_VAR = "TVM_FFI_TEST_LOCK_DIR"
_LOCK_DIR_PREFIX = "tvm-ffi-test-locks"
_LOCK_FILENAME = "gpu.lock"
_R = TypeVar("_R")

# Resolved GPU lock path, cached after the first ``run_with_gpu_lock`` call.
#
# The path is initialized lazily on first use rather than at import time:
# importing this module does not imply the caller will ever run a GPU test, and
# resolving the path touches the filesystem (it creates the lock directory), so
# there is no reason to pay that cost for callers that never lock. Caching also
# avoids recomputing the user tag and temp-dir lookup on every call.
#
# If several workers make their first call concurrently they may each resolve
# the path and race to assign this global. That race is intentionally benign:
# every worker derives the identical path, so whichever assignment wins leaves
# the cache correct.
_GPU_LOCK_PATH: Path | None = None


def _current_user_tag() -> str:
    """Return a filesystem-safe identifier for the current user.

    Falls back to the numeric uid, then to ``unknown``, when a login name is
    not resolvable on the host.
    """
    try:
        return getpass.getuser()
    except Exception:  # any resolution failure falls back to a numeric or default tag
        uid = getattr(os, "getuid", None)
        return str(uid()) if uid is not None else "unknown"


def _resolve_gpu_lock_path() -> Path:
    """Return the path to the machine-local GPU lock file, creating its directory.

    Returns
    -------
    path
        The full path to the ``gpu.lock`` file. The parent directory is created
        if it does not yet exist.

    Notes
    -----
    The lock directory defaults to a per-user directory under the system
    temporary directory, ``<tempdir>/tvm-ffi-test-locks-<user>``. Scoping the
    default to the current user avoids ownership and permission conflicts when
    several users share one host. It can be redirected with the
    ``TVM_FFI_TEST_LOCK_DIR`` environment variable when all cooperating
    processes need an explicitly shared machine-local path.

    """
    lock_dir_override = os.environ.get(_LOCK_DIR_ENV_VAR)
    if lock_dir_override:
        lock_dir = Path(lock_dir_override).expanduser()
    else:
        lock_dir = Path(tempfile.gettempdir()) / f"{_LOCK_DIR_PREFIX}-{_current_user_tag()}"

    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / _LOCK_FILENAME


def run_with_gpu_lock(func: Callable[..., _R], /, *args: Any, **kwargs: Any) -> _R:
    """Run a callable while holding the machine-local GPU lock.

    The lock serializes GPU access across parallel test workers so that
    concurrent device use does not break GPU-related tests. Pass a callable
    that contains the complete live-device lifetime (device creation,
    allocation, execution, synchronization, and result checks); keep work that
    does not touch the device, such as source compilation, outside the callable
    so it can still run in parallel.

    Parameters
    ----------
    func
        Callable containing the complete live local-GPU lifetime.
    args
        Positional arguments forwarded to ``func``.
    kwargs
        Keyword arguments forwarded to ``func``.

    Returns
    -------
    result
        The return value of ``func``.

    """
    # Resolve and cache the lock path on the first call (see ``_GPU_LOCK_PATH``
    # above for why this is lazy and why the concurrent-init race is benign).
    global _GPU_LOCK_PATH  # noqa: PLW0603 -- intentional first-call memoization cache
    lock_path = _GPU_LOCK_PATH
    if lock_path is None:
        lock_path = _resolve_gpu_lock_path()
        _GPU_LOCK_PATH = lock_path
    with FileLock(str(lock_path)):
        return func(*args, **kwargs)
