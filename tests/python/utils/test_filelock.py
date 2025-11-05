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
"""Tests for the FileLock utility."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from tvm_ffi.utils.lockfile import FileLock


def test_basic_lock_acquire_and_release() -> None:
    """Test basic lock acquisition and release."""
    with tempfile.TemporaryDirectory() as temp_dir:
        lock_path = Path(temp_dir) / "test.lock"
        lock = FileLock(str(lock_path))

        # Test acquire
        assert lock.acquire() is True
        assert lock._file_descriptor is not None

        # Test release
        lock.release()
        assert lock._file_descriptor is None


def test_context_manager() -> None:
    """Test FileLock as a context manager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        lock_path = Path(temp_dir) / "test.lock"

        with FileLock(str(lock_path)) as lock:
            assert lock._file_descriptor is not None
            # Lock should be acquired
            assert lock_path.exists()

        # Lock should be released after exiting context
        # Note: file may still exist but should be unlocked


def test_multiple_acquire_attempts() -> None:
    """Test multiple acquire attempts on the same lock instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        lock_path = Path(temp_dir) / "test.lock"
        lock = FileLock(str(lock_path))

        # First acquire should succeed
        assert lock.acquire() is True

        # Second acquire on same instance should fail
        # (can't acquire same lock twice)
        assert lock.acquire() is False

        lock.release()


def test_exception_in_context_manager() -> None:
    """Test that lock is properly released even when exception occurs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        lock_path = Path(temp_dir) / "test.lock"

        # Test that exception is propagated and lock is released
        with pytest.raises(ValueError, match="test exception"):
            with FileLock(str(lock_path)) as lock:
                assert lock._file_descriptor is not None
                raise ValueError("test exception")

        # Lock should be released, so we can acquire it again
        lock2 = FileLock(str(lock_path))
        assert lock2.acquire() is True
        lock2.release()


def test_concurrent_access() -> None:
    """Test concurrent access from multiple processes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        lock_path = Path(temp_dir) / "test.lock"
        counter_file = Path(temp_dir) / "counter.txt"

        # Initialize counter file
        counter_file.write_text("0")

        # Create worker script content

        # Write worker script to a temporary file
        worker_script_path = Path(__file__).parent / "filelock_worker.py"

        # Run multiple worker processes concurrently
        num_workers = 16
        processes = []
        for i in range(num_workers):
            p = subprocess.Popen(
                [sys.executable, str(worker_script_path), str(i), str(lock_path), str(counter_file)]
            )
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.wait(timeout=60.0)
            assert p.returncode == 0, f"Worker process failed with return code {p.returncode}"

        # Check final counter value
        final_count = int(counter_file.read_text().strip())

        # Counter should equal number of workers (no race conditions)
        assert final_count == num_workers


if __name__ == "__main__":
    pytest.main([__file__])
