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

import random
import sys
import time

from tvm_ffi.utils.lockfile import FileLock


def worker(worker_id: int, lock_path: str, counter_file: str) -> int:
    """Worker function that tries to acquire lock and increment counter."""
    try:
        with FileLock(lock_path):
            # Critical section - read, increment, write counter
            with open(counter_file) as f:  # noqa: PTH123
                current_value = int(f.read().strip())

            time.sleep(random.uniform(0.01, 0.1))  # Simulate some work

            with open(counter_file, "w") as f:  # noqa: PTH123
                f.write(str(current_value + 1))

            print(f"Worker {worker_id}: success")
            return 0
    except Exception as e:
        print(f"Worker {worker_id}: error: {e}")
        return 1


if __name__ == "__main__":
    worker_id = int(sys.argv[1])
    lock_path = sys.argv[2]
    counter_file = sys.argv[3]
    sys.exit(worker(worker_id, lock_path, counter_file))
