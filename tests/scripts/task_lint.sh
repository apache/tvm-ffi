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

set -euxo pipefail

cleanup() {
	rm -rf /tmp/$$.*
}
trap cleanup 0

function run_lint {
	echo "Checking file types..."
	python tests/lint/check_file_type.py

	echo "Checking ASF headers..."
	python tests/lint/check_asf_header.py --check

	echo "isort check..."
	isort --check --diff .

	echo "black check..."
	black --check --diff .

	echo "ruff check..."
	ruff check --diff .

	echo "clang-format check..."
	tests/lint/git-clang-format.sh
}

run_lint
