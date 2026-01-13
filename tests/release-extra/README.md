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

# Extra integration tests for release

This folder contains an extra set of regression tests related to important
downstream integration use cases that can be run during release to test integration compatibility of the code.
While most of the tests should be covered by downstream tests and unit tests (UT), it is still helpful to have some
form of quick integration checks during release.

**High-level summary:**

- This folder should not be used in CI (as it integrates downstream usage tests).
- We aim to curate this folder to contain a reasonably minimal set
  that covers representative use cases based on regressions and lessons learned.
- It is not meant to replace downstream tests, but can help catch issues during release.
