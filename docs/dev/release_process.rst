..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Release Process
===============

This guide describes Apache TVM-FFI release workflow for creating a release
candidate, staging artifacts on ASF SVN, finalizing the release tag, and
publishing release artifacts.

.. admonition:: Prerequisite
   :class: hint

   The following environment variables need to be set before running the
   release steps:

   .. code-block:: bash

     export FFI_VERSION="v0.1.7-rc0"
     export FFI_RELEASE_VERSION="v0.1.7"
     export ASF_USERNAME="your-apache-username"

   - Tools: ``git``, ``svn``, ``gpg``, ``gtar``, and ``shasum``;
   - A configured GPG key for signing artifacts.


.. important::
    This tutorial is based on macOS. Adjust commands accordingly for other
    operating systems.


Pre-release checklist (Step 0)
------------------------------

Before running the steps below, create the release candidate tag and open the
release vote.

**Step 0.1.** Tag pre-release ``$FFI_VERSION`` on `<https://github.com/apache/tvm-ffi/releases/new>`__.

**Step 0.2.** Open a voting thread: `https://github.com/apache/tvm-ffi/issues/new <https://github.com/apache/tvm-ffi/issues/new>`__ (`Example <https://github.com/apache/tvm-ffi/issues/359>`__)

Step 1. Create Release Candidate
--------------------------------

This builds and signs the source release artifacts under
``tvm-ffi-$FFI_VERSION/``.

.. code-block:: bash

  _make_tarball() {
    local _ver="$1"
    local _workdir="tvm-ffi-$_ver-release-files"
    local _release_dir="tvm-ffi-$_ver"
    local _tarball="apache-tvm-ffi-$_ver.tar.gz"

    mkdir -p "$_workdir" "$_release_dir"
    (
      cd "$_workdir"

      git clone --recursive https://github.com/apache/tvm-ffi.git ffi-release
      cd ffi-release

      git checkout "$_ver"
      rm -rf .DS_Store
      find . -name ".git*" -print0 | xargs -0 rm -rf

      cd ..
      gtar -czvf "$_tarball" -C ffi-release .
      gpg --armor --output "$_tarball.asc" --detach-sig "$_tarball"
      shasum -a 512 "$_tarball" > "$_tarball.sha512"
    )
    mv "$_workdir/$_tarball" "$_release_dir/"
    mv "$_workdir/$_tarball.asc" "$_release_dir/"
    mv "$_workdir/$_tarball.sha512" "$_release_dir/"
    rm -rf "$_workdir"
  }

  _upload_to_apache_dev_svn() {
    local _ver="$1"
    local _asf_username="$2"
    local _svn_dir="$(pwd)/svn-tvm"
    local _release_dir="tvm-ffi-$_ver"
    (
      svn co --depth=files "https://dist.apache.org/repos/dist/dev/tvm" $_svn_dir
      cp -r "${_release_dir}/" "$_svn_dir/$_release_dir"
      cd "$_svn_dir"
      svn add "$_release_dir"
      svn ci --username "$_asf_username" -m "Add TVM-FFI $_ver"
    )
  }

  _make_tarball "$FFI_VERSION"
  _upload_to_apache_dev_svn "$FFI_VERSION" "$ASF_USERNAME"


Step 2. Conclude Release
------------------------

After the vote passes, retag the release, publish the wheel, bump versions, and
trigger the docs release.

**Step 2.1.** Conclude voting results: `<https://github.com/apache/tvm-ffi/issues/new>`__. (`Example <https://github.com/apache/tvm-ffi/issues/366>`__)

**Step 2.2.** Publish PyPI wheel: `<https://github.com/apache/tvm-ffi/actions/workflows/publish_wheel.yml>`__.
(See :doc:`ci_cd` for how wheels are built with cibuildwheel.)

**Step 2.3.** Update documentation to latest: `<https://github.com/apache/tvm-site/actions/workflows/publish_tvm_ffi_docs.yml>`__.

**Step 2.4.** Re-tag the release candidate to the final release version, and bump the version in the source tree:

.. code-block:: bash

  _retag_and_bump_version() {
    local _ffi_version="$1"
    local _ffi_release_version="$2"

    # Configuration variables
    local _repo_url="git@github.com:apache/tvm-ffi.git"
    local _work_dir="tvm-ffi-release"
    local _git_remote="upstream"
    local _header_file="include/tvm/ffi/c_api.h"

    # 1. Git clone with remote named "upstream"
    echo "Cloning repository..."
    git clone -o "$_git_remote" "$_repo_url" "$_work_dir"
    cd "$_work_dir" || return 1
    git fetch "$_git_remote" --tags

    # 2. Replace tag and push
    local _ffi_commit="$(git rev-parse "${_ffi_version}^{commit}")"
    echo "Creating release tag $_ffi_release_version at commit $_ffi_commit..."
    git tag -a "$_ffi_release_version" "$_ffi_commit" -m "Release $_ffi_release_version"
    git push "$_git_remote" "$_ffi_release_version"
    git push "$_git_remote" --delete "$_ffi_version"

    # 3. Version bump after the release
    local _today=$(date +%Y-%m-%d)
    local _branch_name="${_today}/ver-bump"
    local _current_patch=$(grep "#define TVM_FFI_VERSION_PATCH" "$_header_file" | awk '{print $3}')
    local _new_patch=$((_current_patch + 1))
    git checkout -b "$_branch_name"
    echo "Bumping TVM_FFI_VERSION_PATCH from $_current_patch to $_new_patch"
    sed "s/#define TVM_FFI_VERSION_PATCH $_current_patch/#define TVM_FFI_VERSION_PATCH $_new_patch/" "$_header_file" > "${_header_file}.tmp" && mv "${_header_file}.tmp" "$_header_file"
    echo "Committing and pushing changes..."
    git add "$_header_file"
    git commit -m "chore(release): Version bump after release $_ffi_release_version"
    git push -u "$_git_remote" "$_branch_name"
  }

  _retag_and_bump_version "$FFI_VERSION" "$FFI_RELEASE_VERSION"


Step 3. Upload Release Artifacts
--------------------------------

After the release is final, copy the RC artifacts into ``dist/release`` with the
final version name.

.. code-block:: bash

  _upload_svn() {
    local _ffi_version="$1"
    local _ffi_release_version="$2"
    local _asf_username="$3"
    local _release_dir="tvm-ffi-$_ffi_version"
    local _svn_dir="$(pwd)/svn-tvm-release"
    (
      svn co --depth=files "https://dist.apache.org/repos/dist/release/tvm" $_svn_dir
      mkdir -p "$_svn_dir/tvm-ffi-$_ffi_release_version"
      cp "${_release_dir}/apache-tvm-ffi-$_ffi_version.tar.gz" "$_svn_dir/tvm-ffi-$_ffi_release_version/apache-tvm-ffi-src-$_ffi_release_version.tar.gz"
      cp "${_release_dir}/apache-tvm-ffi-$_ffi_version.tar.gz.asc" "$_svn_dir/tvm-ffi-$_ffi_release_version/apache-tvm-ffi-src-$_ffi_release_version.tar.gz.asc"
      cp "${_release_dir}/apache-tvm-ffi-$_ffi_version.tar.gz.sha512" "$_svn_dir/tvm-ffi-$_ffi_release_version/apache-tvm-ffi-src-$_ffi_release_version.tar.gz.sha512"
      cd "$_svn_dir"
      svn add "tvm-ffi-$_ffi_release_version"
      svn ci --username "$_asf_username" -m "Add TVM FFI $_ffi_release_version"
    )
  }

  _upload_svn $FFI_VERSION $FFI_RELEASE_VERSION $ASF_USERNAME
