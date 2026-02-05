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

"""Build multiple versions of the docs into a single output directory.

Versions are configured in `VERSIONS` (edit in-place).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

VERSIONS: tuple[str, ...] = ("v0.1.6-rc0", "v0.1.5", "main")

ENV_DOCS_VERSION = "TVM_FFI_DOCS_VERSION"
ENV_BASE_URL = "BASE_URL"
ENV_PRETEND_VERSION = "SETUPTOOLS_SCM_PRETEND_VERSION"

VERSIONS_JSON_NAME = "versions.json"

_STUB_FILES: dict[Path, Path] = {
    Path("_stubs/cpp_index.rst"): Path("reference/cpp/generated/index.rst"),
}

logger = logging.getLogger(__name__)


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(("git", *args), cwd=cwd).decode().strip()


def _git_toplevel() -> Path:
    start = Path(__file__).resolve().parent
    return Path(_git("rev-parse", "--show-toplevel", cwd=start)).resolve()


def _normalize_base_url(raw: str) -> str:
    value = raw.strip()
    if not value:
        return "/"
    if not value.startswith("/"):
        value = f"/{value}"
    if value != "/":
        value = value.rstrip("/")
    return value


def _preferred_version(versions: tuple[str, ...], *, latest: str) -> str:
    for v in versions:
        if v != latest:
            return v
    return latest


def _write_versions_json(*, output_root: Path, base_url: str, latest_version: str) -> str:
    base = base_url.rstrip("/")
    preferred = _preferred_version(VERSIONS, latest=latest_version)

    versions_json: list[dict[str, object]] = []
    for name in VERSIONS:
        entry: dict[str, object] = {
            "name": name,
            "version": name,
            "url": f"{base}/{name}/" if base else f"/{name}/",
        }
        if name == preferred:
            entry["preferred"] = True
        versions_json.append(entry)

    out_path = output_root / "_static" / VERSIONS_JSON_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"{json.dumps(versions_json, indent=2)}\n", encoding="utf-8")
    return preferred


def _write_root_index(*, output_root: Path, base_url: str, preferred: str) -> None:
    base = base_url.rstrip("/") or "/"
    target = f"{base}/{preferred}/" if base != "/" else f"/{preferred}/"
    (output_root / "index.html").write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                '<meta charset="utf-8" />',
                "<title>tvm-ffi docs</title>",
                f'<meta http-equiv="refresh" content="0; url={target}" />',
                "<script>",
                f"location.replace('{target}');",
                "</script>",
                f'<p>Redirecting to <a href="{target}">{target}</a>.</p>',
            ]
        ),
        encoding="utf-8",
    )


def _resolve_commit(gitroot: Path, ref: str) -> str:
    try:
        return _git("rev-parse", f"{ref}^{{commit}}", cwd=gitroot)
    except subprocess.CalledProcessError:
        if ref == "main":
            return _git("rev-parse", "HEAD", cwd=gitroot)
        raise


def _archive_extract(gitroot: Path, *, commit: str, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with tempfile.SpooledTemporaryFile() as fp:
        subprocess.check_call(("git", "archive", "--format", "tar", commit), cwd=gitroot, stdout=fp)
        fp.seek(0)
        with tarfile.open(fileobj=fp) as tar_fp:
            try:
                tar_fp.extractall(dst, filter="fully_trusted")
            except TypeError:
                tar_fp.extractall(dst)


def _ensure_stub_files(docs_dir: Path) -> None:
    for src_rel, dst_rel in _STUB_FILES.items():
        dst = docs_dir / dst_rel
        if dst.exists():
            continue
        src = docs_dir / src_rel
        if not src.exists():
            raise FileNotFoundError(f"Missing stub source: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputdir",
        default="docs/_build/html",
        help="Output root directory (default: docs/_build/html)",
    )
    parser.add_argument(
        "--base-url",
        default="/",
        help="Base URL prefix for generated links, e.g. '/' or '/ffi' (default: '/')",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint."""
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args, sphinx_argv = _parse_args(argv)

    gitroot = _git_toplevel()
    docs_confdir = Path(__file__).resolve().parent
    base_url = _normalize_base_url(args.base_url)

    output_root = Path(args.outputdir)
    if not output_root.is_absolute():
        output_root = (gitroot / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    preferred = _write_versions_json(
        output_root=output_root, base_url=base_url, latest_version="main"
    )
    _write_root_index(output_root=output_root, base_url=base_url, preferred=preferred)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for ref in VERSIONS:
            commit = _resolve_commit(gitroot, ref)
            repopath = tmp_root / f"{ref}-{commit}"
            _archive_extract(gitroot, commit=commit, dst=repopath)

            version_docs = repopath / "docs"
            if not version_docs.exists():
                raise FileNotFoundError(f"Missing docs/ for {ref} ({commit})")
            _ensure_stub_files(version_docs)

            version_out = output_root / ref
            version_out.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env.setdefault(ENV_PRETEND_VERSION, "0.0.0")
            env[ENV_BASE_URL] = base_url
            env[ENV_DOCS_VERSION] = ref

            cmd = (
                sys.executable,
                "-m",
                "sphinx",
                *sphinx_argv,
                "-c",
                str(docs_confdir),
                str(version_docs),
                str(version_out),
            )
            logger.info("Building %s -> %s", ref, version_out)
            subprocess.check_call(cmd, cwd=repopath, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
