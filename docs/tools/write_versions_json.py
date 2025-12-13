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

"""Convert sphinx-multiversion metadata into the version switcher JSON.

Usage:
    python docs/tools/write_versions_json.py <html_root> [--base-url /]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from packaging import version as pkg_version

ROOT_VERSION = "main"
DEFAULT_BASE_URL = "/"
METADATA_NAME = "versions_metadata.json"


def _parse_creatordate(raw: str) -> datetime:
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S %z")
    except Exception:
        return datetime.min.replace(tzinfo=None)


def _load_versions(metadata_path: Path) -> list[dict[str, object]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    versions = []
    for name, entry in metadata.items():
        version_label = entry.get("version") or name
        if version_label in {"0.0.0", "0+unknown"}:
            version_label = name
        versions.append(
            {
                "name": name,
                "version": version_label,
                "is_released": bool(entry.get("is_released")),
                "creatordate": _parse_creatordate(entry.get("creatordate", "")),
            }
        )
    return versions


def _pick_preferred(versions: list[dict[str, object]], latest: str) -> str:
    non_main = [v for v in versions if v["name"] != "main"]
    released = [v for v in non_main if v["is_released"]]
    if released:
        return max(released, key=lambda v: v["creatordate"])["name"]
    if non_main:
        return max(non_main, key=lambda v: v["creatordate"])["name"]
    return latest


def _to_switcher(
    versions: list[dict[str, object]], preferred_name: str, base_url: str
) -> list[dict[str, object]]:
    base = base_url.rstrip("/")
    main_entry: dict[str, object] | None = None
    tag_entries: list[dict[str, object]] = []

    for v in versions:
        entry: dict[str, object] = {
            "name": v["name"],
            "version": v["version"],
            "url": f"{base}/{v['name']}/" if base else f"/{v['name']}/",
        }
        if v["name"] == "main":
            main_entry = entry
        else:
            tag_entries.append(entry)

    def _sort_key(entry: dict[str, object]) -> pkg_version.Version:
        name = str(entry["name"])
        label = name[1:] if name.startswith("v") else name
        try:
            return pkg_version.parse(label)
        except Exception:
            return pkg_version.parse("0")

    tag_entries.sort(key=_sort_key, reverse=True)

    ordered = tag_entries
    if main_entry:
        ordered.append(main_entry)

    for entry in ordered:
        if entry["name"] == preferred_name:
            entry["preferred"] = True
    return ordered


def _write_root_index(html_root: Path, target_version: str, base_url: str) -> None:
    base = base_url.rstrip("/") or "/"
    target = f"{base}/{target_version}/" if base != "/" else f"/{target_version}/"
    html_root.mkdir(parents=True, exist_ok=True)
    index_path = html_root / "index.html"
    index_path.write_text(
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
    print(f"Wrote root index redirect to {target}")


def main() -> int:
    """Entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "html_root",
        type=Path,
        help="Root of the built HTML output (expects _static/versions_metadata.json inside)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL prefix (leading slash, no trailing slash) for version links, e.g. '/' or '/ffi'",
    )
    args = parser.parse_args()

    html_root = args.html_root
    metadata_path = html_root / "_static" / METADATA_NAME
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    versions = _load_versions(metadata_path)
    preferred_name = _pick_preferred(versions, ROOT_VERSION)
    output = _to_switcher(versions, preferred_name, args.base_url)

    out_path = html_root / "_static" / "versions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote version switcher data for {len(output)} entries to {out_path}")

    _write_root_index(html_root, preferred_name, args.base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
