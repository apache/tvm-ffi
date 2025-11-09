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
"""Sphinx configuration for the tvm-ffi documentation site."""

# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

import setuptools_scm
import sphinx

os.environ["TVM_FFI_BUILD_DOCS"] = "1"

build_exhale = os.environ.get("BUILD_CPP_DOCS", "0") == "1"
build_rust_docs = os.environ.get("BUILD_RUST_DOCS", "0") == "1"

# Auto-detect sphinx-autobuild: Check if sphinx-autobuild is in the execution path
is_autobuild = any("sphinx-autobuild" in str(arg) for arg in sys.argv)

# -- Path constants -------------------------------------------------------
_DOCS_DIR = Path(__file__).resolve().parent
_RUST_DIR = _DOCS_DIR.parent / "rust"

# -- General configuration ------------------------------------------------
# Determine version without reading pyproject.toml
# Always use setuptools_scm (assumed available in docs env)
__version__ = setuptools_scm.get_version(root="..")

project = "tvm-ffi"

author = "Apache TVM FFI contributors"

version = __version__
release = __version__

# -- Extensions and extension configurations --------------------------------

extensions = [
    "breathe",
    "myst_parser",
    "nbsphinx",
    "autodocsumm",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.ifconfig",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.httpdomain",
    "sphinxcontrib.mermaid",
]

if build_exhale:
    extensions.append("exhale")

breathe_default_project = "tvm-ffi"

breathe_projects = {"tvm-ffi": "./_build/doxygen/xml"}

exhaleDoxygenStdin = """
INPUT = ../include
PREDEFINED             += TVM_FFI_DLL= TVM_FFI_DLL_EXPORT= TVM_FFI_INLINE= \
                          TVM_FFI_EXTRA_CXX_API= TVM_FFI_WEAK= TVM_FFI_DOXYGEN_MODE \
                          __cplusplus=201703
EXCLUDE_SYMBOLS        += *details*  *TypeTraits* std \
                         *use_default_type_traits_v* *is_optional_type_v* *operator* \
EXCLUDE_PATTERNS       += *details.h
ENABLE_PREPROCESSING   = YES
MACRO_EXPANSION        = YES
WARNINGS               = YES
WARN_AS_ERROR          = FAIL_ON_WARNINGS_PRINT   # if your Doxygen version supports it
"""

exhaleAfterTitleDescription = """
This page contains the full API index for the C++ API.
"""

# Setup the exhale extension

exhale_args = {
    "containmentFolder": "reference/cpp/generated",
    "rootFileName": "index.rst",
    "doxygenStripFromPath": "../include",
    "rootFileTitle": "Full API Index",
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": exhaleDoxygenStdin,
    "afterTitleDescription": exhaleAfterTitleDescription,
}
nbsphinx_allow_errors = True
cpp_id_attributes = [
    "TVM_FFI_DLL",
    "TVM_FFI_DLL_EXPORT",
    "TVM_FFI_INLINE",
    "TVM_FFI_EXTRA_CXX_API",
    "TVM_FFI_WEAK",
]

c_id_attributes = [
    "TVM_FFI_DLL",
    "TVM_FFI_DLL_EXPORT",
    "TVM_FFI_WEAK",
]

nbsphinx_execute = "never"

autosectionlabel_prefix_document = True
nbsphinx_allow_directives = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "linkify",
    "attrs_block",
    "substitution",
]

myst_heading_anchors = 3
myst_ref_domains = ["std", "py"]
myst_all_links_external = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

autosummary_generate = True  # actually create stub pages

# Map object names -> stub docnames (no ".rst"; relative to your :toctree: dir)
autosummary_filename_map = {
    "tvm_ffi.device": "tvm_ffi.device_function",
    "tvm_ffi.Device": "tvm_ffi.Device_class",
}

_STUBS = {
    "_stubs/cpp_index.rst": "reference/cpp/generated/index.rst",
}


def _prepare_stub_files() -> None:
    """Move stub files into place if they do not already exist."""
    for src, dst in _STUBS.items():
        src_path = _DOCS_DIR / src
        dst_path = _DOCS_DIR / dst
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not dst_path.exists():
            dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")


def _build_rust_docs() -> None:
    """Build Rust documentation using cargo doc."""
    if not build_rust_docs:
        return

    print("Building Rust documentation...")
    try:
        target_doc = _RUST_DIR / "target" / "doc"

        # In auto-reload mode (sphinx-autobuild), keep incremental builds
        # Otherwise (CI/production), do clean rebuild
        if not is_autobuild and target_doc.exists():
            print("Clean rebuild: removing old documentation...")
            shutil.rmtree(target_doc)

        # Generate documentation (without dependencies)
        subprocess.run(
            ["cargo", "doc", "--no-deps", "--workspace", "--target-dir", "target"],
            check=True,
            cwd=_RUST_DIR,
            env={**os.environ, "RUSTDOCFLAGS": "--cfg docsrs"},
        )

        print(f"Rust documentation built successfully at {target_doc}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to build Rust documentation: {e}")
    except FileNotFoundError:
        print("Warning: cargo not found, skipping Rust documentation build")


def _apply_config_overrides(_: object, config: object) -> None:
    """Apply runtime configuration overrides derived from environment variables."""
    config.build_exhale = build_exhale
    config.build_rust_docs = build_rust_docs


def _copy_rust_docs_to_output(app: sphinx.application.Sphinx, exception: Exception | None) -> None:
    """Copy Rust documentation to the HTML output directory after build completes."""
    if exception is not None or not build_rust_docs:
        return

    src_dir = _RUST_DIR / "target" / "doc"
    dst_dir = Path(app.outdir) / "reference" / "rust" / "generated"

    if src_dir.exists():
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        print(f"Copied Rust documentation from {src_dir} to {dst_dir}")
    else:
        print(
            f"Warning: Rust documentation source directory not found at {src_dir}. Skipping copy."
        )


def setup(app: sphinx.application.Sphinx) -> None:
    """Register custom Sphinx configuration values."""
    _prepare_stub_files()
    _build_rust_docs()
    app.add_config_value("build_exhale", build_exhale, "env")
    app.add_config_value("build_rust_docs", build_rust_docs, "env")
    app.connect("config-inited", _apply_config_overrides)
    app.connect("build-finished", _copy_rust_docs_to_output)
    app.connect("autodoc-skip-member", _filter_inherited_members)
    app.connect("autodoc-process-docstring", _link_inherited_members)


def _filter_inherited_members(app, what, name, obj, skip, options):  # noqa: ANN001, ANN202
    if name in _autodoc_always_show:
        return False
    if "built-in method " in str(obj):
        # Skip: `str.maketrans`, `EnumType.from_bytes`
        return True
    if getattr(obj, "__objclass__", None) in _py_native_classes:
        return True
    return None


def _link_inherited_members(app, what, name, obj, options, lines) -> None:  # noqa: ANN001
    # Only act on members (methods/attributes/properties)
    if what not in {"method", "attribute", "property"}:
        return
    cls = _import_cls(name.rsplit(".", 1)[0])
    if cls is None:
        return

    member_name = name.rsplit(".", 1)[-1]  # just "foo"
    base = _defining_class(cls, member_name)

    # If we can't find a base or this class defines it, nothing to do
    if base is None or base is cls:
        return

    # If it comes from builtins we already hide it; no link needed
    if base in _py_native_classes or getattr(base, "__module__", "") == "builtins":
        return
    owner_fq = f"{base.__module__}.{base.__qualname__}".replace("tvm_ffi.core.", "tvm_ffi.")
    role = "attr" if what in {"attribute", "property"} else "meth"
    lines.clear()
    lines.append(
        f"*Defined in* :class:`~{owner_fq}` *as {what}* :{role}:`~{owner_fq}.{member_name}`."
    )


def _defining_class(cls: type | None, attr_name: str) -> type | None:
    """Find the first class in cls.__mro__ that defines attr_name in its __dict__."""
    if not isinstance(cls, type):
        return None
    method = getattr(cls, attr_name, None)
    if method is None:
        return None
    for base in reversed(inspect.getmro(cls)):
        d = getattr(base, "__dict__", {})
        if d.get(attr_name, None) is method:
            return base
    return None


def _import_cls(cls_name: str) -> type | None:
    """Import and return the class object given its module and class name."""
    try:
        mod, clsname = cls_name.rsplit(".", 1)
        m = importlib.import_module(mod)
        return getattr(m, clsname, None)
    except Exception:
        return None


autodoc_mock_imports = ["torch"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
    "member-order": "bysource",
}
_autodoc_always_show = {
    "__dlpack__",
    "__dlpack_device__",
    "__device_type_name__",
    "__ffi_init__",
    "__from_extern_c__",
    "__from_mlir_packed_safe_call__",
}
# If a member method comes from one of these native types, hide it in the docs
_py_native_classes: tuple[type, ...] = (
    str,
    tuple,
    list,
    dict,
    set,
    frozenset,
    bytes,
    bytearray,
    memoryview,
    int,
    float,
    complex,
    bool,
    object,
)

autodoc_typehints = "description"  # or "none"
always_use_bars_union = True

# Preserve how defaults are written in your source (e.g., DEFAULT_SENTINEL)
# Requires Sphinx ≥ 4.0
autodoc_preserve_defaults = True

# Ask the extension to include defaults alongside types
# 'braces' works well with NumPy-style "Parameters" tables
typehints_defaults = "comma"  # also accepts: "comma", "braces-after"

# Optional: also add stubs for params you didn't list in the docstring
always_document_param_types = True

# Optional (pairs nicely with NumPy style)
napoleon_use_rtype = False
# -- Other Options --------------------------------------------------------

templates_path = []

redirects = {}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "_stubs"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_book_theme"
html_title = project
html_copy_source = True
html_last_updated_fmt = ""

html_favicon = "https://tvm.apache.org/images/logo/tvm-logo-square.png"


footer_dropdown = {
    "name": "ASF",
    "items": [
        ("ASF Homepage", "https://apache.org/"),
        ("License", "https://www.apache.org/licenses/"),
        ("Sponsorship", "https://www.apache.org/foundation/sponsorship.html"),
        ("Security", "https://tvm.apache.org/docs/reference/security.html"),
        ("Thanks", "https://www.apache.org/foundation/thanks.html"),
        ("Events", "https://www.apache.org/events/current-event"),
    ],
}


footer_copyright = "Copyright © 2025, Apache Software Foundation"
footer_note = (
    "Apache TVM, Apache, the Apache feather, and the Apache TVM project "
    + "logo are either trademarks or registered trademarks of the Apache Software Foundation."
)


def footer_html() -> str:
    """Generate HTML for the documentation footer."""
    # Create footer HTML with two-line layout
    # Generate dropdown menu items
    dropdown_items = ""
    for item_name, item_url in footer_dropdown["items"]:
        dropdown_items += f'<li><a class="dropdown-item" href="{item_url}" target="_blank" style="font-size: 0.9em;">{item_name}</a></li>\n'

    footer_dropdown_html = f"""
  <div class="footer-container" style="margin: 5px 0; font-size: 0.9em; color: #6c757d;">
      <div class="footer-line1" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
          <div class="footer-copyright-short">
              {footer_copyright}
          </div>
          <div class="footer-dropdown">
              <div class="dropdown">
                  <button class="btn btn-link dropdown-toggle" type="button" id="footerDropdown" data-bs-toggle="dropdown"
                  aria-expanded="false" style="font-size: 0.9em; color: #6c757d; text-decoration: none; padding: 0; border: none; background: none;">
                      {footer_dropdown["name"]}
                  </button>
                  <ul class="dropdown-menu" aria-labelledby="footerDropdown" style="font-size: 0.9em;">
{dropdown_items}                  </ul>
              </div>
          </div>
      </div>
      <div class="footer-line2" style="font-size: 0.9em; color: #6c757d;">
          {footer_note}
      </div>
  </div>
  """
    return footer_dropdown_html


html_theme_options = {
    "repository_url": "https://github.com/apache/tvm-ffi",
    "use_repository_button": True,
    "show_toc_level": 2,
    "extra_footer": footer_html(),
}

html_context = {
    "display_github": True,
    "github_user": "apache",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_static_path = ["_static"]

# Copy Rust documentation to output if enabled
html_extra_path = ["reference/rust/generated"] if build_rust_docs else []


html_css_files = ["custom.css"]
