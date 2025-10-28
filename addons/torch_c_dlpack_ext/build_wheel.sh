#!/bin/bash
# shellcheck disable=SC1091
set -eux
set -o pipefail

pip install uv

./addons/torch_c_dlpack_ext/build_so.sh 2.4 "$1" cu124
./addons/torch_c_dlpack_ext/build_so.sh 2.5 "$1" cu124
./addons/torch_c_dlpack_ext/build_so.sh 2.6 "$1" cu126
./addons/torch_c_dlpack_ext/build_so.sh 2.7 "$1" cu126
./addons/torch_c_dlpack_ext/build_so.sh 2.8 "$1" cu128
./addons/torch_c_dlpack_ext/build_so.sh 2.9 "$1" cu128

uv venv /base --python "$1" && source /base/bin/activate
uv pip install build auditwheel
uv pip install -v .
cd ./addons/torch_c_dlpack_ext
python -m build
auditwheel repair --exclude libtorch.so --exclude libtorch_cpu.so --exclude libc10.so --exclude libtorch_python.so --plat manylinux_2_28_x86_64 dist/*.whl -w wheelhouse
