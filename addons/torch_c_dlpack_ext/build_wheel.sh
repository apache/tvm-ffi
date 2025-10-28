pip install uv

for torch_minor in {8..9}
do
  uv venv /torch2$torch_minor --python 3.12 && source /torch2$torch_minor/bin/activate
  uv pip install torch==2.$torch_minor --index-url https://download.pytorch.org/whl/cu126
  uv pip install setuptools
  uv pip install -v .
  python python/tvm_ffi/utils/_build_optional_c_dlpack.py --build_dir ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext
  mv ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext/libtorch_c_dlpack_addon.so ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext/libtorch_c_dlpack_addon_torch2$torch_minor.so
  rm /torch2$torch_minor -rf
done

uv venv /base --python 3.12 && source /base/bin/activate
uv pip install build auditwheel
uv pip install -v .
cd ./addons/torch_c_dlpack_ext
python -m build
auditwheel repair --exclude libtorch --exclude libtorch_cpu --exclude libc10 --exclude libtorch_python --plat manylinux_2_28_x86_64 dist/*.whl