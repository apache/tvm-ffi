pip install uv

for torch_minor in {8..9}
do
  uv venv /torch2$torch_minor && source /torch2$torch_minor/bin/activate
  uv pip install torch==2.$torch_minor --index-url https://download.pytorch.org/whl/cu126
  uv pip install setuptools build
  uv pip install -v .
  python python/tvm_ffi/utils/_build_optional_c_dlpack.py --build_dir ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext
  mv ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext/libtorch_c_dlpack_addon.so ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext/libtorch_c_dlpack_addon_torch2$torch_minor.so
  rm /torch2$torch_minor -rf
done

cd ./addons/torch_c_dlpack_ext
python -m build