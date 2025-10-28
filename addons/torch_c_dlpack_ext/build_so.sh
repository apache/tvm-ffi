echo "torch version == $1"
echo "python version == $2"
echo "cuda version == $3"

uv venv /torch$1 --python $2 && source /torch$1/bin/activate
uv pip install torch==$1 --index-url https://download.pytorch.org/whl/$3
uv pip install setuptools
uv pip install -v .
python python/tvm_ffi/utils/_build_optional_c_dlpack.py --build_dir ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext
mv ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext/libtorch_c_dlpack_addon.so ./addons/torch_c_dlpack_ext/torch_c_dlpack_ext/libtorch_c_dlpack_addon_torch$1.so
rm /torch$1 -rf
