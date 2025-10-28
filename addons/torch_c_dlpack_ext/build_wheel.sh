pip install uv

./build_so.sh 2.4 $1 cu124
./build_so.sh 2.5 $1 cu124
./build_so.sh 2.6 $1 cu126
./build_so.sh 2.7 $1 cu126
./build_so.sh 2.8 $1 cu128
./build_so.sh 2.9 $1 cu128

uv venv /base --python $1 && source /base/bin/activate
uv pip install build auditwheel
uv pip install -v .
cd ./addons/torch_c_dlpack_ext
python -m build
auditwheel repair --exclude libtorch.so --exclude libtorch_cpu.so --exclude libc10.so --exclude libtorch_python.so --plat manylinux_2_28_x86_64 dist/*.whl -w wheelhouse
