# libtorch_mlu 执行resnet demo

## 准备输入及构建BUILD

1. 生成所需的.pt文件.
```
python resnet_trace.py
```

2. 重新编译`libtorch_mlu.so`.
```
export BUILD_LIBTORCH=1
export USE_PYTHON=0
cd path/to/your/torch_mlu
python setup.py install
```

3. 编译torch_mlu_demo, 注意需要手动修改`CMakeLists.txt`中的`torch_path`及`torch_mlu_root`所指向的路径, 以及`build.sh`中`DCMAKE_PREFIX_PATH`所指向的路径.
```
mkdir build
cp build.sh build/
cd build
./build.sh
```

## 执行demo

1. 设置动态库地址来查找`libtorch.so`.
```
export LD_LIBRARY_PATH=path/to/your/pytorch/torch/lib:$LD_LIBRARY_PATH
```

2. 执行demo.
```
#program <model_path>
./libtorch_resnet ../resnet_model.pt
```
