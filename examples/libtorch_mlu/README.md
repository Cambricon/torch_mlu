# libtorch_mlu 执行demo

## 说明

## 准备输入及构建BUILD

1. 生成所需的.pt文件.
```
python trace_model.py
```

2. 编译torch_mlu_demo, 注意需要手动修改`CMakeLists.txt`中的`torch_path`及`torch_mlu_path`所指向的路径, 以及`build.sh`中`DCMAKE_PREFIX_PATH`所指向的路径.
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
#program <model_path> <num_threads> <loop_count>
./torch_mlu_demo ../add_model.pt 4 1000
./torch_mlu_demo ../conv_trace_model.pt 4 1000
```
