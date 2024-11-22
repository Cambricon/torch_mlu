# libtorch_mlu 执行demo

## 说明

## 构建BUILD

1. 编译torch_mlu_demo, 注意需要手动修改`CMakeLists.txt`中的`torch_path`及`torch_mlu_path`所指向的路径, 以及`build.sh`中`DCMAKE_PREFIX_PATH`所指向的路径.
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
./torch_mlu_demo
```
3. 标准输出流输出如下内容则证明运行成功.
```
Allocation not reused.
The value of reuslt:
 1
 2
 3
 4
[ mluFloatType{4} ]
Allocation reused.
The demo ran successfully.
```
