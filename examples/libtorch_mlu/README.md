# libtorch_mlu 执行demo

## 说明

## 准备输入及构建BUILD

1. 生成所需的.pt文件.
```
python trace_model.py
```

2. （可选）设置环境变量`TORCH_MLU_HOME`指向编译`torch_mlu`的目录或者`torch_mlu`wheel包所在的目录，`Torch_DIR`指向编译`torch`的目录或者`torch`wheel包所在的目录，如果已安装`torch`以及`torch_mlu`的wheel包则可跳过此步骤.

3. 编译torch_mlu_demo.
```
mkdir build
cp build.sh build/
cd build
./build.sh
```
## 执行demo

1. 执行demo.
```
cp ../run.sh .
./run.sh
```
