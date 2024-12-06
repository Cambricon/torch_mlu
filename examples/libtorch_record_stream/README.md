# libtorch_mlu 执行demo

## 说明

## 构建demo

1. （可选）设置环境变量`TORCH_MLU_HOME`指向编译`torch_mlu`的目录或者`torch_mlu`wheel包所在的目录，`Torch_DIR`指向编译`torch`的目录或者`torch`wheel包所在的目录，如果已安装`torch`以及`torch_mlu`的wheel包则可跳过此步骤.

2. 编译torch_mlu_demo.
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
2. 标准输出流输出如下内容则证明运行成功.
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
