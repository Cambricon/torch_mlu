# Torch-MLU 贡献指南
-   [贡献者许可协议](#贡献者许可协议.md)
-   [从源码构建](#从源码构建.md)
-   [开发指导](#开发指导.md)
    -   [测试用例](#测试用例.md)
    -   [代码风格](#代码风格.md)

<h2 id="贡献者许可协议.md">贡献者许可协议</h2>

在您第一次向 Torch MLU 提交代码之前，需要签署 [CLA(TBD)](https://www.cambricon.com)。

<h2 id="从源码构建.md">从源码构建</h2>

## 软件环境依赖

- 总体与PyTorch社区构建[保持一致](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source)。
- 编译器：推荐g++版本>=9.4.0。
- Python >= 3.10.0。
- CMake >= 3.18.0。
- 寒武纪基础软件平台软件包：CNToolkit、CNNL、CNCL、CNCV、DALI、CNNL_Extra、BANGC OPS。
  > **Note:**
  >
  > 寒武纪发布的docker包含了从源码构建所需要的所有依赖包，推荐您使用寒武纪发布的docker镜像作为开发环境进行源码构建。

## 安装步骤

1. 导入编译与运行测试脚本所需的环境变量。

   ```
   #设置环境变量:（如PYTORCH_HOME，TORCH_MLU_HOME，LD_LIBRARY_PATH等）
   pushd cambricon_torch
   export SRC_PACKAGES=$PWD
   source $SRC_PACKAGES/torch_mlu/scripts/release/env_pytorch.sh
   popd
   ```


2. 在Cambricon PyTorch或Cambricon TORCH_MLU源码目录下安装Virtualenv并激活虚拟环境。本例使用Cambricon TORCH_MLU目录。

   ```
   pushd ${TORCH_MLU_HOME}
   $(which pip3) install virtualenv #安装虚拟环境，此处pip3可按需更换为指定版本
   $(which python3) -m virtualenv venv/pytorch #安装虚拟环境，此处python3可按需更换为指定版本
   source venv/pytorch/bin/activate #激活虚拟环境
   popd
   ```


3. 【可选】对PyTorch进行Patch

    > **注意:**
    >
    > 目前仅 PyTorch 2.1 以及 2.3 版本需要执行此步骤。main, PyTorch 2.4 以后的版本无需执行此步骤。


   ```
   pushd ${TORCH_MLU_HOME}/scripts
   bash apply_patches_to_pytorch.sh
   popd
   ```


4. 安装Cambricon PyTorch所依赖的第三方包，并编译Cambricon PyTorch。

   第三方依赖包列表可在PyTorch源码主目录下的requirements.txt中查询。

   ```
   pushd ${PYTORCH_HOME}
   pip install -r requirements.txt #安装第三方包
   rm -rf build #清理环境
   python setup.py install #开始编译
   popd
   ```


5. 安装Cambricon TORCH_MLU所依赖的第三方包，并编译Cambricon TORCH_MLU。

   第三方依赖包列表可在TORCH_MLU源码主目录下的requirements.txt中查询。

   ```
   pushd ${TORCH_MLU_HOME}
   pip install -r requirements.txt #安装第三方包
   rm -rf build
   python setup.py install #开始编译
   popd
   ```


6. 确认编译结果。

   可在Python中引用PyTorch与Cambricon TORCH_MLU测试是否编译成功。

   ```
   # Python:
   >>> import torch
   >>> import torch_mlu
   ```

7. 运行测试示例。

   该示例需要在MLU服务器上运行。

   ``${TORCH_MLU_HOME}/examples/training`` 目录下为训练脚本。

   ```
   # 训练模式测试
   python $TORCH_MLU_HOME/examples/training/single_card_demo.py
   ```

10. 安装torchvision、torchaudio（可选）。

    对于PyTorch 2.1，运行如下命令：

    ```
    git clone https://github.com/pytorch/vision.git -b v0.16.0
    cd vision
    python setup.py install
    git clone https://github.com/pytorch/audio.git -b v2.1.1
    cd audio
    python setup.py install
    ```

    对于PyTorch 2.3，运行如下命令：

    ```
    git clone https://github.com/pytorch/vision.git -b v0.18.0
    cd vision
    python setup.py install
    git clone https://github.com/pytorch/audio.git -b v2.3.0
    cd audio
    python setup.py install
    ```

    对于PyTorch 2.4，运行如下命令：

    ```
    git clone https://github.com/pytorch/vision.git -b v0.19.0
    cd vision
    python setup.py install
    git clone https://github.com/pytorch/audio.git -b v2.4.0
    cd audio
    python setup.py install
    ```


<h2 id="开发指导.md">开发指导</h2>

-   **[代码风格](#代码风格.md)**

-   **[报告问题](#报告问题.md)**

<h2 id="代码风格.md">代码风格</h2>

Torch-MLU继承了PyTorch社区的编码风格。使用Lintrunner进行代码风格检查。

代码风格检查是**强制**的，如果您提交的代码不能通过代码风格检查，在代码合入流水中会被拦截。

```
# 安装LintRunner
pip install lintrunner
cd torch_mlu
lintrunner init
```

常用的lintrunner命令。建议您在提交代码前在本地执行 `lintrunner -f` 进行自动代码格式化。

```
# 检查head commit的style
lintrunner

# 检查并自动修复style
lintrunner f

# 检查一整个PR/branch，比如你的PR的目标是main, 执行以下:
lintrunner -m main

# 只检查指定的文件:
lintrunner torch/jit/_script.py torch/jit/_trace.py

# 检查仓库里所有的文件:
lintrunner --all-files
```

<h2 id="报告问题.md">报告问题</h2>

您可以使用issue报告问题。我们会尽快回复您的问题。
