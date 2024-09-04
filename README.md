# Cambricon MLU Extension for PyTorch

## 简介

torch_mlu 是[寒武纪科技](https://www.cambricon.com)开发的 PyTorch 扩展包。

它通过PyTorch社区的[后端集成机制](https://pytorch.org/tutorials/advanced/privateuseone.html)允许用户在使用原生社区PyTorch的基础上灵活、快速的接入寒武纪MLU后端。为神经网络计算提供了强大的 MLU 加速。

## 安装

### 二进制wheel包安装

1. 安装底层依赖库。

   当前Cambricon torch_mlu依赖CNToolkit、CNNL、CNCL、CNCV、DALI、CNNL_Extra、BANGC OPS等库。各依赖库安装方法详见对应的用户手册。

   推荐使用Cambricon官方的PyTorch Container进行开发，内部已经预装了编译所需要的依赖组件。

2. 设置环境变量。

   ```
   export NEUWARE_HOME=/usr/local/neuware # 依赖库安装路径，根据依赖库实际安装路径修改
   export PATH=$PATH:$NEUWARE_HOME/bin # 系统环境变量，添加依赖库可行执行文件路径
   export LD_LIBRARY_PATH=$NEUWARE_HOME/lib64:$LD_LIBRARY_PATH # 系统环境变量， 添加依赖库库文件路径
   ```

3. 安装PyTorch二进制wheel安装包。

   对于Cambricon torch_mlu和各个PyTorch版本是否发布对应的wheel包，详情如下表：


   | wheel包名称           | PyTorch 2.1/2.3 | PyTorch 2.4 及以后    |
   |-----------------------|-----------------|-----------------------|
   | torch                 | True            | False                 |
   | torch_mlu             | True            | True                  |
   | torchvision           | False           | False                 |
   | torchaudio            | False           | False                 |

    > **注意:**
    >
    > * Cambricon torch_mlu 依赖于PyTorch，因此需要先安装 PyTorch。
    > * 对于PyTorch 2.1 以及 2.3 版本此步骤需要安装Cambricon torch_mlu 发布的PyTorch的二进制wheel安装包（非官方的安装包）。

   对于PyTorch 2.1，运行如下命令：

   ```
   pip install torch-2.1.0-cp310-cp310-linux_x86_64.whl
   ```

   对于PyTorch 2.3，运行如下命令：

   ```
   pip install torch-2.3.0+cpu-cp310-cp310-linux_x86_64.whl
   ```

   对于PyTorch 2.4，可以按照PyTorch官方指南（https://pytorch.org ）安装Cambricon torch_mlu支持的PyTorch版本，也可以直接使用以下命令。

   ```
   pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu # 安装 PyTorch cpu wheel包
   ```

   > **注意:**
   >
   > 目前对于PyTorch 2.4及以上版本，Cambricon torch_mlu 仅支持基于CPU的PyTorch wheel包进行编译安装，暂不支持 ``x.y.z+cpu.cxx11.abi`` 版本的PyTorch wheel包。


4. 安装torch_mlu二进制wheel包。

   ```
   pip install torch_mlu-{xxx}+torchx.y.z-cp310-cp310-linux_x86_64.whl
   ```

5. 确认安装结果。通过在Python中引用PyTorch与Cambricon torch_mlu测试是否安装成功。

   ```
   # python
   >>> import torch
   >>> import torch_mlu
   >>> a=torch.randn(2,3).mlu() # 该示例需要在MLU服务器上运行。
   >>> a.abs()
   ```

   以上测例说明安装成功。如果安装失败，则会有相应的错误提示。

6. 安装torchvision、torchaudio（可选）。

   对于PyTorch 2.1，运行如下命令：

   ```
   pip install --isolated torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu --no-deps
   ```

   对于PyTorch 2.3，运行如下命令：

   ```
   pip install --isolated torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu --no-deps
   ```

   对于PyTorch 2.4，运行如下命令：

   ```
   pip install --isolated torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu --no-deps
   ```

### 从源码构建

请参考 [CONTRIBUTING.md](CONTRIBUTING.zh.md)。

## 版本号规则

torch_mlu版本号采用`{torch_mlu版本}+torch{社区版本}`命名规则，比如`1.22.0+torch2.4.0`。代表torch_mlu版本号为`1.22.0`，对应社区版本为`2.4.0`.

## 分支规则

### 开发分支

开发分支的命名规则为`r{社区版本}_develop`，比如 `r2.3_develop` 为torch_mlu对应PyTorch社区2.3版本的开发分支。

### Release分支

Release分支从对应社区版本的develop分支中拉出。

Release分支的命名规则为`r{torch_mlu版本}_pt{pytorch社区版本}`，比如 `r1.12_pt2.4.0`，代表torch_mlu版本为`1.22.x`,对应社区版本为PyTorch `2.4.0`。


## 分支生命周期

| PyTorch 版本    | 开发分支      | 分支状态           |  发布时间           | 后续状态              |  计划EOL日期     |
|----------------|--------------|-------------------|-------------------|----------------------|-----------------|
| 2.4.0          | 2.4_develop  | 开发中             |  2024/08/15       |    开发中             |    TBD          |
| 2.3.0          | 2.3_develop  | 开发中             |  2024/06/15       |    开发中             |   2024/10/15    |
| 2.1.0          | 2.3_develop  | 长期支持           |  2024/03/15       |    开发中             |   2025/03/15    |


## 版本配套关系

| CTR-SDK版本     | torch-mlu版本 | PyTorch版本       |  分支              | 镜像([链接](https://developer.cambricon.com/)) |
|----------------|--------------|-------------------|-------------------|----------------------|
| CTR 2.15       | 1.22.0       | 2.4.0             | r1.22_pt2.4.0     | -                    |
|                |              | 2.3.0             | r1.22_pt2.3.0     | -                    |
|                |              | 2.1.0             | r1.22_pt2.1.0     | -                    |


## License

请参考 [LICENSE](LICENSE)。
