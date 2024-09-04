# Usage

使用该工具从 GPU 模型脚本迁移至 MLU 设备运行，转换后的模型脚本只支持 MLU 设备运行。

该工具可对模型脚本进行转换，并对修改位置进行统计，实现开发者快速迁移。

转换工具会根据 《寒武纪PyTorch v1.13.1⽹络移植⼿册》 中的 API 支持列表，将 CUDA 相关接口替换成 MLU 相关接口。

## 注意事项

- 部分模型在转换后仍需要用户按照脚本实际情况进行少量适配，例如，用户自定义的不在 API 支持列表中的 GPU 接口，工具无法自动转换。

- 如需支持转换非默认编码格式文件，建议提前安装chardet第三方python库。


## 执行脚本转换工具

模型转换工具 ``torch_gpu2mlu.py`` 位于 ``torch_mlu/tools`` 目录下。

```shell
python <torch_mlu>/tools/torch_gpu2mlu/torch_gpu2mlu.py -i <模型脚本路径>
```

以下为参数解释：

-i：指定模型脚本路径。



## 输出转换结果

脚本执行后，终端会显示原始脚本路径、转换后脚本路径、以及模型脚本修改日志。

其中，转换后脚本和模型脚本修改日志均位于原始脚本相同路径的 ``_mlu`` 结尾的文件夹中。

以PyTorch ImageNet官方示例程序为例，使用该工具进行脚本转换，显示结果如下：

```shell
python <torch_mlu>/tools/torch_gpu2mlu/torch_gpu2mlu.py -i /tmp/imagenet

Official PyTorch model scripts: /tmp/imagenet

Cambricon PyTorch model scripts: /tmp/imagenet_mlu

Migration Report: /tmp/imagenet_mlu/report.md
```



## 脚本修改日志

脚本修改日志包含修改文件、修改位置对应的行号和修改内容。迁移过程中可以参考该日志进行模型脚本修改。脚本修改日志示例如下：

```shell
| No. |  File  |  Description  |

| 1 | main.py:8 | add "import torch_mlu" |

| 3 | main.py:139 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |

| 4 | main.py:146 | change "torch.cuda.set_device(args.gpu)" to "torch.mlu.set_device(args.gpu) " |

| 5 | main.py:147 | change "model.cuda(args.gpu)" to "model.mlu(args.gpu) " |

| 6 | main.py:155 | change "model.cuda()" to "model.mlu() " |

| 7 | main.py:160 | change "torch.cuda.set_device(args.gpu)" to "torch.mlu.set_device(args.gpu) " |

| 8 | main.py:161 | change "model = model.cuda(args.gpu)" to "model = model.mlu(args.gpu) " |

| 9 | main.py:166 | change "model.cuda()" to "model.mlu() " |

    ... ...
```
