# MLUExtension 编译自定义算子

## 部分说明

1. `python setup.py install`：编译并安装包。

2. 如果实现 BangC 算子的时候按照头文件和实现分离的模式，cncc 的编译选项需要手动包含 BangC 算子头文件所在的路径：`"-I{}".format(os.path.join(cpath, "include")` ，原因在于 cncc 无法正确处理 C++ 头文件，导致编译出错。

3. 和 CUDA 类似，MLUExtension 可以传入参数设置编译架构。在 MLU 架构上可以设置两种编译参数：
- `--bang-arch`：指定同系列的架构，例如：`--bang-arch=compute_50` 等价为 `--bang-mlu-arch=mtp_592`，表示生成 MLU500 系列架构代码。
- `--bang-mlu-arch`：指定特定架构，例如：`--bang-mlu-arch=mtp_372` 表示生成 MLU370 的代码。
默认生成设备相关代码的时候采用 `--bang-arch` 模式，这样保证同系列板卡都可用。如果你需要生成特定架构的代码可以通过传入 `--bang-mlu-arch=xxx`，这样默认的架构参数将失效。
如果不传入任何架构相关信息，MLUBuildExtension 会自动获取板卡架构，保证当前板卡可用，同样采用 `--bang-arch` 参数设置生成特定架构下的代码。
架构参数 `--bang-arch` 或者 `--bang-mlu-arch` 可以通过 `cncc --help` 查询到。

4. 和 CUDA类似，设置环境变量 `TORCH_BANG_ARCH_LIST` 能开启对架构的支持，例如 `TORCH_BANG_ARCH_LIST="3.0;5.0"` 表示对 `--bang-arch` 设置为 `--bang-arch=compute_30 --bang-arch=compute_50`，其含义如上所述。上述设置中优先级顺序为setup.py中传参设置>环境变量设置。
- setup.py 中设置，默认认为用户清楚自己需要生成的是哪个架构的代码，这时候参数以用户指定为准。
- `TORCH_BANG_ARCH_LIST` 环境变量设置次之，此操作主要用来添加生成新的架构代码但是又不想在setup.py中设置的情况。
- 如果不设置，默认通过runtime获取当前架构，默认情况下不用设置架构信息，执行中运行时会自动获取对应架构。
- 设置环境变量同时传入编译器参数 `--bang-mlu-arch=mtp_372` 则以传入参数为准，环境变量不生效。
- 代码架构和运行设备需要匹配，否则容易出现类似 `Found kernel(xxx) but not load(xxx)` 的错误。例如3xx系列编译的代码不应在5xx系列设备运行。
