# 使用 MLUExtension 机制编译自定义算子步骤
## 概述
通常因为用户需求紧急或者是其它原因需要手动实现算子，CATCH 的 MLUExtension 提供了对编译BangC程序功能的封装。用户可以自行实现BangC程序然后通过MLUExtension编译然后使用PyBind11绑定Python接口从而实现对算子的调用。下面的例子将从0开始实现一个sigmoid函数。
为了更方便的描述整个流程，我们将在一个目录中描述整个实现：
```
# mlu_extension
├── README.md：描述算子功能的说明文档
├── mlu_custom_ext：生成的module模块用于在python层导入。
│   ├── __init__.py：python包固有文件
│   ├── extension.py：包含路径检查的函数
│   └── src：mlu代码文件，根据实际情况自己创建，在setup.py中修改即可（建议使用目录管理BangC代码）。
│      ├── ops：
│      │   ├── __init__.py：初始化文件
│      │   └── custom_ops.py：包装模块，可选。
│      └── src
│          ├── custom_sigmoid.cpp：对PyTorch层面Tensor的封装，和自定义算子中xxx_internal里面的实现类似。
│          ├── custom_sigmoid.h：头文件
│          └── mlu：
│               ├── bang_sigmoid_sample.mlu：核心BangC实现。
│               ├── bang_sigmoid_smaple.h：头文件
│               └── kernel.h：Block size/NRAM size 等参数设置 
├── setup.py：构建包的脚本。
└── tests
    └── test_sigmoid.py：对绑定代码的python侧测试。

```

算子层实现在`custom_sigmoid.cpp`中，当前版本PYTORCH社区提供了`TORCH_LIBRARY_FRAGMENT/TORCH_LIBRARY_IMPL`这两个宏来注册算子，社区文档参考`https://pytorch.org/tutorials/advanced/extend_dispatcher.html`

使用方式可以参考如下写法：

```

TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
  m.def("active_sigmoid_mlu(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
  m.impl(TORCH_SELECTIVE_NAME("mlu_custom_ext::active_sigmoid_mlu"), TORCH_FN(active_sigmoid_mlu));
}
```

同样，我们也可以使用pybind去绑定执行函数，类似如下方式

```
PYBIND11_MODULE(libmlu_custom_ext, m) {
  m.def("active_sigmoid_mlu", &active_sigmoid_mlu);
}
```
`libmlu_custom_ext`描述了最后生成的库用于导入，最后生成的库格式大概是`libmlu_custom_ext.cpython-37m-x86_64-linux-gnu`。具体查看[这里](https://pybind11.readthedocs.io/en/stable/basics.html)，其中`m.def("active_sigmoid_mlu", &active_sigmoid_mlu);`表示将`"active_sigmoid_mlu"`绑定一个执行函数`active_sigmoid_mlu`。正如上面所说，我们需要实现`bang_sigmoid_kernel_entry`。这里简单说明一下BangC程序的执行分别对应于BangC程序的三部分：

1. host部分：主要使用cnrt接口完成device内存分配，将host内存拷贝到对应的device位置，创建执行队列之类的任务。
2. host对device的调用：通常用于描述任务如何执行，比如是block任务还是union任务，设置对应任务启动的参数。
3. device部分：这部分通常使用`__mlu_global__`前缀，过于复杂的逻辑通常使用`__mlu_func__`封装成一个函数。

`bang_sigmoid_sample.mlu`算子的BangC实现保存在.mlu文件中
- `bang_sigmoid_kernel`：调用BangC指令完成计算。
- `bang_sigmoid_kernel_entry`：host侧对device的入口。
- `bang_sigmoid_sample`：host接口，用来测试算子实现，非必须。
```
template void bang_sigmoid_sample(float*, float*, int);
template void bang_sigmoid_kernel_entry(cnrtQueue *, float *, float *, int);
```
这两行用来显示特例化模板函数，主要用在头文件和模板分离的情况避免未定义的引用。第一个特例化非必须，主要用于方便在host端测试整个函数逻辑；第二个函数是真正特例化的部分。
```
// bang_sigmoid_sample.h
#pragma once
#include <cnrt.h>
template <typename T>
void bang_sigmoid_kernel_entry(
    cnrtQueue* queue,
    T* d_dst,
    T* d_src,
    int elem_count);

```
上述头文件主要包含的头文件为cnrt.h(cnrtQueue需要)暴露给PyTorch算子层，以通过其完成最终的计算。完成上述三段代码之后PyTorch C++层面的功能就已经就绪。

后续使用`python setup.py install`之后就可以通过：`import mlu_custom_ext mlu_custom_ext.ops.active_sigmoid_mlu`来调用自己实现的函数了。
