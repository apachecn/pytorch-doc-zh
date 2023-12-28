# C++ [¶](#c "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/cpp_index>
>
> 原始地址：<https://pytorch.org/docs/stable/cpp_index.html>




!!! note "笔记"

    如果您正在寻找 PyTorch C++ API 文档，请直接转到[此处](https://pytorch.org/cppdocs/)。


 PyTorch 提供了多种使用 C++ 的功能，最好根据您的需求进行选择。在较高级别上，可以提供以下支持：


## TorchScript C++ API [¶](#torchscript-c-api "此标题的永久链接")


[TorchScript](https://pytorch.org/docs/stable/jit.html) 允许对 Python 中定义的 PyTorch 模型进行序列化，然后在 C++ 中加载和运行，通过编译或跟踪其执行捕获模型代码。您可以在[用 C++ 教程加载 TorchScript 模型](https://pytorch.org/tutorials/advanced/cpp_export.html) 中了解更多信息。这意味着您可以尽可能地用 Python 定义模型，但随后通过 TorchScript 导出它们，以便在生产或嵌入式环境中执行非 Python 执行。 TorchScript C++ API 用于与这些模型和 TorchScript 执行引擎进行交互，包括：



* 加载从 Python 保存的序列化 TorchScript 模型 
* 如果需要，进行简单的模型修改(例如拉出子模块) 
* 使用 C++ Tensor API 构造输入并进行预处理


## 使用 C++ 扩展扩展 PyTorch 和 TorchScript [¶](#extending-pytorch-and-torchscript-with-c-extensions "永久链接到此标题")


 TorchScript 可以通过自定义运算符和自定义类使用用户提供的代码进行增强。注册到 TorchScript 后，可以在从 Python 或 C++ 运行的 TorchScript 代码中调用这些运算符和类，作为序列化 TorchScript 模型的一部分。 [使用自定义 C++ 运算符扩展 TorchScript](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html) 教程逐步介绍了 TorchScript 与 OpenCV 的接口。除了使用自定义运算符包装函数调用之外，C++ 类和结构可以通过类似 pybind11 的接口绑定到 TorchScript 中，该接口在[使用自定义 C++ 类扩展 TorchScript](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html) 教程中进行了解释。


## C++ 中的 Tensor 和 Autograd [¶](#tensor-and-autograd-in-c "此标题的永久链接")


 PyTorch Python API 中的大多数张量和 autograd 操作也可以在 C++ API 中使用。这些包括：



* `torch::Tensor` 方法，例如 `add` /`reshape` /`clone` 。有关可用方法的完整列表，请参阅：<https://pytorch.org/cppdocs/api/classat_1_1_tensor.html>
* C++ 张量索引 API，其外观和行为与 Python API 相同。有关其用法的详细信息，请参阅：<https://pytorch.org/cppdocs/notes/tensor_indexing.html>
* 张量 autograd API 和 `torch::autograd` 包对于用 C++ 构建动态神经网络至关重要前端。更多详情请参见：<https://pytorch.org/tutorials/advanced/cpp_autograd.html>


## 用 C++ 创作模型 [¶](#authoring-models-in-c "此标题的永久链接")


 “在 TorchScript 中创作，在 C++ 中推断”工作流程要求在 TorchScript 中完成模型创作。但是，可能存在必须在 C++ 中创作模型的情况(例如，在不需要 Python 组件的工作流程中)。为了服务此类用例，我们提供了纯粹用 C++ 编写和训练神经网络模型的完整功能，并使用熟悉的组件，例如`torch::nn`/`torch::nn::function`/`torch::optim` 与 Python API 非常相似。



* 有关 PyTorch C++ 模型创作和训练 API 的概述，请参阅：<https://pytorch.org/cppdocs/frontend.html>
* 有关如何使用 API 的详细教程，请参阅：<https://pytorch.org/tutorials/advanced/cpp_frontend.html>
* `torch::nn` /`torch::nn::function` /`torch::optim` 等组件的文档可以在以下位置找到： <https://pytorch.org/cppdocs/api/library_root.html>


## C++ 的打包 [¶](#packaging-for-c "此标题的永久链接")


 有关如何安装和链接 libtorch(包含上述所有 C++ API 的库)的指南，请参阅：<https://pytorch.org/cppdocs/installing.html>。请注意，在 Linux 上提供了两种类型的 libtorch 二进制文件：一种使用 GCC pre-cxx11 ABI 编译，另一种使用 GCC cxx11 ABI 编译，您应该根据系统使用的 GCC ABI 进行选择。