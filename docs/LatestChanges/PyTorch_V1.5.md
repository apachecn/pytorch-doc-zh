# 新版本: PyTorch 1.5发布，新的和更新的API，包括C++前端API与Python的平价

> 发布: 2020年04月21日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch-1-dot-5-released-with-new-and-updated-apis>
> 
> 翻译: <https://pytorch.apachecn.org/docs/LatestChanges/PyTorch_V1.5>

**来自 PyTorch团队**

今天，我们宣布PyTorch 1.5以及新的和更新的库的可用性。此版本包括几个主要的新的API添加和改进。PyTorch现在包括对C++前端的重大更新，计算机视觉模型的“通道最后”内存格式，以及用于模型并行训练的分布式RPC框架的稳定版本。该版本还为hessians和jacobians提供了新的autograd API，以及一个允许创建受pybind启发的自定义C++类的API。

您可以在此处找到详细的发布说明。

## C++前端API（稳定）

C++前端API现在与Python处于同等地位，整体功能已移至“稳定”（以前标记为实验性）。一些主要亮点包括：

* 现在，有了~100%的C++ torch::nn模块/功能的覆盖率和文档，用户可以轻松地将他们的模型从Python API转换为C++ API，使模型创作体验更加流畅。
* C++中的优化器偏离了Python的等价物：C++优化器不能将参数组作为输入，而Python的优化器可以。此外，步进函数的实现并不完全相同。* 在1.5版本中，C++优化器的行为将始终与Python的等效相同。
* C++中缺少张量多维索引API是一个众所周知的问题，并在PyTorch Github问题跟踪器和论坛上发布了许多帖子。之前的解决方法是使用narrow / select / index_select / masked_selec 的组合，与Python 简明的API tensor[:, 0, ..., mask] 语法相比，这是笨重且容易出错的。在1.5版本中，用户可以使用 tensor.index({Slice(), 0, "...", mask}) 来达到同样的目的。

## 计算机视觉模型的“通道最后”内存格式（实验）

“通道最后”内存布局解锁了使用高性能卷积算法和硬件（NVIDIA的Tensor Cores、FBGEMM、QNNPACK）的能力。此外，它被设计为通过运算符自动传播，这允许在内存布局之间轻松切换。

在这里了解有关如何编写内存格式感知运算符的更多信息。

## 自定义C++类（实验）

此版本添加了一个新的API，torch::class_，用于将自定义C++类同时绑定到TorchScript和Python中。这个API在语法上几乎与pybind11相同。它允许用户将他们的C++类及其方法公开给TorchScript类型系统和运行时系统，这样他们就可以从TorchScript和Python实例化和操作任意C++对象。C++绑定示例：

```
template <class T>
struct MyStackClass : torch::CustomClassHolder {
  std::vector<T> stack_;
  MyStackClass(std::vector<T> init) : stack_(std::move(init)) {}

  void push(T x) {
    stack_.push_back(x);
  }
  T pop() {
    auto val = stack_.back();
    stack_.pop_back();
    return val;
  }
};

static auto testStack =
  torch::class_<MyStackClass<std::string>>("myclasses", "MyStackClass")
      .def(torch::init<std::vector<std::string>>())
      .def("push", &MyStackClass<std::string>::push)
      .def("pop", &MyStackClass<std::string>::pop)
      .def("size", [](const c10::intrusive_ptr<MyStackClass>& self) {
        return self->stack_.size();
      });
```

它公开了一个您可以在Python和TorchScript中使用的类，就像这样：

```
@torch.jit.script
def do_stacks(s : torch.classes.myclasses.MyStackClass):
    s2 = torch.classes.myclasses.MyStackClass(["hi", "mom"])
    print(s2.pop()) # "mom"
    s2.push("foobar")
    return s2 # ["hi", "foobar"]
```

你可以[在这里](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html)的教程中尝试一下。

## 分布式RPC框架API（现在稳定）

分布式[RPC框架](https://pytorch.org/docs/stable/rpc.html)在1.4版本中作为实验性推出，建议将分布式RPC框架标记为稳定且不再是实验性的。这项工作涉及许多增强和错误修复，以使分布式RPC框架总体上更加可靠和强大，并添加了一些新功能，包括分析支持，在RPC中使用TorchScript函数，以及一些易于使用的增强功能。以下是框架内各种API的概述：

## RPC API

RPC API允许用户指定要运行的函数和在远程节点上实例化的对象。这些函数被透明地记录，以便梯度可以使用分布式自动通过远程节点反向传播。

## 分布式Autograd

Distributed Autograd将autograd图连接到多个节点，并允许梯度在向后传递期间流动。梯度被累积到上下文中（与Autograd的.grad字段相反），用户必须在dist_autograd.context()管理器下指定其模型的正向传递，以确保正确记录所有RPC通信。目前，只实现了FAST模式（请参阅[此处](https://pytorch.org/docs/stable/rpc/distributed_autograd.html#distributed-autograd-design)了解FAST和SMART模式之间的区别）。

## 分布式优化器

分布式优化器为每个工人创建RRefs，其参数需要梯度，然后使用RPC API远程运行优化器。用户必须收集所有远程参数并将其包装在RRef中，因为这是分布式优化器所需的输入。用户还必须指定分布式autograd context_id，以便优化器知道在哪个上下文中查找渐变。

[在此处](https://pytorch.org/docs/stable/rpc.html)了解有关分布式RPC框架API的更多信息。

## 新的高级AUTOGRAD API（实验）

PyTorch 1.5为torch.autograd.functional子模块带来了包括jacobian、hessian、jvp、vjp、hvp和vhp在内的新功能。此功能基于当前的API，并允许用户轻松执行这些功能。

可以在GitHub上的详细设计讨论[在这里](https://github.com/pytorch/pytorch/issues/30632)找到。

## 不再支持PYTHON 2

从PyTorch 1.5.0开始，我们将不再支持Python 2，特别是2.7版本。今后对Python的支持将仅限于Python 3，特别是Python 3.5、3.6、3.7和3.8（在PyTorch 1.4.0中首次启用）。

我们要感谢整个PyTorch团队和社区对这项工作的所有贡献。

Cheers!

PyTorch 团队