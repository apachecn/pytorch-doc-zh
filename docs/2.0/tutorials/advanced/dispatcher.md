


# 在 C++ 中注册调度运算符 [¶](#registering-a-dispatched-operator-in-c "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/dispatcher>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/dispatcher.html>




 调度程序是 PyTorch 的一个内部组件，
 负责确定当您调用像 
 `torch::add` 这样的函数时实际应该运行哪些代码。
 。这可能很重要，因为 PyTorch 操作需要处理许多横切问题，这些问题是相互叠加的。以下是它处理的一些事情的示例：



* 在算子的 CPU 和 CUDA 实现之间切换，取决于
输入张量的设备。
* 在算子的 autograd 和后端实现之间切换，
取决于是否需要 autograd 处理。
*必要时应用自动转换以实现自动混合精度。
* 当运算符在“vmap”调用下运行时应用批处理规则。
* 如果您正在跟踪要导出的模型，则跟踪操作的执行。



 如果在您的
 [自定义运算符代码](torch_script_custom_ops) 
 中您发现自己
手动编写 if 语句来处理这些情况，调度程序 API 可以
帮助组织您的代码。 （相反，如果您的自定义运算符非常简单
并且仅用于CPU推理，则您可能不需要’不需要使用调度程序，
只需使用基本API即可。）




 在本教程中，我们将描述如何构建自定义运算符
注册以使用调度程序来组织各种组件。我们’ll\假设您熟悉如何
 [注册运算符](torch_script_custom_ops) 
 以及如何编写
a
 [自定义自动分级函数](cpp_autograd) 
 。





## 定义架构和后端实现 [¶](#defining-schema-and-backend-implementations "永久链接到此标题")




 调度程序背后的一般原理是，它将运算符的实现划分为多个内核，每个内核实现
特定
 *调度键* 
 的功能，例如中央处理器、CUDA。调度程序
确定您调用运算符时
最高优先级的调度键是什么（这是通过查看张量参数以及
某些线程本地状态来完成的），并将控制权转移到内核
调度钥匙。最终效果是，当您调用运算符时，我们首先执行 Autograd 内核，然后根据传入张量的设备类型重新分派到后端内核。




 让’s 看一下导致这种情况发生的各个部分
。首先，我们必须定义相关运算符的架构。
与简单的 pybind11 风格的运算符注册不同，我们此时’t 实际上
不提供运算符的实现；我们只是
提供一个模式字符串，指定运算符的类型签名
我们所有其他内核都将遵守该字符串：






```
TORCH_LIBRARY(myops, m) {
 m.def("myadd(Tensor self, Tensor other) -> Tensor");
}

```




 接下来，我们需要实际提供该运算符的一些实现。
具体而言，这是一个在 CPU 上非常简单的加法实现：






```
Tensor myadd_cpu(const Tensor& self_, const Tensor& other_) {
 TORCH_CHECK(self_.sizes() == other_.sizes());
 TORCH_INTERNAL_ASSERT(self_.device().type() == DeviceType::CPU);
 TORCH_INTERNAL_ASSERT(other_.device().type() == DeviceType::CPU);
 Tensor self = self_.contiguous();
 Tensor other = other_.contiguous();
 Tensor result = torch::empty(self.sizes(), self.options());
 const float* self_ptr = self.data_ptr<float>();
 const float* other_ptr = other.data_ptr<float>();
 float* result_ptr = result.data_ptr<float>();
 for (int64_t i = 0; i < result.numel(); i++) {
 result_ptr[i] = self_ptr[i] + other_ptr[i];
 }
 return result;
}

```




 我们’d 喜欢将此函数注册为
 `myops::myadd`
 的实现。
但是，注册它的简单方法 (
 `def("myadd", 
 

 myadd_cpu)`
 ) 将
注册内核以在所有情况下运行，即使张量不是 CPU
张量！ （在内部，我们将这些称为 “catch-all” 内核，因为它们
c捕获所有情况。）为了确保 
 `myadd_cpu`
 仅运行于\ nCPU 张量，我们可以使用
 `TORCH_LIBRARY_IMPL`
 宏：






```
TORCH_LIBRARY_IMPL(myops, CPU, m) {
 m.impl("myadd", myadd_cpu);
}

```




 `TORCH_LIBRARY_IMPL`
 让我们可以在特定的调度键（在本例中为 CPU）上注册操作符的实现。每次调用
 `impl`
 都会将一个CPU 内核与相应的运算符（我们之前在
 `TORCH_LIBRARY` 块中定义）相关联。如果我们还有一个 CUDA 实现
 `myadd_cuda`
 ，
我们可以将其注册在单独的
 `TORCH_LIBRARY_IMPL`
 块中：






```
TORCH_LIBRARY_IMPL(myops, CUDA, m) {
 m.impl("myadd", myadd_cuda);
}

```




 这些注册可以跨文件甚至跨库边界分割；例如，您可以将这两个“TORCH_LIBRARY_IMPL”块编译
为单独的
“myops_cpu”
和
“myops_cuda”
动态库。一般来说，
您的注册结构将如下所示：



1. 单个
 `TORCH_LIBRARY`
 在一个集中位置
 列出命名空间中的每个自定义运算符。
2.每个调度键都有一个 `TORCH_LIBRARY_IMPL`，用于注册该键的实现（例如 CPU 或 CUDA）。如果您愿意，您可以进一步将
 `TORCH_LIBRARY_IMPL`
 块细分为每个运算符的块。如果每个运算符实现都有一个单独的文件，但’ 不想
在标头中公开运算符，这很方便；您只需将注册放入定义您的操作员的
cpp 文件中即可。




 注意




 您知道吗，您还可以为 PyTorch 中现有
核心运算符编写
 `TORCH_LIBRARY_IMPL`
 块？这就是 XLA 对 PyTorch 支持的实现方式：
 `torch_xla`
 库包含
 
 `TORCH_LIBRARY_IMPL`
 ，它为 XLA 调度上的所有基本运算符提供实现
 n键。





## 对于不需要 autograd 的操作员 [¶](#for-operators-that-do-not-need-autograd "永久链接到此标题")




 注意：本节仅适用于 PyTorch 版本
 `>=
 

 1.10`
.




 在下一节中，我们将讨论如何为算子添加 autograd 支持。
但是对于不需要 autograd 支持的操作，应该注册以下内核
以提高可用性，并使您的操作表现得像 PyTorch\xe2\ x80\x99s 内置
运算符。






```
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
 m.impl(op, autogradNotImplementedFallback());
}

```




 上述行注册了一个
 `Autograd`
 内核，该内核在向前附加一个虚拟
 `NotImplemented`
 节点（保留
 `require_grad`
 输入的性质）.
向后，
 `NotImplemented`
 节点会引发错误。这对于在较大的模型中
进行调试很有帮助，而以前很难
准确地指出
`requires_grad`
在前向传播期间丢失的位置。




### 就地或查看操作 [¶](#in-place-or-view-ops "此标题的永久链接")



 为了确保正确性和最佳性能，如果您的操作就地改变输入
或返回与其中一个输入别名的张量，则应采取两个额外
步骤：



1. 除了上面的
 `Autograd`
 内核之外，还注册
 `ADInplaceOrView`
 内核。该内核处理必要的簿记以确保就地或视图操作的正确性。需要注意的是，此 ADInplaceOrView
内核只能与
 `autogradNotImplementedFallback`
 一起使用。





```
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
 m.impl(op, autogradNotImplementedFallback());
}
TORCH_LIBRARY_IMPL(myops, ADInplaceOrView, m) {
 m.impl(op, autogradNotImplementedInplaceOrViewFallback());
}

```



2.上面注册的
 `Autograd`
 或
 `ADInplaceOrView`
 盒装内核
依赖于其日志中的操作员架构信息。如果您的操作就地改变输入或返回与其中一个输入别名的张量，请务必确保您的架构正确反映这一点。有关如何注释架构的更多信息，请参阅
 [此处](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/README.md)。








## 添加 autograd 支持 [¶](#adding-autograd-support "永久链接到此标题")




 此时，我们就有了一个同时具有 CPU 和 CUDA 实现的运算符。我们如何
为其添加 autograd 支持？正如您可能猜到的那样，我们将注册一个
autograd 内核（类似于
 [自定义 autograd 函数](cpp_autograd) 
 教程中描述的’s）！
但是，有一个变化：与CPU 和 CUDA 内核，autograd 内核需要
 *redispatch* 
 ：它需要回调调度程序以访问
推理内核，例如CPU 或 CUDA 实现。




 因此，在我们编写 autograd 内核之前，让’s 编写一个
 *调度函数* 
，它调用调度程序来为您的操作员找到正确的内核。
这个函数构成了公共 C++ API对于您的运算符–事实上，PyTorch’s C++ API 中的所有
张量函数都在底层以相同的
方式调用调度程序。这里’是调度函数的样子：






```
Tensor myadd(const Tensor& self, const Tensor& other) {
 static auto op = torch::Dispatcher::singleton()
 .findSchemaOrThrow("myops::myadd", "")
 .typed<decltype(myadd)>();
 return op.call(self, other);
}

```




 让’s 分解它：



* 在第一行中，我们从调度程序
查找与我们要调度到的运算符相对应的类型化运算符句柄。
 `findSchemaOrThrow`
 接受两个参数：运算符的（命名空间限定的）名称
 ，以及运算符的重载名称（通常只是
空字符串）。
 `typed`
 将动态类型句柄转换为
静态类型句柄（进行运行时测试以确保’ve给定
正确的 C++ 类型），以便我们可以对其进行正常的 C++ 调用。我们传递它
 `decltype(myadd)`
 因为调度函数的类型
与注册到调度程序的底层内核的类型相同。




 为了提高性能，此计算是在静态变量中完成的，因此
我们只需要执行一次（缓慢的）查找。如果您输错了想要调用的运算符的名称，则在您第一次调用此函数时，此查找将出错。
* 在第二行中，我们只需
 `call`
 运算符句柄以及所有传递到调度函数的参数。这实际上将调用调度程序，并最终将控制权转移到适合此调用的任何内核。



 有了调度函数，我们现在可以编写 autograd 内核了：






```
class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
 public:
 static Tensor forward(
 AutogradContext *ctx, torch::Tensor self, torch::Tensor other) {
 at::AutoNonVariableTypeMode g;
 return myadd(self, other);
 }

 static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
 auto grad_output = grad_outputs[0];
 return {grad_output, grad_output};
 }
};

Tensor myadd_autograd(const Tensor& self, const Tensor& other) {
 return MyAddFunction::apply(self, other)[0];
}

```




 autograd 函数按照正常方式使用
 `torch::autograd::Function`
 编写，
除了直接在 
 `forward()`
 中编写实现之外，
我们:\ n


1. 使用
 `at::AutoNonVariableTypeMode`
 RAII
guard 关闭自动分级处理，然后
2.调用调度函数
 `myadd`
 回调调度程序。



 如果没有 (1)，您的调用将无限循环（并且堆栈溢出），因为
 `myadd`
 会将您送回此函数（因为最高优先级调度
key 仍将是 autograd。）使用 (1 ),
autograd 被排除在所考虑的调度键集合之外，
我们将转到下一个处理程序，该处理程序将是 CPU 和 CUDA。




 现在我们可以像注册 CPU/CUDA
函数一样注册这个函数：






```
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
 m.impl("myadd", myadd_autograd);
}

```





 没有10



 在此示例中，我们将内核注册到
 `Autograd`
 ，这会将其安装为
所有后端的
autograd 内核。您还可以使用相应的特定于后端的调度键为特定后端注册优化内核 - 例如，
 `AutogradCPU`
 或
 `AutogradCUDA`
 。要更详细地探索这些和其他调度键选项，请查看 [torch/_python_dispatcher.py](https://github.com/pytorch) 中提供的
 `PythonDispatcher`
 工具/pytorch/blob/master/torch/_python_dispatcher.py) 
.





## 超越 autograd [¶](#going-beyond-autograd "永久链接到此标题")




 从某种意义上说，调度程序’ 并没有做那么多事情：它所做的只是
实现一个美化的 if 语句，大致如下：






```
class MyAddFunction : ... {
public:
 static Tensor forward(
 AutogradContext *ctx, torch::Tensor self, torch::Tensor other) {

 if (self.device().type() == DeviceType::CPU) {
 return add_cpu(self, other);
 } else if (self.device().type() == DeviceType::CUDA) {
 return add_cuda(self, other);
 } else {
 TORCH_CHECK(0, "Unsupported device ", self.device().type());
 }
 }
 ...
}

```




 那么为什么要使用调度程序呢？有几个原因：



1.它是去中心化的。您可以组装运算符的所有部分（CPU、CUDA、Autograd），而无需编写引用所有这些部分的单个集中式语句。重要的是，第三方可以为其他方面注册额外的实现，而无需修补运算符的原始定义。我们’ 将在
 [为新后端扩展调度程序](extend_dispatcher)
 中详细讨论
扩展调度程序。
2.它支持比 CPU、CUDA 和 Autograd 更多的调度密钥。您可以在 `c10/core/DispatchKey.h`
 中查看 PyTorch 当前实现的调度密钥的完整列表。这些调度键
为操作员实现了各种可选功能，如果您
决定您的自定义操作员支持此功能，
您必须为相应的键注册一个内核。
3.调度程序实现了对盒装后备函数的支持，这些函数可以实现一次并应用于系统中的所有操作员。盒装回退可用于为调度键提供
默认行为；如果您使用调度程序来实现操作员，
您还可以选择所有这些操作的后备。



 以下是一些特定的调度键，您可能需要为其定义运算符。




### Autocast [¶](#autocast "此标题的永久链接")



 Autocast 调度键实现了对
 [自动混合精度 (AMP)](https://pytorch.org/docs/stable/amp.html) 
 的支持。
自动转换包装器内核通常会转换传入的
 ` float16`
 或
 `float32`
 CUDA 张量
在运行运算之前
达到某个首选精度。
例如，浮点 CUDA 张量上的 matmul 和卷积通常运行得更快
并且在
 `float16 中使用更少的内存`
 不会影响收敛。
Autocast 包装器仅在
 [启用 autocast 的上下文](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast) 中有效
.




 这里’ 是一个假设的自定义 matmul 的自动转换包装器，及其注册：






```
// Autocast-specific helper functions
#include <ATen/autocast_mode.h>

Tensor mymatmul_autocast(const Tensor& self, const Tensor& other) {
 c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
 return mymatmul(at::autocast::cached_cast(at::kHalf, self),
 at::autocast::cached_cast(at::kHalf, other));
}

TORCH_LIBRARY_IMPL(myops, Autocast, m) {
 m.impl("mymatmul", mymatmul_autocast);
}

```




`cached_cast(kHalf,
 

 张量)`
 将
 `张量`
 转换为
 `float16`
 如果
 `张量`
 是 CUDA 并且
 `float32`
 ，
 否则，
 `tensor`
 保持不变（参见
 [资格政策](https://pytorch.org/docs/stable/amp.html#op-eligibility ) 
 对于本机自动转换的操作）。
这确保网络是否在 
 `float16`
 和
 `float32`
 CUDA 张量、
 `mymatmul` 的任意混合上调用
 `mymatmul`
 
 在
 `float16`
 中运行。同时，使用非 CUDA、整数类型或“float64”输入调用
 `mymatmul`
 不受影响。建议使用
 `cached_cast`
 来遵循您自己的自动转换包装器中的本机资格策略，但不是必需的。例如，如果您想强制
 `float16`
 执行所有输入类型，
您可以
 `return
 

 mymatmul(self.half(),
 

 other)。 half());`
 而不是使用
 `cached_cast`
 。




 请注意，与我们的 autograd 内核一样，我们在重新分派之前从 
dispatch 中排除 
 `Autocast`
 键。




 默认情况下，如果未提供自动转换包装器，
我们将直接转至常规运算符实现（不
发生自动转换）。 （在这个例子中，’ 没有使用
 `myadd`
，因为逐点
加法不需要’ 需要自动转换，并且应该会失败。）




 什么时候应该注册 autocast 包装器？不幸的是，对于 op’s 的首选精度，没有’t
固定的规则。您可以通过查看
 [演员列表](https://pytorch.org/docs/master/amp.html#op-specific-behavior ) 
.
一般指导：



* 进行归约的操作可能应该在
 `float32`
 中执行，
* 任何在后台执行卷积或 gemm 的操作应该
可能在
 `float16`
 中执行，以及
* 其他操作具有多个浮点张量输入应该将它们标准化为通用精度（除非实现支持不同精度的输入）。



 如果您的自定义操作属于第三类，
 `promote_type`
 模板
有助于找出输入张量中存在的最宽的浮点类型，
这是执行类型的最安全选择： 






```
#include <ATen/autocast_mode.h>

Tensor my_multiple_input_op_autocast(const Tensor& t0, const Tensor& t1) {
 c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
 // The required at::kHalf argument is an optimistic initial guess.
 auto exec_type = at::autocast::promote_type(at::kHalf, t0, t1);
 return my_multiple_input_op(at::autocast::cached_cast(exec_type, t0),
 at::autocast::cached_cast(exec_type, t1));
}

```




 如果您的自定义操作是
 [autograd-enabled](#autograd-support)
 ，您只需编写并注册
 autocast 包装器，其名称与注册 autograd 包装器的名称相同。
对于例如，如果您想要自动分级部分中显示的
 `myadd`
 函数的自动转换包装器，则您’d 需要的是






```
Tensor myadd_autocast(const Tensor& self, const Tensor& other) {
 c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
 return myadd(at::autocast::cached_cast(<desired dtype>, self),
 at::autocast::cached_cast(<desired dtype>, other));
}

TORCH_LIBRARY_IMPL(myops, Autocast, m) {
 m.impl("myadd", myadd_autocast);
}

```




 没有单独的方法可以使后向方法自动转换兼容。
但是，自定义自动分级函数中定义的后向方法将以与前向方法的自动转换集相同的
d类型运行，因此您应该选择
 n `<desired
 

 dtype>`
 适用于您的前向和后向方法。





### 批处理 [¶](#batched "此标题的永久链接")



 批处理张量允许您以每个示例的方式编写代码，然后
在“vmap”调用下运行时自动对它们进行批处理。用于编写批处理规则的 API 目前正在开发中，但一旦稳定，您可以通过在批处理键处注册内核来为您的操作员添加对 vmap 的支持。





### Tracer [¶](#tracer "此标题的永久链接")



 Tracer 调度键实现了在运行
 `torch.jit.trace`
 时将运算符调用记录到跟踪中的支持。我们打算提供一个盒装后备方案，以实现对任意操作的跟踪，
请参阅
[问题 #41478](https://github.com/pytorch/pytorch/issues/41478)
 来跟踪
进度。\ n









