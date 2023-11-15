


 在 C++ 中扩展新后端的调度程序
 [¶](#extending-dispatcher-for-a-new-backend-in-c "永久链接到此标题")
============================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/extend_dispatcher>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/extend_dispatcher.html>




 在本教程中，我们将逐步完成扩展调度程序的所有必要步骤，
添加位于`pytorch/pytorch`
 存储库之外的新设备，并维护它以
与本机 PyTorch 设备保持同步。这里我们’假设您’熟悉如何
[在C++中注册调度运算符](调度程序)
以及如何编写
[自定义自动分级函数]( cpp_autograd) 
.





 注意




 本教程涉及 PyTorch 中的许多内部组件，这些组件正在积极改进，
如果您决定遵循本教程，请期待 API 的更改。我们’ 将使用最新的 API
更新本教程。






 ’ 是什么新后端？
 [¶](#what-s-a-new-backend "永久链接到此标题")
---------------------------------------------------------------------------------



 向 PyTorch 添加新后端需要后端扩展程序进行大量开发和维护。
在添加新后端之前，让’s 首先考虑一些常见用例和推荐的解决方案：



* 如果您对现有 PyTorch 运算符有新算法，请向 PyTorch 发送 PR。
* 如果您想提议新运算符，请向 PyTorch 发送功能请求/PR。
* 如果您想添加对某个运算符的支持新设备/硬件（例如 Google TPU 和定制芯片）通常需要使用
特定于硬件的 API 来编写内核，请按照本教程向 PyTorch 添加树外后端。
* 如果您想添加对现有的支持运算符，但具有不同的张量布局/表示
例如稀疏和量化，这强制您的内核以更高效’s的方式编写
考虑到布局/表示限制，请按照本教程并添加输出 - PyTorch 的 of-tree 后端。



 在本教程中，我们’ll 主要关注在下面添加一个新的树外设备。为不同的张量布局添加树外支持可能会与设备共享许多常见步骤，但我们还没有看到此类集成的示例，因此可能需要 PyTorch 进行额外的工作来支持它。 






 获取后端的调度密钥
 [¶](#get-a-dispatch-key-for-your-backend "永久链接到此标题")
---------------------------------------------------------------------------------------------------------------------------



 PyTorch 运算符是用 C++ 实现的，并通过 Python 绑定在 Python 前端提供。
PyTorch 调度程序将运算符的实现划分为多个内核，每个内核
与特定的调度键相关联。在 PyTorch 中支持新后端本质上意味着用 C++ 为每个 PyTorch 运算符编写
内核，然后将它们注册到表示调度程序中的
自定义后端的调度键。




 调度密钥是您在调度系统中的标识符。调度程序查看输入张量上携带的调度键并相应地调用正确的内核。 PyTorch 提供了三个保留的调度键
（及其相应的 Autograd 键），用于对树外后端扩展进行原型设计：



* PrivateUse1/AutogradPrivateUse1
* PrivateUse2/AutogradPrivateUse2
* PrivateUse3/AutogradPrivateUse3



 您可以选择上面的任何键来构建自定义后端的原型。
要在
 `PrivateUse1`
 后端创建 Tensor，您需要在
 `TensorImpl`
 构造函数中设置调度键。






```
/* Example TensorImpl constructor */
TensorImpl(
 Storage&& storage,
 DispatchKeySet ks,
 const caffe2::TypeMeta data_type);

// To create a TensorImpl on PrivateUse1 backend, pass in the following ks to TensorImpl creation.
DispatchKeySet ks = c10::DispatchKeySet{c10::DispatchKey::PrivateUse1, c10::DispatchKey::AutogradPrivateUse1};

```




 请注意，上面的 `TensorImpl`
 类假设您的 Tensor 由 CPU/CUDA 等存储支持。我们还为没有存储的后端提供了`OpaqueTensorImpl`。您可能需要调整/覆盖某些
方法以适应您的自定义硬件。
pytorch 存储库中的一个示例是
 [Vulkan TensorImpl](https://github.com/pytorch/pytorch/blob/1.7/aten/src /ATen/native/vulkan/VulkanOpaqueTensorImpl.h) 
.





 注意




 原型完成后，您计划定期发布后端扩展，请随时向
提交 PR 到
 `pytorch/pytorch`
 为您的后端保留专用的 dispath 密钥。







 获取 PyTorch 运算符的完整列表
 [¶](#get-the-full-list-of-pytorch-operators "永久链接到此标题")
---------------------------------------------------------------------------------------------------------------------



 PyTorch 在生成的文件中提供了可扩展 C++ 运算符的完整列表
 `build/aten/src/ATen/RegistrationDeclarations.h`
 。
此文件仅在从源代码构建 PyTorch 后可用。
此处\xe2\x80 \x99s 文件的片段：






```
Tensor abs(const Tensor & self); // {"schema": "aten::abs(Tensor self) -> Tensor", "dispatch": "True", "default": "True"}
Tensor & abs_(Tensor & self); // {"schema": "aten::abs_(Tensor(a!) self) -> Tensor(a!)", "dispatch": "True", "default": "True"}
Tensor & abs_out(Tensor & out, const Tensor & self); // {"schema": "aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
Tensor absolute(const Tensor & self); // {"schema": "aten::absolute(Tensor self) -> Tensor", "dispatch": "False", "default": "False"}
Tensor & absolute_(Tensor & self); // {"schema": "aten::absolute_(Tensor(a!) self) -> Tensor(a!)", "dispatch": "False", "default": "False"}
Tensor & absolute_out(Tensor & out, const Tensor & self); // {"schema": "aten::absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "False", "default": "False"}
Tensor angle(const Tensor & self); // {"schema": "aten::angle(Tensor self) -> Tensor", "dispatch": "True", "default": "True"}
Tensor & angle_out(Tensor & out, const Tensor & self); // {"schema": "aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
Tensor sgn(const Tensor & self); // {"schema": "aten::sgn(Tensor self) -> Tensor", "dispatch": "True", "default": "True"}

```




’ 有多个字段与单个运算符关联。让’s 使用
 `abs_out`
 作为示例对其进行分解：



* `张量
 

 &
 

 abs_out(张量
 

 &
 

 out,
 

 const
 

 Tensor
 

 &
 

 self);`
 是运算符的 C++ 签名，您的 C++
内核应该与此签名完全匹配。
* `aten::abs.out(Tensor 
 

 self,
 

 *,
 

 张量(a!)
 

 out)
 

 ->
 
\ n Tensor(a!)`
 是表示运算符的唯一模式，
与 C++ 签名相比，它还包含别名和突变注释。这是调度程序用于查找操作员的唯一标识符。
* `dispatch`
 和
 `default`
 是布尔字段，提供有关本机 PyTorch 内核
可以做什么的信息，因此暗示它是否
后端扩展程序实现内核所需的 xe2\x80\x99s。
更多详细信息可以在
 [为新后端注册内核](#register-kernel)
 中找到。







 为新后端注册内核
 [¶](#register-kernels-for-the-new-backend "永久链接到此标题")
----------------------------------------------------------------------------------------------------------------



 要将内核注册到 PyTorch 调度程序，您可以使用 
 `TORCH_LIBRARY_IMPL`
 API（在 C++ 中注册调度操作符）
 中描述的 API 
 :






```
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
 m.impl(<schema_my_op1>, &my_op1);
 m.impl(<schema_my_op2>, &my_op2);
 m.impl(<schema_my_op2_backward>, &my_op2_backward);
}

```




 现在让’s 放大，了解哪些运算符需要来自自定义后端的内核以及内核中的’s\ 到底是什么。




 PyTorch 目前拥有超过 1600 个运算符，并且’ 仍在增长。对于后端扩展来说，要跟上这个速度’ 是不现实的。即使对于 CPU\或 CUDA 等本机后端，通常也需要大量工作来为每个新操作编写专用内核。




 幸运的是，一些本机 PyTorch 内核的编写方式可以分解为
几个已知运算符的组合。换句话说，您只需要实现
一组已知运算符（需要在下面注册的操作），而不是所有 PyTorch 运算符。




 PyTorch 运算符可以分为两类：



* 需要注册的操作：这些操作的 PyTorch 本机实现是特定于后端的，因此需要为自定义后端提供内核。否则在自定义后端调用此类操作将会出错。




> 
> 
> 
> 	+ In
> 	 `RegistrationDeclarations.h`
> 	 这些运算符已
> 	 `dispatch`
> 	 设置为 True\ n> 	 *and* 
> 	`default`
> 	 设置为 False
> 	in 在其随附注释中找到的元数据。
> 
>
* 注册是可选的：后端扩展程序可以跳过注册到这些操作而不牺牲任何支持。
但是，如果后端扩展程序想要覆盖 PyTorch 提供的默认内核，他们仍然可以
将其自定义内核注册到其后端，并且调度程序将仅将其用于您的后端.
例如，PyTorch’s
 `max_pool2d`
 的当前实现返回
 `indices`
 作为前向输出的一部分，
这会在 torch_xla 中产生开销，因此 torch _xla 为
 `max_pool2d`
 注册了自己的内核。




> 
> 
> 
> 	+ In
> 	 `RegistrationDeclarations.h`
> 	 这些运算符已
> 	 `dispatch`
> 	 设置为 False\ n> 	 *或* 
> 	`default`
> 	 设置为 True
> 	in 在其随附注释中找到的元数据。
> 
>





 对新后端的 Autograd 支持
 [¶](#autograd-support-for-the-new-backend "永久链接到此标题")
----------------------------------------------------------------------------------------------------------------------------



 梯度公式大多是纯数学的，因此对所有后端都是通用的。
PyTorch 经常将内核注册为别名调度键 Autograd，这意味着它可以被所有后端使用。




 对于这些运算符，’ 不必担心它们的导数公式，
你只需在
 `RegistrationDeclarations.h` 中编写运算符的前向定义，PyTorch 就会自动为你处理
后向。\ n





```
Tensor my_op1(const Tensor& self, const Tensor& other) {
 // call your backend-specific APIs to implement my_op so that
 // it matches PyTorch's native behavior
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
 m.impl(<schema_my_op1>, &my_op);
}

```




 在某些情况下，PyTorch 向后内核实现也是特定于设备的，因此它们可以从每个后端中挤出
最大性能。对于这些运算符，’ 会看到 op_backward 出现在
 `RegistrationDeclarations.h`
 中，
 也显示为
 *需要注册*
。






```
Tensor my_op2_backward(const Tensor& self, const Tensor& other) {
 // call your backend-specific APIs to implement my_op2_backward so that
 // it matches PyTorch's native behavior
}

// Note backward kernel is still registered to PrivateUse1 instead of AutogradPrivateUse1.
// PyTorch will wrap your backward kernel with proper autograd setup and then link to it in
// my_op2's AutogradPrivateUse1 kernel.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
 m.impl(<schema_my_op2>, &my_op2);
 m.impl(<schema_my_op2_backward>, &my_op2_backward);
}

```




 在少数
 *罕见* 
 情况下，某些运算符的 PyTorch’s 梯度公式可能会假设不’t 泛化
所有后端。在这些情况下，后端扩展程序可以选择通过将 torch::autograd::Function 中的内核注册到相应的调度键来覆盖 PyTorch Autograd 层（例如，如果您
you’ 在后端使用 PrivateUse1，则为 AutogradPrivateUse1）： 






```
class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
 public:
 static Tensor forward(AutogradContext *ctx, torch::Tensor self, torch::Tensor other) {
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

// Register the autograd kernel to AutogradPrivateUse1
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
 m.impl(<myadd_schema>, &myadd_autograd);
}

// Register the inference kernel to PrivateUse1
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
 m.impl(<myadd_schema>, &myadd);
}

```




 通过这个技巧，您可以完全控制后端中`my_add`
 运算符的训练和推理行为。
这里’s
 [示例](https://github.com/pytorch/xla/blob/r1.7/torch_xla/csrc/aten_autograd_ops.h) 
 在
 `pytorch/xla`
 存储库中。






 构建扩展
 [¶](#build-an-extension "永久链接到此标题")
--------------------------------------------------------------------------- -



 通过向 PyTorch 添加 C++ 扩展来支持树外后端。
一旦准备好内核和注册，您就可以通过
编写一个
 `setup.py`
 脚本来构建 C++ 扩展，该脚本使用\ n `setuptools`
 编译 C++ 代码。这里’是来自
 [pytorch/xla repo](https://github.com/pytorch/xla/blob/master/setup.py)的简化示例
：






```
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='torch_xla',
    ext_modules=[
        CppExtension(
            '_XLAC',
            torch_xla_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            extra_link_args=extra_link_args + \
                [make_relative_rpath('torch_xla/lib')],
        ),
    ],
    cmdclass={
        'build_ext': Build,  # Build is a derived class of BuildExtension
    }
    # more configs...
)

```




 请参阅
 [我们的 C++ 扩展教程](https://pytorch.org/tutorials/advanced/cpp_extension.html#building-with-setuptools) 
 了解更多详细信息。






 自定义运算符支持
 [¶](#custom-operator-support "永久链接到此标题")
----------------------------------------------------------------------------------------



 您的新后端应该与 [Python 中扩展的自定义运算符](https://pytorch.org/docs/stable/notes/extending.html) 无缝协作 
 只要自定义运算符，就无需编写任何新内核由现有的
PyTorch 运算符组成（您的后端已支持这些运算符）。




 对于
 [在 C++ 中扩展的自定义运算符](cpp_autograd) 
 它们通常带有
 [后端特定的 C++ 内核实现，例如torchvsion 中的 nms 内核](https://github.com/pytorch/vision/blob/master/torchvision/csrc/ops/cuda/nms_kernel.cu) 
 以及
 [定制的 Python API，例如torch.ops.torchvision.nms](https://github.com/pytorch/vision/blob/master/torchvision/csrc/ops/nms.cpp#L18) 
.
为了支持这些运算符，后端扩展程序将需要为后端编写一个 C++ 内核，并
将其正确注册到调度程序中相应的命名空间，类似于支持 PyTorch 本机运算符。
或者，您还可以在扩展中添加自定义 API，例如
 `torch_xla.core。 Functions.nms`
 用于这些临时请求。






 JIT 支持
 [¶](#jit-support "固定链接到此标题")
-----------------------------------------------------------



 正如我们在
 [在 C++ 中注册 Dispatched Operator](dispatcher) 中提到的，通过
 
 m.impl()
 
 API 注册的内核支持以未装箱和装箱的方式调用。换句话说，您的自定义后端也可以与我们的
JIT 跟踪/脚本前端配合使用，就像 CPU 或 CUDA 等树内后端一样。您还可以在 JIT 图上为后端编写专门的优化
通道。但我们不会在这里讨论它，因为我们还没有在 JIT 中’ 确定集成点，因此当前的后端支持将暂时集中在 eager 前端。






 针对本机 PyTorch 后端测试您的后端
 [¶](#testing-your-backend-against-native-pytorch-backends "永久链接到此标题")
-------------------------------------------------------------------------------------------------------------------------------------------------------------



 PyTorch 使用其
 [通用设备类型测试框架](https://github.com/pytorch/pytorch/blob/master/torch/testing/_internal/common_device_type.py) 让测试在多种设备类型上运行
.
您可以找到有关
 [测试如何使用它](https://github.com/pytorch/pytorch/blob/5a8198eb3c594aa18352930fd21f3c25bd7b7100/torch/testing/_internal/common_device_type.py#L23) 的详细信息以及有关
的信息n [如何添加新的设备类型](https://github.com/pytorch/pytorch/blob/5a8198eb3c594aa18352930fd21f3c25bd7b7100/torch/testing/_internal/common_device_type.py#L369) 
.
添加后，PyTorch 会使用通用设备类型测试框架也将使用您的设备类型运行。
请参阅
 [此 Wiki 页面](https://github.com/pytorch/pytorch/wiki/Writing-tests-that-run-on-all -available-device-types) 
 有关如何实例化测试的示例。




 使用您的设备类型运行 PyTorch’s 现有测试套件对于确保正确性非常重要，
但并非每种设备类型都支持所有 PyTorch 功能。通用设备类型测试
框架允许进行大量自定义，以便设备类型可以选择要运行的测试、
它们支持的数据类型，甚至在比较张量是否相等时使用哪些精度。




 使用通用设备类型测试框架且不随 PyTorch 一起提供的示例设备类型是 XLA。请参阅
 [通用设备类型测试框架的扩展](https://github.com/pytorch/xla/blob/master/test/pytorch_test_base.py)
 ，
其中包含块列表测试、块的示例列出数据类型，并覆盖测试精度。




 通用设备类型测试框架正在积极开发中。要请求功能，请在 PyTorch’s Github 上
提交问题。






 向后兼容性
 [¶](#backward-compatibility "永久链接到此标题")
----------------------------------------------------------------------------------



 目前 PyTorch 无法’ 保证注册运算符的向后兼容性。
可以根据需要添加/修改/删除运算符及其架构。注册的
内核必须
 *完全* 
 与 PyTorch 版本相同。如果 PyTorch 为操作员添加更多参数（
即使使用默认值），您的旧注册将’ 无法工作，直到’ 更新
以匹配 PyTorch’ 的新签名。




 因此，我们
 *强烈建议* 
 树外后端扩展程序仅与主要 PyTorch
 版本同步，以最大程度地减少开发中断。 PyTorch 按季度发布节奏。
后端扩展程序应加入
 *#announcement* 
 频道
 [pytorch.slack.com](http://pytorch.slack.com/) 
 以获取最新版本版本更新。






 已知问题和附加说明
 [¶](#known-issues-additional-notes "此标题的永久链接")
----------------------------------------------------------------------------------------------------


* 并非所有测试套件都是设备通用的。可以通过在 PyTorch 代码库中搜索
 `instantiate_device_type_tests`
 找到可扩展的测试类，例如
 `TestTorchDeviceType,
 

 TestViewOps,
 

 TestTensorDeviceOps, 
 

 TestTypePromotion`
 等
* C++ 中没有用于在自定义后端序列化 python Tensor 对象的扩展点。目前
只能通过修改
 [PyTorch Tensor __reduce_ex__方法](https://github.com/pytorch/pytorch/blob/5640b79bf8a5412a0209a919c05c811d5427cc12/torch/tensor.py#L83-L150) 
 或在树外存储库中进行猴子修补。
* 如果您的后端’t 不允许直接内存访问，则应额外注意支持
视图操作，因为他们’应该共享存储。对视图张量的更改需要传播到其
基张量，反之亦然。
* 如果您的后端’ 不能与本机 PyTorch 一起使用，则优化器的 C++ 中’ 没有扩展点
优化器，例如需要像 torch-xla 一样向后携带要更新的状态。目前，此类用例只能通过在树外存储库中添加自定义 API 或猴子修补来完成。





 未来的工作
 [¶](#future-work "此标题的永久链接")
---------------------------------------------------------------------------



 使 PyTorch 中的每个组件都可扩展以实现树外后端无缝
需要对 PyTorch 内部进行大量更改。以下是我们’正在积极研究的一些项目，可能会改善未来的体验：



* 提高通用测试框架的测试覆盖率。
* 提高
 `Math`
 内核覆盖率和更全面的测试，以确保
 `Math`
 内核行为与其他后端匹配
 `CPU/CUDA`\ n.
* 重构
 `RegistrationDeclarations.h`
 以携带最少的信息并尽可能重用
PyTorch’s 代码生成器。
* 支持后端回退内核以自动将输入转换为 CPU并将结果转换回自定义后端。即使您没有为每个运算符编写内核，这也将允许 “full” 运算符覆盖。





 保持联系
 [¶](#stay-in-touch "此标题的永久链接")
-----------------------------------------------------------------------------



 请使用
 [PyTorch 开发讨论](https://dev-discuss.pytorch.org/) 
 提出问题和讨论。如果您
有任何功能请求或错误报告，请
[在 github 上提交问题](https://github.com/pytorch/pytorch/issues)
 。




 如果您’ 有兴趣帮助完成上述任何未来工作项目（例如为 C++ 中的 PyTorch 运算符添加更多
 `Math`
 内核），请通过 Github 或 Slack 与我们联系！ 









