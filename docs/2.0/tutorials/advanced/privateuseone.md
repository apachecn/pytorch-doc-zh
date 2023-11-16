


 通过 PrivateUse 促进新后端集成1
 [¶](#facilitating-new-backend-integration-by-privateuse1 "永久链接到此标题")
==========================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/privateuseone>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/privateuseone.html>




 在本教程中，我们将逐步完成一些必要的步骤来集成一个新的后端
生活在
 `pytorch/pytorch`
 repo 
 `PrivateUse1`
 之外。请注意，本教程假设
您已经对 PyTorch 有基本的了解。
您是 PyTorch 的高级用户。





 注意




 本教程仅涉及与PrivateUse1机制相关的部分，方便新设备的集成，
其他部分不再涉及。同时，本教程涉及的模块并非都是必需的，
您可以根据自己的实际需要选择对您有帮助的模块。






 什么是 PrivateUse1？
 [¶](#what-is-privateuse1 "此标题的永久链接")
---------------------------------------------------------------------------------



 在 Pytorch 2.0 之前，PyTorch 提供了三个保留的调度键（及其相应的 Autograd 键）
用于对树外后端扩展进行原型设计，这三个调度键如下：



* `PrivateUse1/AutogradPrivateUse1`
* `PrivateUse2/AutogradPrivateUse2`
* `PrivateUse3/AutogradPrivateUse3`



 原型验证通过后，您可以为新后端申请私钥，例如CUDA、XLA、MPS等。




 然而，随着 PyTorch 的快速发展，越来越多的硬件制造商
尝试将其后端集成到 PyTorch 中，这可能会导致以下问题：



* 每个新的后端集成都涉及大量文件修改
* 目前对调度密钥的数量有硬性限制（
 `DispatchKeySet`
 64 位限制）




 注意



通过 PrivateUse1 Key 将新后端集成到 PyTorch 中还存在一个问题，因为不可能同时集成许多后端。幸运的是，这些树外后端很少同时使用。





 鉴于上述原因，社区开始推荐通过
 `PrivateUse1`
 集成新的后端到 PyTorch
 。




 然而，之前的
 `PrivateUse1`
 机制并不完全能够与新后端集成，因为它
在某些模块中缺乏一些相关支持，例如Storage、AMP、Distributed 等。




随着Pytorch 2.1.0的到来，
`PrivateUse1`在新的后端集成方面进行了一系列的优化和增强，现在可以快速支持新设备的集成且高效。






 如何通过 PrivateUse1
 [¶](#how-to-integrate-new-backend-via-privateuse1 "永久链接到此标题")
------------ 集成新后端----------------------------------------------------------------------------------------------------------------------



 在本节中，我们将讨论通过
 `PrivateUse1` 将新后端集成到 Pytorch 的细节，
主要由以下部分组成：



1. 为新后端注册内核。
2.为新后端注册生成器。
3.为新后端注册设备防护。
4.为新的后端元数据注册序列化和反序列化函数。
5.其他模块。



### 
 为新后端注册内核
 [¶](#register-kernels-for-the-new-backend "永久链接到此标题")



 新的后端可能有一些高性能的操作符实现，可以通过
`TORCH_LIBRARY_IMPL`
 API 注册到调度程序

[在 C++ 中注册调度操作符] （调度员）
 。这涉及到
几种情况：



1. 将新后端支持的所有前向算子注册到调度程序，同时注册回退
，这样当新后端不支持某些算子时，这些算子可以回落
到CPU执行确保功能的可用性。





```
at::Tensor wrapper_Custom_Tensor_add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
 // Implementation of add kernel in new backend
 ...
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
 ...
 m.impl("add.Tensor", TORCH_FN(wrapper_Custom_Tensor_add));
 ...
}

void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
 // Add some hints about new devices that do not support and need to fall back to cpu
 at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
 m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}

```



2.如果新后端需要覆盖
 `PyTorch
 

 Autograd
，则通过
 `AutogradPrivateUse1`
 将内核从
 `torch::autograd::Function`
 注册到调度程序
 n 

 层`
 ，调度程序和 autograd 系统会自动调用这些算子的前向和后向实现。





```
class CumtomSeluFunction : public torch::autograd::Function<CumtomSeluFunction> {
 // Implementation of selu kernel in new backend
}

at::Tensor wrapper_AutogradCumstom__selu(const at::Tensor & self) {
 return CumtomSeluFunction::apply(self);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
 ...
 m.impl("selu", TORCH_FN(wrapper_AutogradCustom__selu));
 ...
}

```



3.注册想要支持
 [自动混合精度 (AMP)](https://pytorch.org/docs/stable/amp.html) 的内核，并通过
 `AutocastPrivateUse1`
 向调度程序注册回退机制
 ，autocast系统会在需要的时候自动调用这些内核。





```
TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
 ...
 KERNEL_PRIVATEUSEONE(<operator>, <policy>)
 ...
}

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
 m.fallback(torch::CppFunction::makeFallthrough());
}

```




 需要补充的是，如果想在新的后端支持 AMP，需要通过
 `torch._register_device_module 注册一个新的
 `BackendModule`
 ("backend_name",
 

 BackendModule)`
 ，并且
 `BackendModule`
 需要具有以下 API:



* `get_amp_supported_dtype()
 

 ->
 

 List[torch.dtype]`



 获取 AMP 中新后端支持的 dtype，该后端可能还支持一种
 `dtype`
 。
* `is_autocast_enabled()
 

 ->
 \ n
 布尔`



 检查新后端上是否启用了 AMP。
* `get_autocast_dtype()
 

 ->
 

 torch.dtype`



 获取 AMP 中新后端支持的
 `dtype`
，该值由
 `set_autocast_dtype`
 或
默认
 `dtype`
 设置，并且默认
 `dtype`
 为
 `torch.float16`
.
* `set_autocast_enabled(bool)
 

 ->
 

 None`



 在新后端上启用或禁用 AMP。
* `set_autocast_dtype(dtype)
 

 ->
 

 None`



 在 AMP 中的新后端上设置支持的
 `dtype`
，并且
 `dtype`
 包含在
 `dtypes`
 从
 `get_amp\ 中获取
 \_supported_dtype`
 。




### 
 为新后端注册生成器
 [¶](#register-generator-for-the-new-backend "永久链接到此标题")



 需要支持新设备对应的生成器。目前
 `PrivateUse1`
可以动态
注册自定义生成器，主要分为以下几个步骤。



1. 继承
 `GeneratorImpl`
类，实现新后端对应的生成器类，
并实现各种通用方法。
2.使用单个参数定义新的后端
 `构建器`
:
 `设备
 

 索引`
 。
3.调用
 `REGISTER_GENERATOR_PRIVATEUSE1`
 宏完成动态注册。





```
struct CustomGeneratorImpl : public c10::GeneratorImpl {
 // Implementation of generator in new backend
}

at::Generator make_custom_generator(c10::DeviceIndex device_index) {
 return at::make_generator<CustomGeneratorImpl>(device_index);
}

REGISTER_GENERATOR_PRIVATEUSE1(make_cumstom_generator)

```





### 
 为新后端注册设备防护
 [¶](#register-device-guard-for-the-new-backend "永久链接到此标题")



 PyTorch 通过
 `DeviceGuard`
 提供与设备、流和事件切换相关的功能。
此功能也适用于
 `PrivateUse1`
 Key。



1.继承
 `DeviceGuardImplInterface`
类，实现新后端对应的各种通用方法。
2.调用
 `C10_REGISTER_GUARD_IMPL`
宏完成动态注册。





```
struct CustomGuardImpl final : public c10::impl::DeviceGuardImplInterface {
 // Implementation of guard in new backend
}

C10_REGISTER_GUARD_IMPL(PrivateUse1, CustomGuardImpl);

```





### 
 为新后端元数据注册序列化和反序列化函数
 [¶](#register-serialization-and-deserialization-functions-for-new-backend-metadata "永久链接到此标题")



 PyTorch 目前能够动态注册序列化/反序列化函数，以支持
类中名为
 `backend_meta_`
 的新后端附加元数据

 `TensorImpl.ExtraMeta` 的序列化和反序列化
 。您可以参考以下步骤：



1. 继承
 `BackendMeta`
类，实现新后端对应的
 `CustomBackendMetadata`
，并可在该类中自定义新后端的各个字段。
2.实现新后端的序列化和反序列化函数，函数签名为
 `void(const
 

 at::Tensor&,
 

 std::unordered_map<std::string ,


 布尔>&)`
.
3.调用`TensorBackendMetaRegistry`宏完成动态注册。





```
struct CustomBackendMetadata : public c10::BackendMeta {
 // Implementation of backend metadata in new backend
}

void for_serialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
 // Implementation of serialization
}

void for_deserialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
 // Implementation of deserialization
}

TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1, &for_serialization, &for_deserialization);

```





### 
 其他模块
 [¶](#other-modules "此标题的永久链接")



 除了上述部分之外，还有一些其他模块可以通过
 `PrivateUse1`
 进行扩展，
比如
 `分布式
 

 集体
 

 通信`
 、
 `benchmark
 

 timer`
 以及其他，将来会添加。
关于
 `PrivateUse1`
 集成的一个例子是
 [Ascend NPU]( https://github.com/ascend/pytorch) 
.







 如何通过 Privateuse1 改善用户体验
 [¶](#how-to-improve-user-experience-with-privateuse1 "永久链接到此标题")
---------------------------------------------------------------------------------------------------------------------------------------



 通过`PrivateUse1`集成新设备的首要目标是满足基本的功能需求，
接下来要做的是提高可用性，主要涉及以下几个方面。



1. 向 Pytorch 注册新的后端模块。
2.生成与新后端相关的方法和属性。
3.生成与新后端相关的方法和属性。



### 
 将新后端模块注册到 Pytorch
 [¶](#register-new-backend-module-to-pytorch "永久链接到此标题")



 PyTorch 中一些 CUDA 相关的接口可以通过以下形式调用：
 `torch.cuda.xxx`
 。因此，为了符合用户习惯，通过`PrivateUse1`机制实现的新后端也应该提供类似的接口。




 例如，使用
 `Ascend
 

 NPU`
 :






```
torch._register_device_module('npu', torch_npu.npu)

```




 完成上述操作后，用户可以通过
 `torch.npu.xxx` 调用
 `Ascend
 

 NPU` 的一些专有API





### 
 将 PrivateUse1 重命名为新后端的自定义名称
 [¶](#rename-privateuse1-to-a-custom-name-for-the-new-backend "永久链接到此标题" ）



`PrivateUse1`
 Key 是集成到 PyTorch 中的新后端的内部机制。对于用户来说，与
 `PrivateUse1`
 相比，
与新后端强相关的自定义名称应该更加友好。




 以
 `Ascend
 

 NPU`
 为例，第一次使用会更方便。






```
torch.rand((2,2),device='npu:0')
torch.rand((2,2),device='privateuse1:0')

```




 现在，PyTorch 为自命名的
 `PrivateUse1`
 后端提供了一个新的 C++/Python API，使用起来非常简单。






 Python





```
torch.rename_privateuse1_backend("npu")

```






 C++





```
c10::register_privateuse1_backend("npu")

```







### 
 生成与新后端相关的方法和属性
 [¶](#generate-methods-and-properties-lated-to-the-new-backend "此标题的永久链接”）



 将
 `PrivateUse1`重命名为自定义名称后，
 自动生成与新后端名称相关的属性和方法
 `Tensor、
 

 nn、
 

 Storage `
 新后端的模块。




 以下是
 `Ascend
 

 NPU` 的示例
 :






```
torch.rename_privateuse1_backend("npu")
unsupported_dtype = [torch.quint8]
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True, unsupported_dtype=unsupported_dtype)

```




 然后，您可以使用以下方法和属性:






```
torch.Tensor.npu()
torch.Tensor.is_npu
torch.Storage.npu()
torch.Storage.is_npu
...

```







 未来的工作
 [¶](#future-work "永久链接到此标题")
---------------------------------------------------------------



 `PrivateUse1`
 机制的完善仍在进行中，因此将依次添加新模块的
 `PrivateUse1`
 集成方法。以下是我们正在积极处理的一些项目：



* 添加
 `分布式
 

 集体
 

 通信`
 的集成方法。
* 添加
 `benchmark
 

 定时器`的集成方法
 。





 结论
 [¶](#conclusion "此标题的永久链接")
---------------------------------------------------------------------



 本教程引导您完成通过
 `PrivateUse1`
 将新后端集成到 PyTorch 的过程，包括但不限于
操作员注册、生成器注册、设备防护注册等。同时引入了一些
改善用户体验的方法。









