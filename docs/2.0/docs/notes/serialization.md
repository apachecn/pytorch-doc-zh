# [序列化语义](#id2) [¶](#serialization-semantics "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/serialization>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/serialization.html>


 本说明介绍了如何在 Python 中保存和加载 PyTorch 张量和模块状态，以及如何序列化 Python 模块以便可以在 C++ 中加载它们。


 目录



* [序列化语义](#serialization-semantics)
  + [保存和加载张量](#saving-and-loading-tensors) 
  + [保存和加载张量保留视图](#saving-and-loading-tensors-preserves-views) 
  + [保存和加载 torch.nn.Modules ](#saving-and-loading-torch-nn-modules) 
  + [序列化 torch.nn.Modules 并在 C++ 中加载它们](#serializing-torch-nn-modules-and-loading-them-in-c) 
  + [跨 PyTorch 版本保存和加载 ScriptModules](#saving-and-loading-scriptmodules-across-pytorch-versions) 
    - [torch.div 执行整数除法](#torch-div-performing-integer-division) 
    - [torch.full 总是推断 float dtype](#torch-full-always-inferring-a-float-dtype)
  + [实用函数](#utility-functions)


## [保存和加载张量](#id3) [¶](#saving-and-loading-tensors "永久链接到此标题")


[`torch.save()`](../generated/torch.save.html#torch.save "torch.save") 和 [`torch.load()`](../generated/torch.load.html#torch.load "torch.load") 让您轻松保存和加载张量：


```
>>> t = torch.tensor([1., 2.])
>>> torch.save(t, 'tensor.pt')
>>> torch.load('tensor.pt')
tensor([1., 2.])

```


 按照惯例，PyTorch 文件通常使用“.pt”或“.pth”扩展名编写。


[`torch.save()`](../generated/torch.save.html#torch.save "torch.save") 和 [`torch.load()`](../generated/torch.load.html#torch.load "torch.load") 默认使用 Python 的 pickle，因此您还可以将多个张量保存为 Python 对象的一部分，例如元组、列表和字典：


```
>>> d = {'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.])}
>>> torch.save(d, 'tensor_dict.pt')
>>> torch.load('tensor_dict.pt')
{'a': tensor([1., 2.]), 'b': tensor([3., 4.])}

```


 如果数据结构是可pickle的，则还可以保存包含PyTorch张量的自定义数据结构。


## [保存和加载张量保留视图](#id4) [¶](#saving-and-loading-tensors-preserves-views "永久链接到此标题")


 保存张量会保留它们的视图关系：


```
>>> numbers = torch.arange(1, 10)
>>> evens = numbers[1::2]
>>> torch.save([numbers, evens], 'tensors.pt')
>>> loaded_numbers, loaded_evens = torch.load('tensors.pt')
>>> loaded_evens *= 2
>>> loaded_numbers
tensor([ 1, 4, 3, 8, 5, 12, 7, 16, 9])

```


 在幕后，这些张量共享相同的“存储”。有关视图和存储的更多信息，请参阅[张量视图](https://pytorch.org/docs/main/tensor_view.html)。


 当 PyTorch 保存张量时，它会分别保存它们的存储对象和张量元数据。这是一个将来可能会改变的实现细节，但它通常可以节省空间，并让 PyTorch 轻松重建加载的张量之间的视图关系。例如，在上面的代码片段中，只有一个存储被写入“tensors.pt”。


 然而，在某些情况下，保存当前存储对象可能是不必要的，并且会创建过大的文件。在下面的代码片段中，比保存的张量大得多的存储被写入文件中：


```
>>> large = torch.arange(1, 1000)
>>> small = large[0:5]
>>> torch.save(small, 'small.pt')
>>> loaded_small = torch.load('small.pt')
>>> loaded_small.storage().size()
999

```


 与仅将小张量中的五个值保存到“small.pt”不同，它与大张量共享的存储中的 999 个值被保存并加载。


 当保存元素少于其存储对象的张量时，可以通过首先克隆张量来减小保存的文件的大小。克隆张量会生成一个新张量，其中包含一个仅包含张量中的值的新存储对象：


```
>>> large = torch.arange(1, 1000)
>>> small = large[0:5]
>>> torch.save(small.clone(), 'small.pt')  # saves a clone of small
>>> loaded_small = torch.load('small.pt')
>>> loaded_small.storage().size()
5

```


 然而，由于克隆张量彼此独立，因此它们没有原始张量所具有的视图关系。如果在保存小于其存储对象的张量时文件大小和视图关系都很重要，则必须注意构造新的张量，以最小化其存储对象的大小，但在保存之前仍然具有所需的视图关系。


## [保存和加载 torch.nn.Modules](#id5) [¶](#saving-and-loading-torch-nn-modules "永久链接到此标题")


 另请参阅：[教程：保存和加载模块](https://pytorch.org/tutorials/beginner/saving_loading_models.html)


 在 PyTorch 中，模块的状态经常使用“状态字典”进行序列化。模块的状态字典包含其所有参数和持久缓冲区：


```
>>> bn = torch.nn.BatchNorm1d(3, track_running_stats=True)
>>> list(bn.named_parameters())
[('weight', Parameter containing: tensor([1., 1., 1.], requires_grad=True)),
 ('bias', Parameter containing: tensor([0., 0., 0.], requires_grad=True))]

>>> list(bn.named_buffers())
[('running_mean', tensor([0., 0., 0.])),
 ('running_var', tensor([1., 1., 1.])),
 ('num_batches_tracked', tensor(0))]

>>> bn.state_dict()
OrderedDict([('weight', tensor([1., 1., 1.])),
 ('bias', tensor([0., 0., 0.])),
 ('running_mean', tensor([0., 0., 0.])),
 ('running_var', tensor([1., 1., 1.])),
 ('num_batches_tracked', tensor(0))])

```


 出于兼容性原因，建议不要直接保存模块，而是仅保存其状态字典。 Python 模块甚至有一个函数， [`load_state_dict()`](../generated/torch.nn.Module.html#torch.nn.Module.load_state_dict "torch.nn.Module.load_state_dict") ，从状态指令恢复它们的状态：


```
>>> torch.save(bn.state_dict(), 'bn.pt')
>>> bn_state_dict = torch.load('bn.pt')
>>> new_bn = torch.nn.BatchNorm1d(3, track_running_stats=True)
>>> new_bn.load_state_dict(bn_state_dict)
<All keys matched successfully>

```


 请注意，状态字典首先使用 [`torch.load()`](../generated/torch.load.html#torch.load "torch.load") 从其文件中加载，然后使用 [`load_state_dict()`](../generated/torch.nn.Module.html#torch.nn.Module.load_state_dict "torch.nn.Module.load_state_dict") 恢复状态。


 即使自定义模块和包含其他模块的模块也具有状态字典，并且可以使用此模式：


```
# A module with two linear layers
>>> class MyModule(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(4, 2)
        self.l1 = torch.nn.Linear(2, 1)

      def forward(self, input):
        out0 = self.l0(input)
        out0_relu = torch.nn.functional.relu(out0)
        return self.l1(out0_relu)

>>> m = MyModule()
>>> m.state_dict()
OrderedDict([('l0.weight', tensor([[ 0.1400, 0.4563, -0.0271, -0.4406],
                                   [-0.3289, 0.2827, 0.4588, 0.2031]])),
             ('l0.bias', tensor([ 0.0300, -0.1316])),
             ('l1.weight', tensor([[0.6533, 0.3413]])),
             ('l1.bias', tensor([-0.1112]))])

>>> torch.save(m.state_dict(), 'mymodule.pt')
>>> m_state_dict = torch.load('mymodule.pt')
>>> new_m = MyModule()
>>> new_m.load_state_dict(m_state_dict)
<All keys matched successfully>

```


## [序列化 torch.nn.Modules 并在 C++ 中加载它们](#id6) [¶](#serializing-torch-nn-modules-and-loading-them-in-c "永久链接到此标题")


 另请参阅：[教程：用 C++ 加载 TorchScript 模型](https://pytorch.org/tutorials/advanced/cpp_export.html)


 ScriptModules 可以序列化为 TorchScript 程序并使用 [`torch.jit.load()`](../generated/torch.jit.load.html#torch.jit.load "torch.jit.load") 加载。序列化对所有模块的方法、子模块、参数和属性进行编码，并且允许序列化程序在 C++ 中加载(即无需 Python)。


 [`torch.jit.save()`](../generated/torch.jit.save.html#torch.jit.save "torch.jit.save") 和 [`torch.save()` 之间的区别](../generated/torch.save.html#torch.save "torch.save") 可能不会立即清晰。 [`torch.save()`](../generated/torch.save.html#torch.save "torch.save") 使用 pickle 保存 Python 对象。这对于原型设计、研究和培训特别有用。另一方面， [`torch.jit.save()`](../generated/torch.jit.save.html#torch.jit.save "torch.jit.save") 将 ScriptModule 序列化为可以以 Python 或 C++ 加载。这在保存和加载 C++ 模块或使用 C++ 运行在 Python 中训练的模块时非常有用，这是部署 PyTorch 模型时的常见做法。


 要在 Python 中编写脚本、序列化并加载模块：


```
>>> scripted_module = torch.jit.script(MyModule())
>>> torch.jit.save(scripted_module, 'mymodule.pt')
>>> torch.jit.load('mymodule.pt')
RecursiveScriptModule( original_name=MyModule
 (l0): RecursiveScriptModule(original_name=Linear)
 (l1): RecursiveScriptModule(original_name=Linear) )

```


 跟踪的模块也可以使用 [`torch.jit.save()`](../generated/torch.jit.save.html#torch.jit.save "torch.jit.save") 保存，但需要注意的是跟踪的代码路径被序列化。以下示例演示了这一点：


```
# A module with control flow
>>> class ControlFlowModule(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(4, 2)
        self.l1 = torch.nn.Linear(2, 1)

      def forward(self, input):
        if input.dim() > 1:
            return torch.tensor(0)

        out0 = self.l0(input)
        out0_relu = torch.nn.functional.relu(out0)
        return self.l1(out0_relu)

>>> traced_module = torch.jit.trace(ControlFlowModule(), torch.randn(4))
>>> torch.jit.save(traced_module, 'controlflowmodule_traced.pt')
>>> loaded = torch.jit.load('controlflowmodule_traced.pt')
>>> loaded(torch.randn(2, 4)))
tensor([[-0.1571], [-0.3793]], grad_fn=<AddBackward0>)

>>> scripted_module = torch.jit.script(ControlFlowModule(), torch.randn(4))
>>> torch.jit.save(scripted_module, 'controlflowmodule_scripted.pt')
>>> loaded = torch.jit.load('controlflowmodule_scripted.pt')
>> loaded(torch.randn(2, 4))
tensor(0)

```


 上述模块有一个 if 语句，该语句不是由跟踪输入触发的，因此不是跟踪模块的一部分，也不会与其一起序列化。但是，脚本化模块包含 if 语句并与其一起序列化。请参阅 [ TorchScript 文档](https://pytorch.org/docs/stable/jit.html) 了解有关脚本和跟踪的更多信息。


 最后，在 C++ 中加载模块：


```
>>> torch::jit::script::Module module;
>>> module = torch::jit::load('controlflowmodule_scripted.pt');

```


 有关如何在 C++ 中使用 PyTorch 模块的详细信息，请参阅 [PyTorch C++ API 文档](https://pytorch.org/cppdocs/)。


## [跨 PyTorch 版本保存和加载 ScriptModules](#id7) [¶](# saving-and-loading-scriptmodules-across-pytorch-versions "永久链接到此标题")


 PyTorch 团队建议使用相同版本的 PyTorch 保存和加载模块。旧版本的 PyTorch 可能不支持新模块，而新版本可能已删除或修改旧行为。这些更改在 PyTorch 的 [发行说明](https://github.com/pytorch/pytorch/releases) 中有明确描述，依赖于已更改功能的模块可能需要更新才能继续正常工作。在有限的情况下(如下所述)，PyTorch 将保留序列化脚本模块的历史行为，因此它们不需要更新。


### [torch.div 执行整数除法](#id8) [¶](#torch-div-performing-integer-division "此标题的永久链接")


 在 PyTorch 1.5 及更早版本中，当给定两个整数输入时，[`torch.div()`](../generated/torch.div.html#torch.div "torch.div") 将执行楼层除法：


```
# PyTorch 1.5 (and earlier)
>>> a = torch.tensor(5)
>>> b = torch.tensor(3)
>>> a / b
tensor(1)

```


 然而，在 PyTorch 1.7 中， [`torch.div()`](../generated/torch.div.html#torch.div "torch.div") 将始终对其输入执行真正的除法，就像 Python 中的除法一样3：


```
# PyTorch 1.7
>>> a = torch.tensor(5)
>>> b = torch.tensor(3)
>>> a / b
tensor(1.6667)

```


 [`torch.div()`](../generated/torch.div.html#torch.div "torch.div") 的行为保留在序列化的 ScriptModule 中。也就是说，使用 1.6 之前的 PyTorch 版本序列化的 ScriptModule将继续看到 [`torch.div()`](../generated/torch.div.html#torch.div "torch.div") 在给定两个整数输入时执行楼层划分，即使加载了较新版本的 PyTorch 也是如此。然而，使用 [`torch.div()`](../generated/torch.div.html#torch.div "torch.div") 并在 PyTorch 1.6 及更高版本上序列化的 ScriptModule 无法在早期版本的 PyTorch 中加载，因为那些早期版本不理解新行为。


### [torch.full 总是推断 float dtype](#id9) [¶](#torch-full-always-inferring-a-float-dtype "永久链接到此标题")


 在 PyTorch 1.5 及更早版本中，[`torch.full()`](../generated/torch.full.html#torch.full "torch.full") 始终返回浮点张量，无论给出的填充值如何：


```
# PyTorch 1.5 and earlier
>>> torch.full((3,), 1)  # Note the integer fill value...
tensor([1., 1., 1.])     # ...but float tensor!

```


 然而，在 PyTorch 1.7 中，[`torch.full()`](../generated/torch.full.html#torch.full "torch.full") 将从填充值推断返回的张量的 dtype：


```
# PyTorch 1.7
>>> torch.full((3,), 1)
tensor([1, 1, 1])

>>> torch.full((3,), True)
tensor([True, True, True])

>>> torch.full((3,), 1.)
tensor([1., 1., 1.])

>>> torch.full((3,), 1 + 1j)
tensor([1.+1.j, 1.+1.j, 1.+1.j])

```


 [`torch.full()`](../generated/torch.full.html#torch.full "torch.full") 的行为保留在序列化的 ScriptModule 中。也就是说，使用 1.6 之前的 PyTorch 版本序列化的 ScriptModule 将继续默认情况下 seeto​​rch.full 返回浮点张量，即使给定 bool 或整数填充值也是如此。使用 [`torch.full()`](../generated/torch.full.html#torch.full "torch.full") 并在 PyTorch 1.6 及更高版本上序列化的 ScriptModule 无法在早期版本的 PyTorch 中加载，但是，因为那些早期版本不理解新行为。


## [实用功能](#id10) [¶](#utility-functions "此标题的永久链接")


 以下实用函数与序列化相关：

 > `torch.serialization.register_package(priority, tagger, deserializer)` [[source]](../_modules/torch/serialization.html#register_package)[¶](#torch.serialization.register_package "此定义的永久链接")


 注册可调用对象，用于标记和反序列化具有关联优先级的存储对象。标记在保存时将设备与存储对象相关联，而反序列化则在加载时将存储对象移动到适当的设备。 `tagger` 和 `deserializer` 按照它们的 `priority` 给定的顺序运行，直到标记器/反序列化器返回一个不是 None 的值。


 要覆盖全局注册表中设备的反序列化行为，可以使用比现有标记器更高的优先级注册标记器。


 此函数还可用于为新设备注册标记器和解串器。


 参数 

* **priority** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.12)")) – 指示与标记器和解串器关联的优先级，其中较低的值表示较高的优先级。
* **tagger** ([*Callable*](https://docs.python.org/3/library/typing.html#typing.Callable "(in Python v3.12)")[[[*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(在 Python v3.12 中)")[Storage, [*TypedStorage*](../storage.html#torch.TypedStorage "torch.storage.TypedStorage"), [*UntypedStorage*](../storage.html#torch.UntypedStorage "torch.storage.UntypedStorage")]],[*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(在 Python v3 中.12)")[[*str*](https://docs.python.org/3/library/stdtypes.html#str "(在 Python v3.12 中)")]]) – 可调用，接受存储对象并将其标记设备作为字符串或 None 返回。
* **deserializer** ([*Callable*](https://docs.python.org/3/library/typing.html#typing.Callable "(在 Python v3.12 中)")[[[*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(在 Python v3 中.12)")[Storage, [*TypedStorage*](../storage.html#torch.TypedStorage "torch.storage.TypedStorage"), [*UntypedStorage*](../storage.html#torch.UntypedStorage "torch.storage.UntypedStorage")], [*str*](https://docs.python.org/3/library/stdtypes.html#str "(Python 中) v3.12)")],[*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(在 Python v3.12 中)")[[*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(Python v3.12)")[Storage, [*TypedStorage*](../storage.html#torch.TypedStorage "torch.storage.TypedStorage"), [*UntypedStorage*](../storage.html#torch.UntypedStorage "torch.storage.UntypedStorage")]]]) – 可调用，接受存储对象和设备字符串并返回适当设备上的存储对象 或 None。


 Return: **None**


 例子


```
>>> def ipu_tag(obj):
>>>     if obj.device.type == 'ipu':
>>>         return 'ipu'
>>> def ipu_deserialize(obj, location):
>>>     if location.startswith('ipu'):
>>>         ipu = getattr(torch, "ipu", None)
>>>         assert ipu is not None, "IPU device module is not loaded"
>>>         assert torch.ipu.is_available(), "ipu is not available"
>>>         return obj.ipu(location)
>>> torch.serialization.register_package(11, ipu_tag, ipu_deserialize)

```


 > `torch.serialization.get_default_load_endianness()` [[source]](../_modules/torch/serialization.html#get_default_load_endianness) [¶](#torch.serialization.get_default_load_endianness "此定义的永久链接")


 获取加载文件的后备字节顺序


 如果保存的检查点中不存在字节顺序标记，则该字节顺序将用作后备。 默认情况下，它是“本机”字节顺序。


Returns: **Optional[LoadEndianness]**

Return type: **default_load_endian**


 > `torch.serialization.set_default_load_endianness(endianness)` [[source]](../_modules/torch/serialization.html#set_default_load_endianness)[¶](#torch.serialization.set_default_load_endianness "此定义的永久链接")


 设置加载文件的后备字节顺序


 如果保存的检查点中不存在字节顺序标记，则此字节顺序将用作后备。默认情况下，它是“native”字节顺序。


 Parameters: **endianness** – 新的后备字节顺序