# torch.utils.dlpack [¶](#torch-utils-dlpack "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/dlpack>
>
> 原始地址：<https://pytorch.org/docs/stable/dlpack.html>


---

> `torch.utils.dlpack.from_dlpack(ext_tensor)` → [Tensor](tensors.html#torch.Tensor "torch.Tensor") [[source]](_modules/torch/utils/dlpack.html#from_dlpack)[¶](#torch.utils.dlpack.from_dlpack"此定义的永久链接")


 将外部库中的tensor转换为`torch.Tensor`。


 返回的 PyTorch tensor将与输入tensor(可能来自另一个库)共享内存。请注意，就地操作也会影响输入tensor的数据。这可能会导致意外的问题(例如，其他库可能具有只读标志或不可变的数据结构)，因此用户只有在确定这样做没问题时才应该这样做。


 Parameters


* **ext_tensor** (具有 `__dlpack__` 属性的对象，或 DLPack 胶囊) – 要转换的tensor或 DLPack 胶囊。


 如果 `ext_tensor` 是一个tensor(或 ndarray)对象，它必须支持 `__dlpack__` 协议(即有一个 `ext_tensor.__dlpack__` 方法) 。否则，`ext_tensor`可能是一个 DLPack 胶囊，它是一个不透明的`PyCapsule`实例，通常由`to_dlpack`函数或方法生成。


 Return type


[*tensor*](tensors.html#torch.Tensor "torch.Tensor")


 例子：


```
>>> import torch.utils.dlpack
>>> t = torch.arange(4)

# Convert a tensor directly (supported in PyTorch >= 1.10)
>>> t2 = torch.from_dlpack(t)
>>> t2[:2] = -1  # show that memory is shared
>>> t2
tensor([-1, -1, 2, 3])
>>> t
tensor([-1, -1, 2, 3])

# The old-style DLPack usage, with an intermediate capsule object
>>> capsule = torch.utils.dlpack.to_dlpack(t)
>>> capsule
<capsule object "dltensor" at ...>
>>> t3 = torch.from_dlpack(capsule)
>>> t3
tensor([-1, -1, 2, 3])
>>> t3[0] = -9  # now we're sharing memory between 3 tensors
>>> t3
tensor([-9, -1, 2, 3])
>>> t2
tensor([-9, -1, 2, 3])
>>> t
tensor([-9, -1, 2, 3])

```


> `torch.utils.dlpack.to_dlpack(tensor) ` → PyCapsule [¶](#torch.utils.dlpack.to_dlpack"此定义的永久链接")


 返回表示tensor的不透明对象(`DLPack 胶囊`)。




!!! note "笔记"

    `to_dlpack` 是一个传统的 DLPack 接口。 它返回的胶囊除了用作 `from_dlpack` 的输入之外不能用于 Python 中的任何其他用途。 DLPack 更惯用的用法是直接在张量对象上调用 `from_dlpack` - 当该对象具有 `__dlpack__` 方法时，这才起作用，PyTorch 和大多数其他库现在确实拥有该方法。


!!! warning "警告"

    每个使用`to_dlpack`生成的胶囊仅调用一次`from_dlpack`。多次消耗胶囊时的行为是未定义的。


 Parameters


* **tensor** – 要导出的tensor


 DLPack 胶囊共享tensor的内存。
