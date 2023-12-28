# tensor视图 [¶](#tensor-views "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/tensor_view>
>
> 原始地址：<https://pytorch.org/docs/stable/tensor_view.html>


 PyTorch 允许tensor成为现有tensor的“视图”。视图tensor与其基本tensor共享相同的基础数据。支持“View”可以避免显式数据复制，从而使我们能够进行快速且内存高效的整形、切片和逐元素操作。


 例如，要获取现有tensor“t”的视图，您可以调用“t.view(...)”。


```
>>> t = torch.rand(4, 4)
>>> b = t.view(2, 8)
>>> t.storage().data_ptr() == b.storage().data_ptr()  # `t` and `b` share the same underlying data.
True
# Modifying view tensor changes base tensor as well.
>>> b[0][0] = 3.14
>>> t[0][0]
tensor(3.14)

```


 由于视图与其基本tensor共享基础数据，因此如果您在视图中编辑数据，它也会反映在基本tensor中。


 通常，PyTorch 操作会返回一个新的tensor作为输出，例如[`add()`](generated/torch.Tensor.add.html#torch.Tensor.add "torch.Tensor.add") 。但是在视图操作的情况下，输出是输入tensor的视图，以避免不必要的数据复制创建视图时不会发生数据移动，视图tensor只是改变它解释相同数据的方式。查看连续tensor可能会产生非连续tensor。用户应额外注意，因为连续性可能会对性能产生隐式影响。 [`transpose()`](generated/torch.Tensor.transpose.html#torch.Tensor.transpose "torch.Tensor.transpose") 是一个常见的例子。


```
>>> base = torch.tensor([[0, 1],[2, 3]])
>>> base.is_contiguous()
True
>>> t = base.transpose(0, 1)  # `t` is a view of `base`. No data movement happened here.
# View tensors might be non-contiguous.
>>> t.is_contiguous()
False
# To get a contiguous tensor, call `.contiguous()` to enforce
# copying data when `t` is not contiguous.
>>> c = t.contiguous()

```


 作为参考，以下是 PyTorch 中视图操作的完整列表：



* 基本切片和索引操作，例如`tensor[0, 2:, 1:7:2]` 返回基本 `tensor` 的视图，请参阅下面的注释。
* [`adjoint()`](generated/torch.Tensor.adjoint.html#torch.Tensor.adjoint "torch.Tensor.adjoint")
* [`as_strided()`](generated/torch.Tensor.as_strided.html#torch.Tensor.as_strided "torch.Tensor.as_strided")
* [`detach() `](generated/torch.Tensor.detach.html#torch.Tensor.detach "torch.Tensor.detach")
* [`diagonal()`](generated/torch.Tensor.diagonal.html#torch.Tensor.diagonal "torch.Tensor.diagonal")
* [`expand()`](generated/torch.Tensor.expand.html#torch.Tensor.expand "torch.Tensor.expand")
* [`expand_as()`](generated/torch.Tensor.expand_as.html#torch.Tensor.expand_as“torch.Tensor.expand_as”)
* [`movedim()`](generated/torch.Tensor.movedim.html#torch.Tensor.movedim“火炬.Tensor.movedim")
* [`narrow()`](generated/torch.Tensor.narrow.html#torch.Tensor.narrow "torch.Tensor.narrow")
* [`permute()`](generated/torch.Tensor.permute.html#torch.Tensor.permute "torch.Tensor.permute")
* [`select()`](generated/torch.Tensor.select.html#torch.Tensor.select "torch.Tensor.select ")
* [`squeeze()`](generated/torch.Tensor.squeeze.html#torch.Tensor.squeeze "torch.Tensor.squeeze")
* [`transpose()`](generated/torch.Tensor.transpose.html#torch.Tensor.transpose "torch.Tensor.transpose")
* [`t()`](generated/torch.Tensor.t.html#torch.Tensor.t "torch.Tensor.t")
* [ `T`](tensors.html#torch.Tensor.T "torch.Tensor.T")
* [`H`](tensors.html#torch.Tensor.H "torch.Tensor.H")
* [`mT `](tensors.html#torch.Tensor.mT "torch.Tensor.mT")
* [`mH`](tensors.html#torch.Tensor.mH "torch.Tensor.mH")
* [`real`](generated/torch.Tensor.real.html#torch.Tensor.real "torch.Tensor.real")
* [`imag`](generated/torch.Tensor.imag.html#torch.Tensor.imag"torch.Tensor.imag")
* `view_as_real()`
* [`unflatten()`](generated/torch.Tensor.unflatten.html#torch.Tensor.unflatten "torch.Tensor.unflatten")
* [`unfold ()`](generated/torch.Tensor.unfold.html#torch.Tensor.unfold "torch.Tensor.unfold")
* [`unsqueeze()`](generated/torch.Tensor.unsqueeze.html#torch.Tensor.unsqueeze "torch.Tensor.unsqueeze")
* [`view()`](generated/torch.Tensor.view.html#torch.Tensor.view "torch.Tensor.view")
* [`view_as() `](generated/torch.Tensor.view_as.html#torch.Tensor.view_as "torch.Tensor.view_as")
* [`unbind()`](generated/torch.Tensor.unbind.html#torch.Tensor.unbind "torch.Tensor.unbind")
* [`split()`](generated/torch.Tensor.split.html#torch.Tensor.split "torch.Tensor.split")
* [`hsplit()`](generated/torch.Tensor.hsplit.html#torch.Tensor.hsplit "torch.Tensor.hsplit")
* [`vsplit()`](generated/torch.Tensor.vsplit.html#torch.Tensor.vsplit "torch.Tensor.vsplit")
* [`tensor_split()`](generated/torch.Tensor.tensor_split.html#torch.Tensor.tensor_split "torch.Tensor.tensor_split")
* `split_with_sizes()`
* [ `swapaxes()`](generated/torch.Tensor.swapaxes.html#torch.Tensor.swapaxes "torch.Tensor.swapaxes")
* [`swapdims()`](generated/torch.Tensor.swapdims.html#torch.Tensor.swapdims "torch.Tensor.swapdims")
* [`chunk()`](generated/torch.Tensor.chunk.html#torch.Tensor.chunk "torch.Tensor.chunk")
* [`indices() `](generated/torch.Tensor.indices.html#torch.Tensor.indices "torch.Tensor.indices") (仅限稀疏tensor)
* [`values()`](generated/torch.Tensor.values.html#torch.Tensor.values "torch.Tensor.values") (仅限稀疏tensor)




!!! note "笔记"

    当通过索引访问tensor的内容时，PyTorch 遵循 Numpy 的行为，即基本索引返回视图，而高级索引返回副本。通过基本索引或高级索引进行的分配都是就地的。请参阅 [Numpy 索引文档](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html) 中的更多示例。


 还值得一提的是一些具有特殊行为的操作：



* [`reshape()`](generated/torch.Tensor.reshape.html#torch.Tensor.reshape "torch.Tensor.reshape") , [`reshape_as()`](generated/torch.Tensor.reshape_as.html#torch.Tensor.reshape_as "torch.Tensor.reshape_as") 和 [`flatten()`](generated/torch.Tensor.flatten.html#torch.Tensor.flatten "torch.Tensor.flatten") 可以返回无论是视图还是新tensor，用户代码都不应该依赖于它是否是视图。
* [`contigously()`](generated/torch.Tensor.contigulous.html#torch.Tensor.contigulous "torch.Tensor.contigulous ") 如果输入tensor已经是连续的，则返回**本身**，否则它通过复制数据返回一个新的连续tensor。


 有关 PyTorch 内部实现的更详细的演练，请参阅 [ezyang 关于 PyTorch Internals 的博文](http://blog.ezyang.com/2019/05/pytorch-internals/) 。