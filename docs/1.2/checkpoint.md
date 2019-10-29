# torch.utils.checkpoint
> 译者:  [belonHan](https://github.com/belonHan)

注意

checkpointing的实现方法是在向后传播期间重新运行已被checkpint的前向传播段。 所以会导致像RNG这类(模型)的持久化的状态比实际更超前。默认情况下，checkpoint包含了使用RNG状态的逻辑(例如通过dropout)，与non-checkpointed传递相比,checkpointed具有更确定的输出。RNG状态的存储逻辑可能会导致一定的性能损失。如果不需要确定的输出，设置全局标志(global flag) `torch.utils.checkpoint.preserve_rng_state=False` 忽略RNG状态在checkpoint时的存取。


```py
torch.utils.checkpoint.checkpoint(function, *args)
```

checkpoint模型或模型的一部分

checkpoint通过计算换内存空间来工作。与向后传播中存储整个计算图的所有中间激活不同的是，checkpoint不会保存中间激活部分，而是在反向传递中重新计算它们。它被应用于模型的任何部分。

具体来说，在正向传播中，`function`将以`torch.no_grad()`方式运行 ，即不存储中间激活,但保存输入元组和 `function`的参数。在向后传播中，保存的输入变量以及 `function`会被取回，并且`function`在正向传播中被重新计算.现在跟踪中间激活，然后使用这些激活值来计算梯度。

Warning
警告

Checkpointing 在 [`torch.autograd.grad()`](autograd.html#torch.autograd.grad "torch.autograd.grad")中不起作用, 仅作用于 [`torch.autograd.backward()`](autograd.html#torch.autograd.backward "torch.autograd.backward").

警告

如果function在向后执行和前向执行不同，例如,由于某个全局变量，checkpoint版本将会不同，并且无法被检测到。

参数:

*   **function** - 描述在模型的正向传递或模型的一部分中运行的内容。它也应该知道如何处理作为元组传递的输入。例如，在LSTM中，如果用户通过 ，应正确使用第一个输入作为第二个输入(activation, hidden)functionactivationhidden
*   **args** – 包含输入的元组function

| Returns: | 输出 |
| --- | --- |

```py
torch.utils.checkpoint.checkpoint_sequential(functions, segments, *inputs)
```

用于checkpoint sequential模型的辅助函数

Sequential模型按顺序执行模块/函数。因此，我们可以将这样的模型划分为不同的段(segment)，并对每个段进行checkpoint。除最后一段外的所有段都将以`torch.no_grad()`方式运行，即，不存储中间活动。将保存每个checkpoint段的输入，以便在向后传递中重新运行该段。

checkpointing工作方式: [`checkpoint()`](#torch.utils.checkpoint.checkpoint "torch.utils.checkpoint.checkpoint").

警告

Checkpointing无法作用于[`torch.autograd.grad()`](autograd.html#torch.autograd.grad "torch.autograd.grad"), 只作用于[`torch.autograd.backward()`](autograd.html#torch.autograd.backward "torch.autograd.backward").

参数:

*   **functions** – 按顺序执行的模型， 一个 [`torch.nn.Sequential`](nn.html#torch.nn.Sequential "torch.nn.Sequential")对象,或者一个由modules或functions组成的list。
*   **segments** – 段的数量
*   **inputs** – 输入,Tensor组成的元组



| Returns: | 按顺序返回每个`*inputs`的结果
| --- | --- |


例子

```py
>>> model = nn.Sequential(...)
>>> input_var = checkpoint_sequential(model, chunks, input_var)

```

