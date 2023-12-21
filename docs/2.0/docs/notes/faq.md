# 常见问题 [¶](#frequently-asked-questions "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/faq>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/faq.html>


## 我的模型报告“cuda 运行时错误(2)：内存不足” [¶](#my-model-reports-cuda-runtime-error-2-out-of-memory "永久链接到此标题")


 正如错误消息所示，您的 GPU 内存不足。由于我们经常在 PyTorch 中处理大量数据，小错误可能会迅速导致您的程序耗尽所有 GPU；幸运的是，这些情况下的修复通常很简单。以下是一些需要检查的常见事项：


**不要在训练循环中累积历史记录。** 默认情况下，涉及需要梯度的变量的计算将保留历史记录。这意味着您应该避免在计算中使用此类变量，这些变量将超出您的训练循环，例如在跟踪统计数据时。相反，您应该分离变量或访问其基础数据。


 有时，当可微变量出现时，它可能是不明显的。考虑以下训练循环(摘自[来源](https://discuss.pytorch.org/t/high-memory-usage-while-training/162))：


```
total_loss = 0
for i in range(10000):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output)
    loss.backward()
    optimizer.step()
    total_loss += loss

```


 这里，“total_loss”正在整个训练循环中累积历史记录，因为“loss”是一个具有 autograd 历史记录的可微变量。您可以通过编写 Total_loss += float(loss) 来解决此问题。


 此问题的其他实例：[1](https://discuss.pytorch.org/t/resolved-gpu-out-of-memory-error-with-batch-size-1/3719)。


**不要保留不需要的张量和变量。** 如果将张量或变量分配给局部变量，Python 将不会释放分配，直到局部变量超出范围。您可以使用“del x”释放此引用。类似地，如果将张量或变量分配给对象的成员变量，则在该对象超出范围之前它不会释放。如果您不保留不需要的临时内存，您将获得最佳的内存使用率。


 当地人的范围可能比你想象的要大。例如：


```
for i in range(5):
    intermediate = f(input[i])
    result += g(intermediate)
output = h(result)
return output

```


 在这里，即使在执行“h”时，“intermediate”仍然保持活动状态，因为它的范围超出了循环的末尾。为了更早地释放它，你应该在使用完它后`del middle`。


**避免在太大的序列上运行 RNN。** 通过 RNN 反向传播所需的内存量与 RNN 输入的长度成线性比例；因此，如果您尝试向 RNN 提供太长的序列，您将耗尽内存。


 这种现象的技术术语是[通过时间反向传播](https://en.wikipedia.org/wiki/Backpropagation_through_time)，并且有很多关于如何实现truncatedBPTT的参考，包括在[词语言模型](https://github.com/pytorch/examples/tree/master/word_language_model)示例；截断由“repackage”函数处理，如[本论坛帖子](https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226)中所述。


**不要使用太大的线性层。** 线性层 `nn.Linear(m, n)` 使用 $O(nm)$ 内存：也就是说，权重的内存需求与特征数量呈二次方关系。通过这种方式很容易[耗尽你的记忆](https://github.com/pytorch/pytorch/issues/958)(并且记住你将需要至少两倍大小的权重，因为你还需要存储梯度。)


**考虑检查点。** 您可以使用 [checkpoint](https://pytorch.org/docs/stable/checkpoint.html) 来权衡计算的内存。


## 我的 GPU 内存未正确释放 [¶](#my-gpu-memory-isn-t-freed-properly "永久链接到此标题")


 PyTorch 使用缓存内存分配器来加速内存分配。因此，“nvidia-smi”中显示的值通常不反映真实的内存使用情况。有关 GPU 内存管理的更多详细信息，请参阅[内存管理](cuda.html#cuda-memory-management)。


 如果即使 Python 退出后您的 GPU 内存也没有释放，则很可能某些 Python 子进程仍然存在。你可以通过`ps -elf | 找到它们。 grep python` 并使用 `kill -9 [pid]` 手动杀死它们。


## 我的内存不足异常处理程序无法分配内存 [¶](#my-out-of-memory-exception-handler-can-t-allocate-memory "永久链接到此标题")


 您可能有一些代码尝试从内存不足错误中恢复。


```
try:
    run_model(batch_size)
except RuntimeError: # Out of memory
    for _ in range(batch_size):
        run_model(1)

```


 但你会发现，当你确实耗尽内存时，你的恢复代码也无法分配。这是因为 python 异常对象保存了对引发错误的堆栈帧的引用。这会阻止原始张量对象被释放。解决方案是将 OOM 恢复代码移到“ except”子句之外。


```
oom = False
try:
    run_model(batch_size)
except RuntimeError: # Out of memory
    oom = True

if oom:
    for _ in range(batch_size):
        run_model(1)

```


## 我的数据加载器工作程序返回相同的随机数 [¶](#my-data-loader-workers-return-identical-random-numbers "永久链接到此标题")


 您可能使用其他库在数据集中生成随机数，并且工作子进程通过“fork”启动。请参阅 [`torch.utils.data.DataLoader`](../data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") 的文档，了解如何在工作人员中正确设置随机种子它的`worker_init_fn`选项。


## 我的循环网络无法使用数据并行性 [¶](#my-recurrent-network-doesn-t-work-with-data-parallelism "永久链接到此标题")


 在 [`Module`](../generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") 与 [`DataParallel`](../generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") 或 [`data_parallel()`](../generated/torch.nn.function.torch.nn.parallel.data_parallel.html#torch.nn.parallel.data_parallel“torch.nn.parallel.data_parallel”)。每个设备上每个“forward()”的输入将只是整个输入的一部分。因为解包操作 [`torch.nn.utils.rnn.pad_packed_sequence()`](../generated/torch.nn.utils.rnn.pad_packed_sequence.html#torch.nn.utils.rnn.pad_packed_sequence "torch.nn.utils.rnn.pad_packed_sequence")默认情况下仅填充到它看到的最长输入，即该特定设备上最长的输入，当结果聚集在一起时会发生大小不匹配。因此，您可以利用 [`pad_packed_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html#torch.nn.utils.rnn.pad_packed_sequence "torch.nn.utils.rnn.pad_packed_sequence") 以确保 `forward()` 调用返回相同长度的序列。例如，你可以这样写：


```
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MyModule(nn.Module):
    # ... __init__, other methods, etc.

    # padded_input is of shape [B x T x *](batch_first mode) and contains
    # the sequences sorted by lengths
    # B is the batch size
    # T is max sequence length
    def forward(self, padded_input, input_lengths):
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths,
                                            batch_first=True)
        packed_output, _ = self.my_lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=total_length)
        return output


m = MyModule().cuda()
dp_m = nn.DataParallel(m)

```


 此外，当批量维度为暗淡“1”(即“batch_first=False”)且数据并行性时，需要格外小心。在这种情况下， pack_padd_sequence `padding_input` 的第一个参数的形状为 `[T x B x *]` 并且应该沿着暗淡的 `1` 分散，但第二个参数 `input_lengths`形状为“[B]”，并且应该沿着暗淡的“0”分散。需要额外的代码来操纵张量形状。