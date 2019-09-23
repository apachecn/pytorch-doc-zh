# 常见问题

## 我的模型报告“CUDA运行时错误（2）：内存不足”

由于错误信息提示，您已经在GPU上运行的内存不足。因为我们经常处理大量的数据PyTorch的，小错可迅速导致你的程序使用了所有的GPU
;幸运的是，在这些情况下修复往往比较简单。这里有一些共同的东西进行检查：

**不要堆积在你的训练循环的历史。**
缺省情况下，涉及的变量需要梯度计算将保持历史。这意味着你应该避免在计算中，将活过你的训练循环，例如，跟踪统计信息时，使用这样的变量。相反，你应该分离变量或访问其基础数据。

有时，它可以是非明显，当微变量可能发生。考虑下面的训练循环（从[源删节](https://discuss.pytorch.org/t/high-
memory-usage-while-training/162)）：

    
    
    total_loss = 0
    for i in range(10000):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output)
        loss.backward()
        optimizer.step()
        total_loss += loss
    

在这里，`total_loss`是在你的训练循环积累的历史，因为`损失 `是autograd历史上的一个微变量。可以通过编写 total_loss
+ =浮子（损失）代替解决这个问题。

这个问题的其他情况：HTG0] 1 [HTG1。

**不要守住张量和你不需要的变量。 [HTG1如果您分配一个张量或变量到本地，Python将不会解除分配，直到当地超出范围。您可以使用`德尔 ×
`释放此引用。同样地，如果分配一个张量或变量的一个对象的成员变量，它不会解除分配，直到对象超出范围。你会得到最好的内存使用情况，如果你不抓住你不需要的临时。**

当地人的范围可能比预期的大。例如：

    
    
    for i in range(5):
        intermediate = f(input[i])
        result += g(intermediate)
    output = h(result)
    return output
    

在此，`中间体 `遗体活甚至当`H`正在执行，因为它的范围挤出过去的循环的结束。较早释放它，你应该`德尔 中间体 `当你用它做。

**不要太大序列运行RNNs。**
的存储器通过RNN到backpropagate，所需的量与的RNN输入的长度成线性比例;这样，就会耗尽存储器如果试图养活RNN的序列太长。

造成这一现象的技术术语为[经过时间](https://en.wikipedia.org/wiki/Backpropagation_through_time)反向传播，并有大量关于如何实现截断BPTT，包括[字的语言模型](https://github.com/pytorch/examples/tree/master/word_language_model)例如参考;截断被处理如[这个论坛帖子](https://discuss.pytorch.org/t/help-
clarifying-repackage-hidden-in-word-language-model/226)所述的`重新打包 `功能。

**不要使用太大的线性层。** 的线性层`nn.Linear（M， N） `使用 O  （ n的 M  ） O（nm）的 O  （ n的 M  ）
存储器：即，权重的存储器要求与特征的数量的二次方成比例。这是很容易为[通过您的记忆吹](https://github.com/pytorch/pytorch/issues/958)这种方式（请记住，你将需要重中的至少两倍的大小，因为你还需要存储的梯度。）

## 我的GPU内存不释放正确

PyTorch使用缓存内存分配器，以加快内存分配。其结果是，在`NVIDIA-SMI`通常不反映真实的存储器使用所示的值。参见[ 内存管理
](cuda.html#cuda-memory-management)关于GPU内存管理的更多细节。

如果退出的Python即使您的GPU内存不释放，这很可能是一些Python子进程仍然活着。您可能会发现他们通过`PS  -elf  |  用grep  蟒
`和手动杀死它们与`杀 -9  [PID]`。

## 我的数据加载工返乡相同随机数

您可能使用其他库，以生成数据集中的随机数。例如，当工人的子过程通过`叉 `开始与NumPy的RNG被复制。参见[ `
torch.utils.data.DataLoader`](../data.html#torch.utils.data.DataLoader
"torch.utils.data.DataLoader")的与它的`worker_init_fn工人[关于如何正确设置随机种子文件HTG12 ]
`选项。

## 我经常性的网络不与数据并行工作

有在使用`收拾 序列的微妙 - & GT ;  复发 网络 - & GT ;  解压 序列 `在[ `模块图案 `
](../nn.html#torch.nn.Module "torch.nn.Module")与[ `数据并行 `
](../nn.html#torch.nn.DataParallel "torch.nn.DataParallel")或[ `
data_parallel（） `](../nn.functional.html#torch.nn.parallel.data_parallel
"torch.nn.parallel.data_parallel")。输入到每个`向前（） `每个设备上仅是整个输入的一部分。因为解包运算[ `
torch.nn.utils.rnn.pad_packed_sequence（） `
](../nn.html#torch.nn.utils.rnn.pad_packed_sequence
"torch.nn.utils.rnn.pad_packed_sequence")缺省仅焊盘到它看到最长输入，即，所述最长那个特定的设备，当结果被聚集在一起会发生大小不匹配。因此，可以改为取`
total_length`的[ `pad_packed_sequence参数的优势（） `
](../nn.html#torch.nn.utils.rnn.pad_packed_sequence
"torch.nn.utils.rnn.pad_packed_sequence")，以确保该`向前（） `调用返回相同长度的序列。例如，你可以这样写：

    
    
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    
    class MyModule(nn.Module):
        # ... __init__, other methods, etc.
    
        # padded_input is of shape [B x T x *] (batch_first mode) and contains
        # the sequences sorted by lengths
        #   B is the batch size
        #   T is max sequence length
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
    

另外，加倍小心需要采取当批量尺寸是昏暗`1`（即`batch_first =假
`）与数据并行性。在这种情况下，pack_padded_sequence的第一个参数`padding_input`将形状的`[T  × B  ×
*]`和应沿着昏暗`1`分散，但第二个参数`input_lengths`将形状的`[B]`和应沿着昏暗`0
`散射。将需要额外的代码来处理张量的形状。

[Next ![](../_static/images/chevron-right-
orange.svg)](large_scale_deployments.html "Features for large-scale
deployments") [![](../_static/images/chevron-right-orange.svg)
Previous](extending.html "Extending PyTorch")

* * *

©版权所有2019年，Torch 贡献者。
