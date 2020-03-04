

# 常见问题解答  

> 译者：[冯宝宝](https://github.com/PEGASUS1993)

## 我的模型报告“cuda runtime error(2): out of memory”  

正如错误消息所示，您的GPU显存已耗尽。由于经常在PyTorch中处理大量数据，因此小错误会迅速导致程序耗尽所有GPU资源; 幸运的是，这些情况下的修复通常很简单。这里有一些常见点需要检查：  

**不要在训练循环中积累历史记录。** 默认情况下，涉及需要梯度计算的变量将保留历史记录。这意味着您应该避免在计算中使用这些变量，因为这些变量将超出您的训练循环，例如，在跟踪统计数据时。相反，您应该分离变量或访问其基础数据。  

有时，当可微分变量发生时，它可能是不明显的。考虑以下训练循环(从[源代码](https://discuss.pytorch.org/t/high-memory-usage-while-training/162)中删除）：  


```py
total_loss = 0
for i in range(10000):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output)
    loss.backward()
    optimizer.step()
    total_loss += loss

```    

在这里，total_loss在您的训练循环中累积历史记录，因为丢失是具有自动记录历史的可微分变量。 您可以通过编写total_loss + = float(loss）来解决此问题。  

此问题的其他实例：[1](https://discuss.pytorch.org/t/resolved-gpu-out-of-memory-error-with-batch-size-1/3719)。  

**不要抓住你不需要的张量或变量。** 如果将张量或变量分配给本地，则在本地超出范围之前，Python不会解除分配。您可以使用`del x`释放此引用。 同样，如果将张量或向量分配给对象的成员变量，则在对象超出范围之前不会释放。如果您没有保留不需要的临时工具，您将获得最佳的内存使用量。  

本地规模大小可能比您预期的要大。 例如：  

```py
for i in range(5):
    intermediate = f(input[i])
    result += g(intermediate)
output = h(result)
return output

```  

在这里，即使在执行h时，中间变量仍然存在，因为它的范围超出了循环的末尾。要提前释放它，你应该在完成它时使用del。  

**不要在太大的序列上运行RNN。** 通过RNN反向传播所需的存储量与RNN的长度成线性关系; 因此，如果您尝试向RNN提供过长的序列，则会耗尽内存。

这种现象的技术术语是随着时间的推移而反向传播，并且有很多关于如何实现截断BPTT的参考，包括在单词语言模型示例中; 截断由重新打包功能处理，如本论坛帖子中所述。

**不要使用太大的线性图层。** 线性层nn.Linear(m，n）使用O(nm)存储器：也就是说，权重的存储器需求与特征的数量成比例。 以这种方式很容易占用你的存储(并且记住，你将至少需要两倍存储权值的内存量，因为你还需要存储梯度。）


## My GPU memory isn’t freed properly  

PyTorch使用缓存内存分配器来加速内存分配。 因此，`nvidia-smi`中显示的值通常不会反映真实的内存使用情况。 有关GPU内存管理的更多详细信息，请参阅[内存管理](cuda.html#cuda-memory-management) 。

如果在Python退出后你的GPU内存仍旧没有被释放，那么很可能是一些Python子进程仍处于活动状态。你可以通过`ps -elf |grep python`找到它们并用`kill -9 [pid]`手动结束这些进程。  

## My data loader workers return identical random numbers  

您可能正在数据集中使用其他库来生成随机数。 例如，当通过`fork`启动工作程序子进程时，NumPy的RNG会重复。有关如何使用`worker_init_fn`选项在工作程序中正确设置随机种子的文档，请参阅torch.utils.data.DataLoader文档。  

## My recurrent network doesn’t work with data parallelism  

在具有`DataParallel`或`data_parallel()`的模块中使用`pack sequence -> recurrent network -> unpack sequence`模式时有一个非常微妙的地方。每个设备上的`forward()`的输入只会是整个输入的一部分。由于默认情况下，解包操作`torch.nn.utils.rnn.pad_packed_sequence()`仅填充到其所见的最长输入，即该特定设备上的最长输入，所以在将结果收集在一起时会发生尺寸的不匹配。因此，您可以利用`pad_packed_sequence()`的 `total_length`参数来确保`forward()`调用返回相同长度的序列。例如，你可以写：


```py
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MyModule(nn.Module):
    #  ... __init__, 以及其他访求

    # padding_input 的形状是[B x T x *](batch_first 模式），包含按长度排序的序列
    # B 是批量大小
    # T 是最大序列长度
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

另外，在批量的维度为dim 1(即 batch_first = False )时需要注意数据的并行性。在这种情况下，pack_padded_sequence 函数的的第一个参数 padding_input 维度将是 [T x B x *] ，并且应该沿dim 1 (第1轴）分散，但第二个参数 input_lengths 的维度为 [B]，应该沿dim 0 (第0轴）分散。需要额外的代码来操纵张量的维度。
