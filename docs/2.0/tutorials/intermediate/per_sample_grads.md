
 每个样本梯度
 [¶](#per-sample-gradients "永久链接到此标题")
===========================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/per_sample_grads>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/per_sample_grads.html>





 这是什么？
 [¶](#what-is-it "此标题的永久链接")
-----------------------------------------------------------------------


每样本梯度计算正在计算一批数据中每个样本的梯度。它在差分隐私、元学习和优化研究中是一个有用的量。





 注意




 本教程需要 PyTorch 2.0.0 或更高版本。







```
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# Here's a simple CNN and loss function:

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        output = x
        return output

def loss_fn(predictions, targets):
    return F.nll_loss(predictions, targets)

```




 让’s 生成一批虚拟数据，并假设我们’ 正在使用 MNIST 数据集。
虚拟图像为 28 x 28，我们使用大小为 64 的小批量。 






```
device = 'cuda'

num_models = 10
batch_size = 64
data = torch.randn(batch_size, 1, 28, 28, device=device)

targets = torch.randint(10, (64,), device=device)

```




 在常规模型训练中，我们会通过模型转发小批量，然后调用.backward() 来计算梯度。这将生成整个小批量的
‘average’ 梯度：






```
model = SimpleCNN().to(device=device)
predictions = model(data)  # move the entire mini-batch through the model

loss = loss_fn(predictions, targets)
loss.backward()  # back propagate the 'average' gradient of this mini-batch

```




 与上述方法相反，每样本梯度计算
 等价于：



* 对于数据的每个单独样本，执行前向和后向
传递以获得单独的（每个样本）梯度。





```
def compute_grad(sample, target):
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = loss_fn(prediction, target)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(data, targets):
 """ manually process each sample with per sample gradient """
    sample_grads = compute_grad([data[i], targets[i]) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads

per_sample_grads = compute_sample_grads(data, targets)

```




`sample_grads[0]`
 是 model.conv1.weight 的每个样本梯度。
 `model.conv1.weight.shape`
 是
 `[32,\ n 

 1,
 

 3,
 

 3]`
 ;请注意批次中每个样本
有一个梯度，总共 64 个。






```
print(per_sample_grads[0].shape)

```






```
torch.Size([64, 32, 1, 3, 3])

```






 每个样本梯度，
 *有效的方法* 
 ，使用函数变换
 [¶](#per-sample-grads-the-efficient-way-using-function-transforms "Permalink到此标题")
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



 我们可以通过使用函数变换来有效地计算每个样本的梯度。




 `torch.func`
 函数转换 API 对函数进行转换。
我们的策略是定义一个计算损失的函数，然后应用
转换来构造一个计算每个样本梯度的函数。




 我们’ 将使用
 `torch.func.function_call`
 函数将
 `nn.Module`
 视为函数。




 首先，让’s 将状态从
 `模型`
 提取到两个字典、
参数和缓冲区中。我们’将分离它们，因为我们’不会使用
常规的 PyTorch autograd（例如 Tensor.backward()、torch.autograd.grad）。






```
from torch.func import functional_call, vmap, grad

params = {k: v.detach() for k, v in model.named_parameters()}
buffers = {k: v.detach() for k, v in model.named_buffers()}

```




 接下来，让’s 定义一个函数来计算给定单个输入而不是一批输入的模型的损失。此函数接受参数、输入和目标非常重要，因为我们将
对它们进行转换。




 注意 - 由于模型最初是为了处理批次而编写的，因此我们’ll
使用
 `torch.unsqueeze`
 添加批次维度。






```
def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = functional_call(model, (params, buffers), (batch,))
    loss = loss_fn(predictions, targets)
    return loss

```




 现在，让’s 使用
 `grad`
 变换创建一个新函数，用于计算
相对于
 `compute_loss` 第一个参数的梯度
 （即
 `params`
 ）。






```
ft_compute_grad = grad(compute_loss)

```




 `ft_compute_grad`
 函数计算单个
（样本，目标）对的梯度。我们可以使用
 `vmap`
 让它计算整批样本和目标的梯度。注意
 `in_dims=(None,
 

 None,
 

 0,
 

 0)`
 因为我们希望映射
 `ft\在数据和目标的第 0 维上\_compute_grad`
，并为每个维度使用相同的
 `params`
 和
缓冲区。






```
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

```




 最后，让’s 使用我们的变换函数来计算每个样本的梯度：






```
ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)

```




 我们可以仔细检查使用
 `grad`
 和
 `vmap`
 的结果是否与
单独手工处理每个结果的结果匹配：






```
for per_sample_grad, ft_per_sample_grad in zip(per_sample_grads, ft_per_sample_grads.values()):
    assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=3e-3, rtol=1e-5)

```




 快速说明：
 `vmap`
 可以转换的函数类型存在限制。最好的转换函数是
纯函数：输出仅由输入决定的函数，
并且没有副作用（例如突变）。
 `vmap`
 无法处理
突变任意 Python 数据结构，但它能够处理许多
-place PyTorch 操作。






 性能比较
 [¶](#performance-comparison "永久链接到此标题")
--------------------------------------------------------------------------------------------------



 想了解
 `vmap`
 的性能比较如何？




 目前，最佳结果是在较新的 GPU’ 上获得的，例如 A100
(Ampere)，我们’ 在本示例中看到了高达 25 倍的加速，但这里是
一些结果我们的构建机器：






```
def get_perf(first, first_descriptor, second, second_descriptor):
 """takes torch.benchmark objects and compares delta of second vs first."""
    second_res = second.times[0]
    first_res = first.times[0]

    gain = (first_res-second_res)/first_res
    if gain < 0: gain *=-1
    final_gain = gain*100

    print(f"Performance delta: {final_gain:.4f} percent improvement with {first_descriptor} ")

from torch.utils.benchmark import Timer

without_vmap = Timer(stmt="compute_sample_grads(data, targets)", globals=globals())
with_vmap = Timer(stmt="ft_compute_sample_grad(params, buffers, data, targets)",globals=globals())
no_vmap_timing = without_vmap.timeit(100)
with_vmap_timing = with_vmap.timeit(100)

print(f'Per-sample-grads without vmap {no_vmap_timing}')
print(f'Per-sample-grads with vmap {with_vmap_timing}')

get_perf(with_vmap_timing, "vmap", no_vmap_timing, "no vmap")

```






```
Per-sample-grads without vmap <torch.utils.benchmark.utils.common.Measurement object at 0x7fafdb23dae0>
compute_sample_grads(data, targets)
  88.86 ms
  1 measurement, 100 runs , 1 thread
Per-sample-grads with vmap <torch.utils.benchmark.utils.common.Measurement object at 0x7fafdb237d00>
ft_compute_sample_grad(params, buffers, data, targets)
  8.64 ms
  1 measurement, 100 runs , 1 thread
Performance delta: 928.0958 percent improvement with vmap

```




 还有其他优化的解决方案（如 
 <https://github.com/pytorch/opacus>
 ）
 在 PyTorch 中计算每个样本的梯度，其性能也比朴素方法更好。但’ 很酷，组合
 `vmap`
 和
 `grad`
 给我们带来
 很好的加速。




 一般来说，使用
 `vmap`
 进行矢量化应该比在 for 循环中
 运行函数更快，并且与手动批处理竞争。但也有一些例外，例如我们是否未’ 为特定操作实现
 `vmap`
 规则，或者底层内核’ 未针对较旧的硬件进行优化
 （GPU）。如果您发现任何此类情况，请通过在 GitHub 上
提出问题来告知我们。




**脚本的总运行时间:** 
 ( 0 分 10.457 秒)
