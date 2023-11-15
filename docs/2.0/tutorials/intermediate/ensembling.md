


 没有10



 单击
 [此处](#sphx-glr-download-intermediate-ensembling-py)
 下载完整的示例代码








 模型集成
 [¶](#model-ensembling "此标题的固定链接")
====================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/ensembling>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/ensembling.html>




 本教程说明如何使用
 `torch.vmap` 对模型集成进行矢量化
 。





 什么是模型集成？
 [¶](#what-is-model-ensembling "此标题的永久链接")
------------------------------------------------------------------------------------------



 模型集成将多个模型的预测组合在一起。
传统上，这是通过在某些输入上分别运行每个模型
然后组合预测来完成的。但是，如果您’ 正在运行具有相同架构的模型，则可以使用
 `torch.vmap`
 将它们组合在一起
。
 `vmap`
 是一个函数转换跨输入张量的维度映射函数。它的用例之一是消除
for 循环并通过矢量化加速它们。




 让’s 演示如何使用简单 MLP 的集合来执行此操作。





 注意




 本教程需要 PyTorch 2.0.0 或更高版本。







```
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# Here's a simple MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

```




 让’s 生成一批虚拟数据并假装我们’ 正在使用
an MNIST 数据集。因此，虚拟图像为 28 x 28，并且我们有一个
大小为 64 的小批量。此外，假设我们想要组合
来自 10 个不同模型的预测。






```
device = 'cuda'
num_models = 10

data = torch.randn(100, 64, 1, 28, 28, device=device)
targets = torch.randint(10, (6400,), device=device)

models = [SimpleMLP().to(device) for _ in range(num_models)]

```




 我们有几个用于生成预测的选项。也许我们想为每个模型提供不同的随机小批量数据。或者，
也许我们希望通过每个模型运行相同的小批量数据（例如
如果我们正在测试不同模型初始化的效果）。




 选项 1：每个模型使用不同的小批量






```
minibatches = data[:num_models]
predictions_diff_minibatch_loop = model([minibatch) for model, minibatch in zip(models, minibatches)]

```




 选项 2：相同的小批量






```
minibatch = data[0]
predictions2 = model([minibatch) for model in models]

```






 使用
 `vmap`
 对整体进行矢量化
 [¶](#using-vmap-to-vectorize-the-ensemble "永久链接到此标题")
--------------------------------------------------------------------------------------------------------------------



 让’s 使用
 `vmap`
 来加速for 循环。我们必须首先准备模型
以便与
 `vmap`
 一起使用。



首先，让’s 通过堆叠每个
参数将模型的状态组合在一起。例如，
 `model[i].fc1.weight`
 具有形状
 `[784,
 

 128]`
 ；我们将
将 10 个模型中每个模型的
 `.fc1.weight`
 堆叠起来，以产生一个大的
形状权重
 `[10,
 

 784,
 
\ n 128]`
 。




 PyTorch 提供
 `torch.func.stack_module_state`
 便捷函数来执行此操作。






```
from torch.func import stack_module_state

params, buffers = stack_module_state(models)

```




 接下来，我们需要定义一个函数来
 `vmap`
 。该函数应该在给定参数、缓冲区和输入的情况下使用这些参数、缓冲区和输入运行模型。我们’ 将使用
 `torch.func.function_call`
 来帮助解决：






```
from torch.func import functional_call
import copy

# Construct a "stateless" version of one of the models. It is "stateless" in
# the sense that the parameters are meta Tensors and do not have storage.
base_model = copy.deepcopy(models[0])
base_model = base_model.to('meta')

def fmodel(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))

```




 选项 1：为每个模型使用不同的小批量获取预测。




 默认情况下，
 `vmap`
 将所有输入的第一个维度上的函数映射到
传入的函数。使用
 `stack_module_state`
 后，每个
 `params`
 和缓冲区都有一个大小为 ‘num_models’ 的附加维度在
前面，小批量的尺寸为 ‘num_models’。






```
print([p.size(0) for p in params.values()]) # show the leading 'num_models' dimension

assert minibatches.shape == (num_models, 64, 1, 28, 28) # verify minibatch has leading dimension of size 'num_models'

from torch import vmap

predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)

# verify the ``vmap`` predictions match the
assert torch.allclose(predictions1_vmap, torch.stack(predictions_diff_minibatch_loop), atol=1e-3, rtol=1e-5)

```






```
[10, 10, 10, 10, 10, 10]

```




 选项 2：使用相同的小批量数据获取预测。




`vmap`
 有一个
 `in_dims`
 参数，用于指定要映射的维度。
通过使用
 `None`
 ，我们告诉
 `vmap`
 我们想要相同的小批量应用于所有
这 10 个模型。






```
predictions2_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, minibatch)

assert torch.allclose(predictions2_vmap, torch.stack(predictions2), atol=1e-3, rtol=1e-5)

```




 快速说明：
 `vmap`
 可以转换的函数类型存在限制。最好的转换函数是纯函数：
输出仅由输入决定的函数，
没有副作用（例如突变）。
 `vmap`
 无法处理
任意突变Python 数据结构，但它能够处理许多就地
PyTorch 操作。






 性能
 [¶](#performance "此标题的永久链接")
------------------------------------------------------------------------------------



 对性能数据感到好奇吗？这里’ 是数字的样子。






```
from torch.utils.benchmark import Timer
without_vmap = Timer(
    stmt="[model(minibatch) for model, minibatch in zip(models, minibatches)]",
    globals=globals())
with_vmap = Timer(
    stmt="vmap(fmodel)(params, buffers, minibatches)",
    globals=globals())
print(f'Predictions without vmap {without_vmap.timeit(100)}')
print(f'Predictions with vmap {with_vmap.timeit(100)}')

```






```
Predictions without vmap <torch.utils.benchmark.utils.common.Measurement object at 0x7f67c9641270>
[model(minibatch) for model, minibatch in zip(models, minibatches)]
  2.23 ms
  1 measurement, 100 runs , 1 thread
Predictions with vmap <torch.utils.benchmark.utils.common.Measurement object at 0x7f67c96415d0>
vmap(fmodel)(params, buffers, minibatches)
  843.27 us
  1 measurement, 100 runs , 1 thread

```




 使用
 `vmap`
 可以大幅提升’ 的速度！




 一般来说，使用
 `vmap`
 进行矢量化应该比在 for 循环中
 运行函数更快，并且与手动批处理竞争。但也有一些例外，例如我们是否未’ 为特定操作实现
 `vmap`
 规则，或者底层内核’ 未针对较旧的硬件进行优化
 （GPU）。如果您发现任何此类情况，请通过在 GitHub 上提出问题来告知我们。




**脚本的总运行时间:** 
 ( 0 分 0.794 秒)






[`下载
 

 Python
 

 源
 

 代码:
 

 ensembling.py`](../_downloads/626f23350a6d0b457ded1932a69ec7eb/ensembling.py)






[`下载
 

 Jupyter
 

 笔记本:
 

 ensembling.ipynb`](../_downloads/1342193c7104875f1847417466d1417c/ensembling.ipynb)






[Sphinx-Gallery 生成的图库](https://sphinx-gallery.github.io)









