
 使用自定义函数融合卷积和批归一化
 [¶](#fusing-volving-and-batch-norm-using-custom-function "永久链接到此标题")
====================================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/custom_function_conv_bn_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/custom_function_conv_bn_tutorial.html>



将相邻的卷积层和批归一化层融合在一起通常是一种推理时间优化，以提高运行时间。它通常是通过完全消除批归一化层并更新前面的卷积的权重和偏差来实现的[0]。但是，此技术
不适用于训练模型。




 在本教程中，我们将展示一种不同的技术来融合两个层
，该技术可以在训练期间应用。此优化的目的不是改进运行时间，
而是减少内存使用。



这种优化背后的想法是看到卷积和批规范（以及许多其他操作）都需要在向前传递过程中保存其输入的副本。对于大
批量大小，这些保存的输入占用了大部分内存，
因此能够避免为每个
卷积批量范数对分配另一个输入张量可以显着减少。



在本教程中，我们通过将卷积和批标准化组合到单个层（作为自定义函数）来避免这种额外的分配。在这个组合层的前向
中，我们按原样执行普通卷积和批归一化，
唯一的区别是我们只保存卷积的输入。
为了获得批归一化的输入，需要向后通过
，我们在向后传递期间再次重新向前计算卷积。




 需要注意的是，这种优化的使用是视情况而定的。
虽然（通过避免保存一个缓冲区）我们总是减少在前向传递结束时
分配的内存，但在某些情况下
 *峰值* 
 分配的内存
实际上可能不会减少。有关详细信息，请参阅最后一部分。




 为简单起见，在本教程中我们硬编码
 
 bias=False
 
 、
 
 stride=1
 
 、
 
 padding=0
 
 、
 
 Conv2D 的 dilation=1
 
 、
and
 
 groups=1
 
 。对于 BatchNorm2D，我们硬编码
 
 eps=1e-3
 
 、
 
动量=0.1
 
 、
 
 affine=False
 
 和
 
 track\ \_running_statistics=False
 
 。另一个小差异
是我们在批量范数的计算中
在平方根之外的分母中添加了epsilon。




 [0]
 <https://nenadmarkus.com/p/fusing-batchnorm-and-conv/>





 卷积的向后公式实现
 [¶](#backward-formula-implementation-for-CNN "永久链接到此标题")
----------------------------------------------------------------------------------------------------------------------------------------



 实现自定义函数需要我们自己实现向后的
。在这种情况下，我们需要 Conv2D
 和 BatchNorm2D 的后向公式。最终我们’d 在统一的
后向函数中将它们链接在一起，但下面我们首先将它们实现为自己的
自定义函数，以便我们可以单独验证它们的正确性






```
import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

def convolution_backward(grad_out, X, weight):
    grad_input = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1)).transpose(0, 1)
    grad_X = F.conv_transpose2d(grad_out, weight)
    return grad_X, grad_input

class Conv2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight):
        ctx.save_for_backward(X, weight)
        return F.conv2d(X, weight)

    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        X, weight = ctx.saved_tensors
        return convolution_backward(grad_out, X, weight)

```




 当使用
 `gradcheck`
 进行测试时，使用双精度非常重要






```
weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
X = torch.rand(10, 3, 7, 7, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(Conv2D.apply, (X, weight))

```






```
True

```






 批标准化的向后公式实现
 [¶](#backward-formula-implementation-for-batch-norm "永久链接到此标题")
-------------------------------------------------------------------------------------------------------------------------------------



 Batch Norm 有两种模式：训练和
 `eval`
 模式。在训练模式下，样本统计数据是输入的函数。在
 `eval`
 模式下，
我们使用保存的运行统计数据，这些统计数据不是输入的函数。
这使得非训练模式’s 向后显着更简单。下面
我们仅实现并测试训练模式案例。






```
def unsqueeze_all(t):
    # Helper function to ``unsqueeze`` all the dimensions that we reduce over
    return t[None, :, None, None]

def batch_norm_backward(grad_out, X, sum, sqrt_var, N, eps):
    # We use the formula: ``out = (X - mean(X)) / (sqrt(var(X)) + eps)``
    # in batch norm 2D forward. To simplify our derivation, we follow the
    # chain rule and compute the gradients as follows before accumulating
    # them all into a final grad_input.
    # 1) ``grad of out wrt var(X)`` * ``grad of var(X) wrt X``
    # 2) ``grad of out wrt mean(X)`` * ``grad of mean(X) wrt X``
    # 3) ``grad of out wrt X in the numerator`` * ``grad of X wrt X``
    # We then rewrite the formulas to use as few extra buffers as possible
    tmp = ((X - unsqueeze_all(sum) / N) * grad_out).sum(dim=(0, 2, 3))
    tmp *= -1
    d_denom = tmp / (sqrt_var + eps)**2  # ``d_denom = -num / denom**2``
    # It is useful to delete tensors when you no longer need them with ``del``
    # For example, we could've done ``del tmp`` here because we won't use it later
    # In this case, it's not a big difference because ``tmp`` only has size of (C,)
    # The important thing is avoid allocating NCHW-sized tensors unnecessarily
    d_var = d_denom / (2 * sqrt_var)  # ``denom = torch.sqrt(var) + eps``
    # Compute ``d_mean_dx`` before allocating the final NCHW-sized grad_input buffer
    d_mean_dx = grad_out / unsqueeze_all(sqrt_var + eps)
    d_mean_dx = unsqueeze_all(-d_mean_dx.sum(dim=(0, 2, 3)) / N)
    # ``d_mean_dx`` has already been reassigned to a C-sized buffer so no need to worry

    # ``(1) unbiased_var(x) = ((X - unsqueeze_all(mean))**2).sum(dim=(0, 2, 3)) / (N - 1)``
    grad_input = X * unsqueeze_all(d_var * N)
    grad_input += unsqueeze_all(-d_var * sum)
    grad_input *= 2 / ((N - 1) * N)
    # (2) mean (see above)
    grad_input += d_mean_dx
    # (3) Add 'grad_out / <factor>' without allocating an extra buffer
    grad_input *= unsqueeze_all(sqrt_var + eps)
    grad_input += grad_out
    grad_input /= unsqueeze_all(sqrt_var + eps)  # ``sqrt_var + eps > 0!``
    return grad_input

class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, eps=1e-3):
        # Don't save ``keepdim`` values for backward
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.save_for_backward(X)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        X, = ctx.saved_tensors
        return batch_norm_backward(grad_out, X, ctx.sum, ctx.sqrt_var, ctx.N, ctx.eps)

```




 使用 `gradcheck` 进行测试






```
a = torch.rand(1, 2, 3, 4, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(BatchNorm.apply, (a,), fast_mode=False)

```






```
True

```






 融合卷积和 BatchNorm
 [¶](#fusing-volving-and-batchnorm "永久链接到此标题")
----------------------------------------------------------------------------------------------------------------------



 现在大部分工作已经完成，我们可以将它们组合在一起。请注意，在 (1) 中我们仅保存单个缓冲区用于向后，但这也意味着我们在 (5) 中重新计算前向卷积。另请参阅 (2)、(3)、(4) 和 (6) 中，’ 与上面示例的代码
完全相同。






```
class FusedConvBN2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, conv_weight, eps=1e-3):
        assert X.ndim == 4  # N, C, H, W
        # (1) Only need to save this single buffer for backward!
        ctx.save_for_backward(X, conv_weight)

        # (2) Exact same Conv2D forward from example above
        X = F.conv2d(X, conv_weight)
        # (3) Exact same BatchNorm2D forward from example above
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        # Try to do as many things in-place as possible
        # Instead of `out = (X - a) / b`, doing `out = X - a; out /= b`
        # avoids allocating one extra NCHW-sized buffer here
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        X, conv_weight, = ctx.saved_tensors
        # (4) Batch norm backward
        # (5) We need to recompute conv
        X_conv_out = F.conv2d(X, conv_weight)
        grad_out = batch_norm_backward(grad_out, X_conv_out, ctx.sum, ctx.sqrt_var,
                                       ctx.N, ctx.eps)
        # (6) Conv2d backward
        grad_X, grad_input = convolution_backward(grad_out, X, conv_weight)
        return grad_X, grad_input, None, None, None, None, None

```




 下一步是将我们的功能变体包装在有状态的
 
 nn.Module







```
import torch.nn as nn
import math

class FusedConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, exp_avg_factor=0.1,
                 eps=1e-3, device=None, dtype=None):
        super(FusedConvBN, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Conv parameters
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.conv_weight = nn.Parameter(torch.empty(*weight_shape, **factory_kwargs))
        # Batch norm parameters
        num_features = out_channels
        self.num_features = num_features
        self.eps = eps
        # Initialize
        self.reset_parameters()

    def forward(self, X):
        return FusedConvBN2DFunction.apply(X, self.conv_weight, self.eps)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))

```




 使用
 `gradcheck`
 验证我们的后向公式的正确性






```
weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
X = torch.rand(2, 3, 4, 4, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(FusedConvBN2DFunction.apply, (X, weight))

```






```
True

```






 测试我们的新层
 [¶](#testing-out-our-new-layer "永久链接到此标题")
-------------------------------------------------------------------------------------------



 使用
 `FusedConvBN`
 训练基本网络
下面的代码是对此处示例进行一些轻微修改后的：
 <https://github.com/pytorch/examples/tree/master/mnist>






```
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# Record memory allocated at the end of the forward pass
memory_allocated = [[],[]]

class Net(nn.Module):
    def __init__(self, fused=True):
        super(Net, self).__init__()
        self.fused = fused
        if fused:
            self.convbn1 = FusedConvBN(1, 32, 3)
            self.convbn2 = FusedConvBN(32, 64, 3)
        else:
            self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(32, affine=False, track_running_stats=False)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if self.fused:
            x = self.convbn1(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
        F.relu_(x)
        if self.fused:
            x = self.convbn2(x)
        else:
            x = self.conv2(x)
            x = self.bn2(x)
        F.relu_(x)
        x = F.max_pool2d(x, 2)
        F.relu_(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.dropout(x)
        F.relu_(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        if fused:
            memory_allocated[0].append(torch.cuda.memory_allocated())
        else:
            memory_allocated[1].append(torch.cuda.memory_allocated())
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # Use inference mode instead of no_grad, for free improved test-time performance
    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
train_kwargs = {'batch_size': 2048}
test_kwargs = {'batch_size': 2048}

if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                          transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

```






```
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/9912422 [00:00<?, ?it/s]
100%|##########| 9912422/9912422 [00:00<00:00, 275602312.46it/s]
Extracting ../data/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/28881 [00:00<?, ?it/s]
100%|##########| 28881/28881 [00:00<00:00, 99618169.26it/s]
Extracting ../data/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/1648877 [00:00<?, ?it/s]
100%|##########| 1648877/1648877 [00:00<00:00, 333151471.49it/s]
Extracting ../data/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/4542 [00:00<?, ?it/s]
100%|##########| 4542/4542 [00:00<00:00, 39118128.89it/s]
Extracting ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw

```






 内存使用情况比较
 [¶](#a-comparison-of-memory-usage "永久链接到此标题")
------------------------------------------------------------------------------------------------



 如果启用了 CUDA，则打印
 
 fused=True
 
 和
 
 fused=False
 
 的内存使用情况，例如在 NVIDIA GeForce RTX 3070、NVIDIA CUDA\xc2 上运行\xae 深度神经网络库 (cuDNN) 8.0.5：融合峰值内存：1.56GB，
未融合峰值内存：2.68GB




 值得注意的是，
 *峰值* 
 此模型的内存使用情况可能会有所不同，具体取决于
所使用的特定 cuDNN 卷积算法。对于较浅的模型，融合模型的峰值内存分配可能会超过未融合模型的峰值内存！这是因为分配给计算
某些 cuDNN 卷积算法的内存可能足够高，“hide” 典型峰值
您期望接近向后传递的开始位置。



出于这个原因，我们还记录并显示了在前向传递结束时分配的内存作为近似值，并证明我们确实为每个融合的“conv-bn”对少分配了一个缓冲区。 






```
from statistics import mean

torch.backends.cudnn.enabled = True

if use_cuda:
    peak_memory_allocated = []

    for fused in (True, False):
        torch.manual_seed(123456)

        model = Net(fused=fused).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

        for epoch in range(1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()
        peak_memory_allocated.append(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    print("cuDNN version:", torch.backends.cudnn.version())
    print()
    print("Peak memory allocated:")
    print(f"fused: {peak_memory_allocated[0]/1024**3:.2f}GB, unfused: {peak_memory_allocated[1]/1024**3:.2f}GB")
    print("Memory allocated at end of forward pass:")
    print(f"fused: {mean(memory_allocated[0])/1024**3:.2f}GB, unfused: {mean(memory_allocated[1])/1024**3:.2f}GB")

```






```
Train Epoch: 0 [0/60000 (0%)]   Loss: 2.348850
Train Epoch: 0 [4096/60000 (7%)]        Loss: 7.906003
Train Epoch: 0 [8192/60000 (13%)]       Loss: 3.856603
Train Epoch: 0 [12288/60000 (20%)]      Loss: 2.177455
Train Epoch: 0 [16384/60000 (27%)]      Loss: 1.875102
Train Epoch: 0 [20480/60000 (33%)]      Loss: 1.706392
Train Epoch: 0 [24576/60000 (40%)]      Loss: 1.608641
Train Epoch: 0 [28672/60000 (47%)]      Loss: 1.696115
Train Epoch: 0 [32768/60000 (53%)]      Loss: 1.412414
Train Epoch: 0 [36864/60000 (60%)]      Loss: 1.309023
Train Epoch: 0 [40960/60000 (67%)]      Loss: 1.213390
Train Epoch: 0 [45056/60000 (73%)]      Loss: 1.128412
Train Epoch: 0 [49152/60000 (80%)]      Loss: 0.853271
Train Epoch: 0 [53248/60000 (87%)]      Loss: 0.860089
Train Epoch: 0 [57344/60000 (93%)]      Loss: 0.757735

Test set: Average loss: 0.3410, Accuracy: 9148/10000 (91%)

Train Epoch: 0 [0/60000 (0%)]   Loss: 2.349130
Train Epoch: 0 [4096/60000 (7%)]        Loss: 7.946081
Train Epoch: 0 [8192/60000 (13%)]       Loss: 3.232975
Train Epoch: 0 [12288/60000 (20%)]      Loss: 2.598296
Train Epoch: 0 [16384/60000 (27%)]      Loss: 1.941216
Train Epoch: 0 [20480/60000 (33%)]      Loss: 2.460819
Train Epoch: 0 [24576/60000 (40%)]      Loss: 2.008726
Train Epoch: 0 [28672/60000 (47%)]      Loss: 1.662222
Train Epoch: 0 [32768/60000 (53%)]      Loss: 1.274567
Train Epoch: 0 [36864/60000 (60%)]      Loss: 1.402421
Train Epoch: 0 [40960/60000 (67%)]      Loss: 1.448435
Train Epoch: 0 [45056/60000 (73%)]      Loss: 1.024507
Train Epoch: 0 [49152/60000 (80%)]      Loss: 0.944104
Train Epoch: 0 [53248/60000 (87%)]      Loss: 0.862664
Train Epoch: 0 [57344/60000 (93%)]      Loss: 0.817179

Test set: Average loss: 0.4785, Accuracy: 8615/10000 (86%)

cuDNN version: 8902

Peak memory allocated:
fused: 4.44GB, unfused: 1.90GB
Memory allocated at end of forward pass:
fused: 0.99GB, unfused: 1.37GB

```




**脚本的总运行时间:** 
 ( 0 分 21.041 秒)
