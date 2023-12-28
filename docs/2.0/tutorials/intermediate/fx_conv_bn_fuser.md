# (测试版)在FX中构建一个卷积/BATCH NORM 融合器



> 译者：[方小生](https://github.com/fangxiaoshen)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/fx_conv_bn_fuser>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html>



**作者**: [Horace He](https://github.com/chillee)

在本教程中，我们将使用 FX(一个用于 PyTorch 的可组合函数转换的工具包)来执行以下操作： 

1. 查找数据依赖项中的conv/batch norm。

2. 对于 1) 中找到的模式，将批量范数统计数据折叠到卷积权重中。

 请注意，此优化仅适用于推理模式下的模型(即 mode.eval())。我们将构建此处存在的融合器：https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/fx/experimental/fuser.py。

首先，让我们进行一些导入(稍后我们将在代码中使用所有这些)。

```python
from typing import Type, Dict, Any, Tuple, Iterable
import copy
import torch.fx as fx
import torch
import torch.nn as nn
```

在本教程中，我们将创建一个由卷积和batch norms组成的模型。请注意，该模型有一些棘手的组件 - 一些conv/batch norm patterns隐藏在 Sequentials 中，并且其中一个 BatchNorms 包装在另一个模块中。

```python
class WrappedBatchNorm(nn.Module): # 定义封装BatchNorm2d的自定义模块

    def __init__(self):
        super().__init__()
        self.mod = nn.BatchNorm2d(1) # 创建BatchNorm2d层

    def forward(self, x):
        return self.mod(x) # 前向传播直接调用BatchNorm2d层


class M(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 1, 1) # 卷积层1
        
        self.bn1 = nn.BatchNorm2d(1) # 批标准化层1
        
        self.conv2 = nn.Conv2d(1, 1, 1) # 卷积层2
        
        self.nested = nn.Sequential( # 序列模块
            nn.BatchNorm2d(1), # 批标准化层2
            nn.Conv2d(1, 1, 1),
        )
        
        self.wrapped = WrappedBatchNorm() # 封装批标准化层的自定义模块

    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x) # 应用批标准化层1
        
        x = self.conv2(x)
        
        x = self.nested(x) # 应用序列模块,包含批标准化层2
        
        x = self.wrapped(x) # 应用封装BatchNorm的自定义模块
        
        return x

model = M()

model.eval()
```

## 将卷积与Batch Norm融合1

PyTorch中尝试自动融合convolution and batch norm的主要挑战之一是PyTorch无法提供访问计算图的简单方法。FX通过象征性地跟踪调用的实际操作来解决这个问题，这样我们就可以通过前向调用、嵌套在Sequential模块中或封装在用户定义的模块中来跟踪计算。

```python
traced_model = torch.fx.symbolic_trace(model)
print(traced_model.graph)
```

这为我们的模型提供了一个图形表示。请注意，隐藏在序列模块中的模块以及封装的模块都已内联到图中。这是默认的抽象级别，但可以由pass-writer进行配置。更多信息可在外汇概述中找到https://pytorch.org/docs/master/fx.html#module-torch.fx

## 卷积与Batch Norm融合2

与其他一些融合不同，融合convolution and batch norm不需要任何新的算子。相反，由于推理过程中的批处理范数由逐点加法和乘法组成，因此这些运算可以“baked”到前面的卷积的权重中。这使我们能够完全从我们的模型中删除batch norm！阅读https://nenadmarkus.com/p/fusing-batchnorm-and-conv/了解更多详细信息。此处代码https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/nn/utils/fusion.py。

```python
def fuse_conv_bn_eval(conv, bn):
    """
    给定一个卷积模块A和批标准化模块B,返回一个卷积模块C,
    使得C(x) == B(A(x)),仅在推理模式下成立。
    """
    assert(not (conv.training or bn.training)), "融合仅用于推理!"

    fused_conv = copy.deepcopy(conv) 

    # 融合卷积和BN的权重及偏置
    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
        fused_conv.weight, fused_conv.bias, 
        bn.running_mean, bn.running_var, bn.eps, 
        bn.weight, bn.bias)
    
    return fused_conv


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    
    # 处理None情况
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)  
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    
    # 计算融合后的权重和偏置    
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    conv_w = conv_w * (bn_w / bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1)) 
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)
```

## FX Fusion 实现过程

现在我们有了计算图以及融合convolution 和 batch norm的方法，剩下的就是迭代FX图并应用所需的融合。

```python
def _parent_name(target: str) -> Tuple[str, str]:
    """
    将qualname拆分成父路径和最后一个属性名
    例如:'foo.bar.baz' -> ('foo.bar', 'baz')
    """
    *parent, name = target.rsplit('.', 1) 
    return parent[0] if parent else '', name

# 替换图中某个节点对应的模块
def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module) # 替换模块

def fuse(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model) # 深拷贝模型

    # 获得图表示
    fx_model: fx.GraphModule = fx.symbolic_trace(model) 

    modules = dict(fx_model.named_modules()) # 获得所有模块

    # 遍历图中的每个节点
    for node in fx_model.graph.nodes:
        if node.op != 'call_module': 
            continue # 跳过非模块调用节点
        
        # 检查节点是否为卷积->批标准化
        if type(modules[node.target]) is nn.BatchNorm2d and type(modules[node.args[0].target]) is nn.Conv2d:  
            if len(node.args[0].users) > 1:
                continue  # 卷积层输出被多个节点使用时跳过
            
            # 获得卷积层和BN层
            conv = modules[node.args[0].target] 
            bn = modules[node.target]
            
            # 融合参数
            fused_conv = fuse_conv_bn_eval(conv, bn)  
            
            # 替换图中的卷积层
            replace_node_module(node.args[0], modules, fused_conv)
            
            # 替换BN层的使用为卷积层
            node.replace_all_uses_with(node.args[0])  
            
            # 删除BN节点
            fx_model.graph.erase_node(node)

    fx_model.graph.lint() # 检查图的正确性
    fx_model.recompile()  # 重新编译
    
    return fx_model
```

**注意：**

出于演示目的，我们在这里进行了一些简化，例如仅匹配2D卷积。查看https://github.com/pytorch/pytorch/blob/master/torch/fx/experimental/fuser.py以获得更详细的过程。

## 测试我们的Fusion Pass

我们现在可以在我们最初的玩具模型上运行这个融合过程，并验证我们的结果是否相同。此外，我们可以打印出融合模型的代码，并验证是否没有更多的batch norms。

```python
fused_model = fuse(model)
print(fused_model.code)
inp = torch.randn(5, 1, 1, 1)
torch.testing.assert_allclose(fused_model(inp), model(inp))
```

## 在ResNet18上对我们的融合进行基准测试

我们可以在像ResNet18这样的更大模型上测试我们的融合过程，看看这个过程在多大程度上提高了推理性能。

```python
import torchvision.models as models
import time

rn18 = models.resnet18()
rn18.eval()

inp = torch.randn(10, 3, 224, 224)
output = rn18(inp)

def benchmark(model, iters=20):
    for _ in range(10):
        model(inp)
    begin = time.time()
    for _ in range(iters):
        model(inp)
    return str(time.time()-begin)

fused_rn18 = fuse(rn18)
print("Unfused time: ", benchmark(rn18))
print("Fused time: ", benchmark(fused_rn18))
```

正如我们之前看到的，我们的FX转换的输出是(“torchscriptable”)PyTorch代码，我们可以轻松地对输出进行**jit.script**，以尝试进一步提高我们的性能。通过这种方式，我们的FX模型转换与TorchScript完美结合。

```python
jit_rn18 = torch.jit.script(fused_rn18)
print("jit time: ", benchmark(jit_rn18))


############
# Conclusion
# ----------
# As we can see, using FX we can easily write static graph transformations on
# PyTorch code.
#
# Since FX is still in beta, we would be happy to hear any
# feedback you have about using it. Please feel free to use the
# PyTorch Forums (https://discuss.pytorch.org/) and the issue tracker
# (https://github.com/pytorch/pytorch/issues) to provide any feedback
# you might have.
```

