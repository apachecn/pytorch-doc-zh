# (测试版)用FX构建一个简单的CPU性能分析器

> 译者：[方小生](https://github.com/fangxiaoshen)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/fx_profiling_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html>

**作者**: [James Reed](https://github.com/jamesr66a)

在本教程中，我们将使用FX执行以下操作：

1. 获取PyTorch Python代码，这样我们就可以检查和收集有关代码的结构和执行的统计信息。
2. 构建一个小类，作为一个简单的性能“探查器”，从实际运行中收集有关模型每个部分的运行时统计信息。

在本教程中，我们将使用torchvision ResNet18模型进行演示。

```python
import torch
import torch.fx
import torchvision.models as models

rn18 = models.resnet18()
rn18.eval()
```

**Out:**

```python
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
```

现在我们有了我们的模型，我们想更深入地检查它的性能。也就是说，对于下面的调用，模型的哪些部分花费的时间最长？

```python
input = torch.randn(5, 3, 224, 224)
output = rn18(input)
```

回答这个问题的一种常见方法是浏览程序源代码，添加在程序中不同点收集时间戳的代码，并比较这些时间戳之间的差异，以查看时间戳之间区域的使用时间。

这种技术当然适用于PyTorch代码，但如果我们不必复制模型代码并对其进行编辑，尤其是我们没有编写的代码(比如这个torchvision模型)，那就更好了。相反，我们将使用FX来自动化这个“插入”过程，而无需修改任何源。

首先，让我们排除一些导入(我们稍后将在代码中使用所有这些)。

```python
import statistics, tabulate, time
from typing import Any, Dict, List
from torch.fx import Interpreter
```

**注意：**

```
tabulate是一个外部库，它不是PyTorch的依赖项。我们将使用它来更容易地可视化性能数据。请确保您已经从您最喜欢的Python包源安装了它。
```

## 用符号跟踪捕获模型

我们将使用FX的符号跟踪机制，在我们可以操作和检查的数据结构中捕获模型的定义。

```python
traced_rn18 = torch.fx.symbolic_trace(rn18)
print(traced_rn18.graph)
```

**Out:**

```
graph():
    %x : torch.Tensor [#users=1] = placeholder[target=x]
    %conv1 : [#users=1] = call_module[target=conv1](args = (%x,), kwargs = {})
    %bn1 : [#users=1] = call_module[target=bn1](args = (%conv1,), kwargs = {})
    %relu : [#users=1] = call_module[target=relu](args = (%bn1,), kwargs = {})
    %maxpool : [#users=2] = call_module[target=maxpool](args = (%relu,), kwargs = {})
    %layer1_0_conv1 : [#users=1] = call_module[target=layer1.0.conv1](args = (%maxpool,), kwargs = {})
    %layer1_0_bn1 : [#users=1] = call_module[target=layer1.0.bn1](args = (%layer1_0_conv1,), kwargs = {})
    %layer1_0_relu : [#users=1] = call_module[target=layer1.0.relu](args = (%layer1_0_bn1,), kwargs = {})
    %layer1_0_conv2 : [#users=1] = call_module[target=layer1.0.conv2](args = (%layer1_0_relu,), kwargs = {})
    %layer1_0_bn2 : [#users=1] = call_module[target=layer1.0.bn2](args = (%layer1_0_conv2,), kwargs = {})
    %add : [#users=1] = call_function[target=operator.add](args = (%layer1_0_bn2, %maxpool), kwargs = {})
    %layer1_0_relu_1 : [#users=2] = call_module[target=layer1.0.relu](args = (%add,), kwargs = {})
    %layer1_1_conv1 : [#users=1] = call_module[target=layer1.1.conv1](args = (%layer1_0_relu_1,), kwargs = {})
    %layer1_1_bn1 : [#users=1] = call_module[target=layer1.1.bn1](args = (%layer1_1_conv1,), kwargs = {})
    %layer1_1_relu : [#users=1] = call_module[target=layer1.1.relu](args = (%layer1_1_bn1,), kwargs = {})
    %layer1_1_conv2 : [#users=1] = call_module[target=layer1.1.conv2](args = (%layer1_1_relu,), kwargs = {})
    %layer1_1_bn2 : [#users=1] = call_module[target=layer1.1.bn2](args = (%layer1_1_conv2,), kwargs = {})
    %add_1 : [#users=1] = call_function[target=operator.add](args = (%layer1_1_bn2, %layer1_0_relu_1), kwargs = {})
    %layer1_1_relu_1 : [#users=2] = call_module[target=layer1.1.relu](args = (%add_1,), kwargs = {})
    %layer2_0_conv1 : [#users=1] = call_module[target=layer2.0.conv1](args = (%layer1_1_relu_1,), kwargs = {})
    %layer2_0_bn1 : [#users=1] = call_module[target=layer2.0.bn1](args = (%layer2_0_conv1,), kwargs = {})
    %layer2_0_relu : [#users=1] = call_module[target=layer2.0.relu](args = (%layer2_0_bn1,), kwargs = {})
    %layer2_0_conv2 : [#users=1] = call_module[target=layer2.0.conv2](args = (%layer2_0_relu,), kwargs = {})
    %layer2_0_bn2 : [#users=1] = call_module[target=layer2.0.bn2](args = (%layer2_0_conv2,), kwargs = {})
    %layer2_0_downsample_0 : [#users=1] = call_module[target=layer2.0.downsample.0](args = (%layer1_1_relu_1,), kwargs = {})
    %layer2_0_downsample_1 : [#users=1] = call_module[target=layer2.0.downsample.1](args = (%layer2_0_downsample_0,), kwargs = {})
    %add_2 : [#users=1] = call_function[target=operator.add](args = (%layer2_0_bn2, %layer2_0_downsample_1), kwargs = {})
    %layer2_0_relu_1 : [#users=2] = call_module[target=layer2.0.relu](args = (%add_2,), kwargs = {})
    %layer2_1_conv1 : [#users=1] = call_module[target=layer2.1.conv1](args = (%layer2_0_relu_1,), kwargs = {})
    %layer2_1_bn1 : [#users=1] = call_module[target=layer2.1.bn1](args = (%layer2_1_conv1,), kwargs = {})
    %layer2_1_relu : [#users=1] = call_module[target=layer2.1.relu](args = (%layer2_1_bn1,), kwargs = {})
    %layer2_1_conv2 : [#users=1] = call_module[target=layer2.1.conv2](args = (%layer2_1_relu,), kwargs = {})
    %layer2_1_bn2 : [#users=1] = call_module[target=layer2.1.bn2](args = (%layer2_1_conv2,), kwargs = {})
    %add_3 : [#users=1] = call_function[target=operator.add](args = (%layer2_1_bn2, %layer2_0_relu_1), kwargs = {})
    %layer2_1_relu_1 : [#users=2] = call_module[target=layer2.1.relu](args = (%add_3,), kwargs = {})
    %layer3_0_conv1 : [#users=1] = call_module[target=layer3.0.conv1](args = (%layer2_1_relu_1,), kwargs = {})
    %layer3_0_bn1 : [#users=1] = call_module[target=layer3.0.bn1](args = (%layer3_0_conv1,), kwargs = {})
    %layer3_0_relu : [#users=1] = call_module[target=layer3.0.relu](args = (%layer3_0_bn1,), kwargs = {})
    %layer3_0_conv2 : [#users=1] = call_module[target=layer3.0.conv2](args = (%layer3_0_relu,), kwargs = {})
    %layer3_0_bn2 : [#users=1] = call_module[target=layer3.0.bn2](args = (%layer3_0_conv2,), kwargs = {})
    %layer3_0_downsample_0 : [#users=1] = call_module[target=layer3.0.downsample.0](args = (%layer2_1_relu_1,), kwargs = {})
    %layer3_0_downsample_1 : [#users=1] = call_module[target=layer3.0.downsample.1](args = (%layer3_0_downsample_0,), kwargs = {})
    %add_4 : [#users=1] = call_function[target=operator.add](args = (%layer3_0_bn2, %layer3_0_downsample_1), kwargs = {})
    %layer3_0_relu_1 : [#users=2] = call_module[target=layer3.0.relu](args = (%add_4,), kwargs = {})
    %layer3_1_conv1 : [#users=1] = call_module[target=layer3.1.conv1](args = (%layer3_0_relu_1,), kwargs = {})
    %layer3_1_bn1 : [#users=1] = call_module[target=layer3.1.bn1](args = (%layer3_1_conv1,), kwargs = {})
    %layer3_1_relu : [#users=1] = call_module[target=layer3.1.relu](args = (%layer3_1_bn1,), kwargs = {})
    %layer3_1_conv2 : [#users=1] = call_module[target=layer3.1.conv2](args = (%layer3_1_relu,), kwargs = {})
    %layer3_1_bn2 : [#users=1] = call_module[target=layer3.1.bn2](args = (%layer3_1_conv2,), kwargs = {})
    %add_5 : [#users=1] = call_function[target=operator.add](args = (%layer3_1_bn2, %layer3_0_relu_1), kwargs = {})
    %layer3_1_relu_1 : [#users=2] = call_module[target=layer3.1.relu](args = (%add_5,), kwargs = {})
    %layer4_0_conv1 : [#users=1] = call_module[target=layer4.0.conv1](args = (%layer3_1_relu_1,), kwargs = {})
    %layer4_0_bn1 : [#users=1] = call_module[target=layer4.0.bn1](args = (%layer4_0_conv1,), kwargs = {})
    %layer4_0_relu : [#users=1] = call_module[target=layer4.0.relu](args = (%layer4_0_bn1,), kwargs = {})
    %layer4_0_conv2 : [#users=1] = call_module[target=layer4.0.conv2](args = (%layer4_0_relu,), kwargs = {})
    %layer4_0_bn2 : [#users=1] = call_module[target=layer4.0.bn2](args = (%layer4_0_conv2,), kwargs = {})
    %layer4_0_downsample_0 : [#users=1] = call_module[target=layer4.0.downsample.0](args = (%layer3_1_relu_1,), kwargs = {})
    %layer4_0_downsample_1 : [#users=1] = call_module[target=layer4.0.downsample.1](args = (%layer4_0_downsample_0,), kwargs = {})
    %add_6 : [#users=1] = call_function[target=operator.add](args = (%layer4_0_bn2, %layer4_0_downsample_1), kwargs = {})
    %layer4_0_relu_1 : [#users=2] = call_module[target=layer4.0.relu](args = (%add_6,), kwargs = {})
    %layer4_1_conv1 : [#users=1] = call_module[target=layer4.1.conv1](args = (%layer4_0_relu_1,), kwargs = {})
    %layer4_1_bn1 : [#users=1] = call_module[target=layer4.1.bn1](args = (%layer4_1_conv1,), kwargs = {})
    %layer4_1_relu : [#users=1] = call_module[target=layer4.1.relu](args = (%layer4_1_bn1,), kwargs = {})
    %layer4_1_conv2 : [#users=1] = call_module[target=layer4.1.conv2](args = (%layer4_1_relu,), kwargs = {})
    %layer4_1_bn2 : [#users=1] = call_module[target=layer4.1.bn2](args = (%layer4_1_conv2,), kwargs = {})
    %add_7 : [#users=1] = call_function[target=operator.add](args = (%layer4_1_bn2, %layer4_0_relu_1), kwargs = {})
    %layer4_1_relu_1 : [#users=1] = call_module[target=layer4.1.relu](args = (%add_7,), kwargs = {})
    %avgpool : [#users=1] = call_module[target=avgpool](args = (%layer4_1_relu_1,), kwargs = {})
    %flatten : [#users=1] = call_function[target=torch.flatten](args = (%avgpool, 1), kwargs = {})
    %fc : [#users=1] = call_module[target=fc](args = (%flatten,), kwargs = {})
    return fc
```

这为我们提供了ResNet18模型的Graph表示。图形由一系列相互连接的节点组成。每个节点表示Python代码中的一个调用站点(无论是对函数、模块还是方法)，边(在每个节点上表示为args和kwargs)表示在这些调用站点之间传递的值。有关Graph表示和FX其他API的更多信息，请参阅FX文档https://pytorch.org/docs/master/fx.html。

## 创建一个评测解释器

下一步，我们将创建一个继承自torch.fx.Interpreter的类。虽然symbolic_trace生成的GraphModule编译了调用GraphModule时运行的Python代码，但运行GraphModule的另一种方法是逐个执行Graph中的每个节点。这就是解释器提供的功能：它逐节点解释图形。

通过继承解释器，我们可以覆盖各种功能并安装我们想要的评测行为。我们的目标是拥有一个对象，我们可以将模型传递给该对象，调用该模型1次或多次，然后获得关于模型和模型的每个部分在这些运行过程中花费的时间的统计信息。

让我们定义ProfileInterpreter类：

```python
class ProfilingInterpreter(Interpreter):
    def __init__(self, mod : torch.nn.Module):
        # 在构造函数中符号追踪模型,让用户不需要显式写追踪代码
        gm = torch.fx.symbolic_trace(mod) 
        super().__init__(gm)
        
        # 存储两个统计信息:
        # 1. 模型总执行时间列表
        self.total_runtime_sec : List[float] = []
        # 2. 每个节点执行时间列表的映射表
        self.runtimes_sec : Dict[torch.fx.Node, List[float]] = {}

    # 覆盖模型执行入口函数
    def run(self, *args) -> Any:
        # 记录开始时间
        t_start = time.time()
        # 委托父类运行模型
        return_val = super().run(*args)
        # 记录结束时间 
        t_end = time.time()
        # 记录模型执行总时间
        self.total_runtime_sec.append(t_end - t_start) 
        return return_val

    # 覆盖节点执行函数 
    def run_node(self, n : torch.fx.Node) -> Any:
        # 记录节点开始时间
        t_start = time.time()
        # 委托父类执行节点
        return_val = super().run_node(n)
        # 记录节点结束时间
        t_end = time.time()
        # 记录该节点执行时间
        self.runtimes_sec.setdefault(n, []).append(t_end - t_start)
        return return_val

    # 生成统计结果摘要
    def summary(self, should_sort : bool = False) -> str:
        # 收集每个节点的统计信息
        node_summaries = []
        # 计算模型平均执行时间
        mean_total_runtime = statistics.mean(self.total_runtime_sec)
        
        for node, runtimes in self.runtimes_sec.items():
            # 计算节点平均执行时间
            mean_runtime = statistics.mean(runtimes) 
            # 计算节点执行时间占比
            pct_total = mean_runtime / mean_total_runtime * 100
            # 收集节点统计信息
            node_summaries.append([node.op, str(node), mean_runtime, pct_total])

        # 按执行时间排序
        if should_sort:
            node_summaries.sort(key=lambda s: s[2], reverse=True)

        # 生成表格格式的统计结果
        headers = ['Op type', 'Op', 'Average runtime (s)', 'Pct total runtime']
        return tabulate.tabulate(node_summaries, headers=headers)
```

**注意：**

```
我们使用Python的time.time函数来提取墙上的时钟时间戳并进行比较。这不是衡量性能的最准确的方法，只会给我们一个一阶近似值。我们使用这种简单的技术只是为了在本教程中进行演示。
```

## 研究ResNet18的性能

我们现在可以使用ProfileInterpreter来检查我们的ResNet18模型的性能特征；

```python
interp = ProfilingInterpreter(rn18)
interp.run(input)
print(interp.summary(True))
```

**Out:**

```
Op type        Op                       Average runtime (s)    Pct total runtime
-------------  ---------------------  ---------------------  -------------------
call_module    conv1                            0.00597668             8.14224
call_module    maxpool                          0.00543523             7.4046
call_module    layer4_1_conv1                   0.00426865             5.81533
call_module    layer1_0_conv1                   0.00406504             5.53794
call_module    layer4_1_conv2                   0.00402284             5.48045
call_module    layer4_0_conv2                   0.00396276             5.3986
call_module    layer1_0_conv2                   0.00365305             4.97668
call_module    layer1_1_conv2                   0.00350165             4.77043
call_module    layer2_1_conv2                   0.00348306             4.74509
call_module    layer1_1_conv1                   0.00340676             4.64115
call_module    layer2_0_conv2                   0.00339365             4.62329
call_module    layer3_1_conv1                   0.00333881             4.54858
call_module    layer3_1_conv2                   0.00331688             4.5187
call_module    layer2_1_conv1                   0.00326943             4.45407
call_module    layer3_0_conv2                   0.00323319             4.4047
call_module    layer4_0_conv1                   0.00231767             3.15744
call_module    layer2_0_conv1                   0.00209737             2.85732
call_module    layer3_0_conv1                   0.00195694             2.66601
call_module    bn1                              0.00132489             1.80495
call_module    layer2_0_downsample_0            0.000714064            0.972794
call_module    layer3_0_downsample_0            0.000543594            0.740558
call_module    layer4_0_downsample_0            0.00048089             0.655134
call_function  add                              0.000338316            0.4609
call_module    relu                             0.000316858            0.431667
call_function  add_1                            0.000294209            0.400811
call_module    layer1_0_bn1                     0.000254154            0.346243
call_module    fc                               0.000243187            0.331302
call_module    layer1_0_bn2                     0.000216007            0.294274
call_module    layer1_1_bn1                     0.000182867            0.249126
call_module    layer1_1_bn2                     0.000170946            0.232886
call_function  add_3                            0.000162363            0.221193
call_module    layer3_1_bn1                     0.000135183            0.184165
call_module    layer2_1_bn2                     0.000134945            0.18384
call_module    layer3_1_bn2                     0.000132799            0.180917
call_module    layer2_0_downsample_1            0.00011754             0.160129
call_module    avgpool                          0.000114202            0.155582
call_module    layer1_0_relu                    0.000112772            0.153633
call_module    layer2_0_bn1                     0.000109196            0.148761
call_module    layer1_0_relu_1                  0.000102997            0.140316
call_module    layer2_0_bn2                     0.000102758            0.139991
call_module    layer2_1_bn1                     0.000100613            0.137068
call_module    layer3_0_downsample_1            9.36985e-05            0.127649
call_module    layer4_1_bn1                     9.36985e-05            0.127649
call_module    layer3_0_bn2                     9.25064e-05            0.126025
call_function  add_5                            9.13143e-05            0.124401
call_function  add_2                            9.10759e-05            0.124076
call_module    layer4_0_bn2                     8.86917e-05            0.120828
call_module    layer1_1_relu                    8.58307e-05            0.11693
call_module    layer3_0_bn1                     8.51154e-05            0.115956
call_module    layer1_1_relu_1                  8.41618e-05            0.114657
call_module    layer4_1_bn2                     8.24928e-05            0.112383
call_module    layer4_0_bn1                     7.82013e-05            0.106536
call_module    layer4_0_downsample_1            7.77245e-05            0.105887
call_function  add_7                            7.62939e-05            0.103938
call_function  add_4                            6.36578e-05            0.0867232
call_module    layer2_0_relu                    5.6982e-05             0.0776287
call_function  add_6                            5.6982e-05             0.0776287
call_module    layer4_1_relu                    5.57899e-05            0.0760046
call_module    layer2_1_relu_1                  5.55515e-05            0.0756798
call_module    layer2_0_relu_1                  5.34058e-05            0.0727566
call_module    layer4_0_relu                    5.10216e-05            0.0695085
call_module    layer2_1_relu                    5.07832e-05            0.0691837
call_module    layer4_0_relu_1                  4.74453e-05            0.0646364
call_module    layer3_1_relu                    4.43459e-05            0.0604139
call_module    layer3_0_relu                    4.41074e-05            0.0600891
call_module    layer3_1_relu_1                  4.41074e-05            0.0600891
call_module    layer4_1_relu_1                  4.33922e-05            0.0591147
call_module    layer3_0_relu_1                  4.24385e-05            0.0578155
call_function  flatten                          3.40939e-05            0.0464473
placeholder    x                                1.88351e-05            0.0256597
output         output                           1.40667e-05            0.0191636
```

有两件事我们应该在这里指出：

- MaxPool2d占用的时间最多。这是一个已知的问题：https://github.com/pytorch/pytorch/issues/51393
- BatchNorm2d也占用大量时间。我们可以继续这一思路，并在Conv BN Fusion with FX教程中对此进行优化。

## 结论

正如我们所看到的，使用FX，我们可以很容易地以机器可解释的格式捕获PyTorch程序(即使是我们没有源代码的程序！)，并将其用于分析，例如我们在这里所做的性能分析。FX为PyTorch项目打开了一个充满可能性的激动人心的世界。

最后，由于FX仍处于测试阶段，我们很高兴听到您对使用它的任何反馈。请随时使用PyTorch论坛(https://discuss.pytorch.org/)和问题跟踪器(https://github.com/pytorch/pytorch/issues)以提供您可能有的任何反馈。

脚本的总运行时间：(0分0.339秒)。
