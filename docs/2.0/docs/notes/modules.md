# 模块 [¶](#modules "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/modules>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/modules.html>


 PyTorch 使用模块来表示神经网络。模块有：



* **状态计算的构建块。** PyTorch 提供了强大的模块库，使定义新的自定义模块变得简单，从而可以轻松构建复杂的多层神经网络。
* **与 PyTorch 紧密集成** [ autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) **system.** 模块使指定 PyTorch 优化器更新的可学习参数变得简单。
* **易于使用和转换** 模块可以直接保存和恢复、在 CPU /GPU /TPU 设备之间传输、修剪、量化等。


 本注释描述了模块，适用于所有 PyTorch 用户。由于模块对于 PyTorch 来说非常重要，因此本笔记中的许多主题都在其他笔记或教程中进行了详细阐述，并且此处还提供了许多这些文档的链接。



* [简单的自定义模块](#a-simple-custom-module)
* [作为构建块的模块](#modules-as-building-blocks)
* [使用模块进行神经网络训练](#neural-network-training-with-modules)
* [模块状态](#module-state)
* [模块初始化](#module-initialization)
* [模块挂钩](#module-hooks)
* [高级功能](#advanced-features)
  + [分布式训练](#distributed-training) 
  + [分析性能](#profiling-performance) 
  + [通过量化提高性能](#improving-performance-with-quantization) 
  + [通过修剪提高内存使用率](#improving-memory-usage-with-pruning)
  + [参数化](#parametrizations)
  + [使用 FX 转换模块](#transforming-modules-with-fx)


## [一个简单的自定义模块](#id4) [¶](#a-simple-custom-module "永久链接到此标题")


 首先，让我们看一下 PyTorch 的 [`Linear`](../generated/torch.nn.Linear.html#torch.nn.Linear "torch.nn.Linear") 模块的更简单的自定义版本。此模块对其输入应用仿射变换。


```
import torch
from torch import nn

class MyLinear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(in_features, out_features))
    self.bias = nn.Parameter(torch.randn(out_features))

  def forward(self, input):
    return (input @ self.weight) + self.bias

```


 这个简单的模块具有以下模块的基本特征：



* **它继承自 Module 基类。** 所有模块都应该子类 [`Module`](../generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")与其他模块的可组合性。
* **它定义了一些在计算中使用的“状态”。** 这里，状态由定义仿射变换的随机初始化的“权重”和“偏差”tensor组成。因为其中每个都被定义为 [`Parameter`](../generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter "torch.nn.parameter.Parameter") ，所以它们是 *为模块注册*，并将自动跟踪并从调用 [`parameters()`](../generated/torch.nn.Module.html#torch.nn.Module.parameters "torch.nn.Module.parameters")。参数可以被视为模块计算的“可学习”方面(稍后会详细介绍)。请注意，模块不需要有状态，也可以是无状态的。
* **它定义了一个执行计算的forward()函数。** 对于这个仿射变换模块，输入是矩阵乘以“权重”参数(使用“@”简写符号)并添加到“bias”参数以生成输出。更一般地，模块的“forward()”实现可以执行涉及任意数量的输入和输出的任意计算。


 这个简单的模块演示了模块如何将状态和计算打包在一起。可以构造和调用该模块的实例：


```
m = MyLinear(4, 3)
sample_input = torch.randn(4)
m(sample_input)
: tensor([-0.3037, -1.0413, -4.2057], grad_fn=<AddBackward0>)

```


 请注意，模块本身是可调用的，并且调用它会调用其“forward()”函数。此名称引用了“前向传递”和“后向传递”的概念，这些概念适用于每个模块。“前向传递” ” 负责将模块表示的计算应用到给定的输入(如上面的代码片段所示)。 “向后传递”计算模块输出相对于其输入的梯度，这可用于通过梯度下降方法“训练”参数。 PyTorch 的 autograd 系统自动处理这种向后传递计算，因此不需要为每个模块手动实现“backward()”函数。 [使用模块进行神经网络训练](#neural-network-training-with-modules) 详细介绍了通过连续的前向/后向传递来训练模块参数的过程。


 模块注册的完整参数集可以通过调用 [`parameters()`](../generated/torch.nn.Module.html#torch.nn.Module.parameters "torch.nn. Module.parameters") 或 [`named_parameters()`](../generated/torch.nn.Module.html#torch.nn.Module.named_pa​​rameters "torch.nn.Module.named_pa​​rameters") ，其中后者包括每个参数的名称：


```
for parameter in m.named_parameters():
  print(parameter)
: ('weight', Parameter containing:
tensor([[ 1.0597,  1.1796,  0.8247],
        [-0.5080, -1.2635, -1.1045],
        [ 0.0593,  0.2469, -1.4299],
        [-0.4926, -0.5457,  0.4793]], requires_grad=True))
('bias', Parameter containing:
tensor([ 0.3634,  0.2015, -0.8525], requires_grad=True))

```


 一般来说，模块注册的参数是模块计算中应该“学习”的方面。本说明的后面部分展示了如何使用 PyTorch 的优化器之一来更新这些参数。不过，在我们开始之前，让我们首先检查一下模块如何相互组合。


## [模块作为构建块](#id5) [¶](#modules-as-building-blocks "此标题的永久链接")


 模块可以包含其他模块，使它们成为开发更复杂功能的有用构建块。最简单的方法是使用 [`Sequential`](../generated/torch.nn.Sequential.html#torch.nn.Sequential "torch.nn.Sequential")模块。它允许我们将多个模块链接在一起：


```
net = nn.Sequential(
  MyLinear(4, 3),
  nn.ReLU(),
  MyLinear(3, 1)
)

sample_input = torch.randn(4)
net(sample_input)
: tensor([-0.6749], grad_fn=<AddBackward0>)

```


 请注意， [`Sequential`](../generated/torch.nn.Sequential.html#torch.nn.Sequential "torch.nn.Sequential") 自动将第一个 `MyLinear` 模块的输出作为输入提供给 [` ReLU`](../generated/torch.nn.ReLU.html#torch.nn.ReLU "torch.nn.ReLU") ，并将其输出作为第二个 `MyLinear` 模块的输入。如图所示，它仅限于具有单个输入和输出的模块的有序链接。


 一般来说，建议为最简单用例之外的任何内容定义自定义模块，因为这为如何使用子模块进行模块计算提供了充分的灵活性。


 例如，这是一个作为自定义模块实现的简单神经网络：


```
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.l0 = MyLinear(4, 3)
    self.l1 = MyLinear(3, 1)
  def forward(self, x):
    x = self.l0(x)
    x = F.relu(x)
    x = self.l1(x)
    return x

```


 该模块由两个“子模块”或“子模块”(“l0”和“l1”)组成，它们定义神经网络的层，并用于模块的“forward()”方法内的计算。模块的直接子级可以通过调用 [`children()`](../generated/torch.nn.Module.html#torch.nn.Module.children "torch.nn.Module.children") 进行迭代或 [`named_children()`](../generated/torch.nn.Module.html#torch.nn.Module.named_children "torch.nn.Module.named_children") ：


```
net = Net()
for child in net.named_children():
  print(child)
: ('l0', MyLinear())
('l1', MyLinear())

```


 要深入了解不仅仅是直接子级， [`modules()`](../generated/torch.nn.Module.html#torch.nn.Module.modules "torch.nn.Module.modules") 和 [` name_modules()`](../generated/torch.nn.Module.html#torch.nn.Module.named_modules "torch.nn.Module.named_modules")*递归地*迭代模块及其子模块：


```
class BigNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = MyLinear(5, 4)
    self.net = Net()
  def forward(self, x):
    return self.net(self.l1(x))

big_net = BigNet()
for module in big_net.named_modules():
  print(module)
: ('', BigNet(
  (l1): MyLinear()
  (net): Net(
    (l0): MyLinear()
    (l1): MyLinear()
  )
))
('l1', MyLinear())
('net', Net(
  (l0): MyLinear()
  (l1): MyLinear()
))
('net.l0', MyLinear())
('net.l1', MyLinear())

```


 有时，模块需要动态定义子模块。 [`ModuleList`](../generated/torch.nn.ModuleList.html#torch.nn.ModuleList "torch.nn.ModuleList") 和 [`ModuleDict` ](../generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict "torch.nn.ModuleDict") 模块在这里很有用；他们从列表或字典中注册子模块：


```
class DynamicNet(nn.Module):
  def __init__(self, num_layers):
    super().__init__()
    self.linears = nn.ModuleList(
      [MyLinear(4, 4) for _ in range(num_layers)])
    self.activations = nn.ModuleDict({
      'relu': nn.ReLU(),
      'lrelu': nn.LeakyReLU()
    })
    self.final = MyLinear(4, 1)
  def forward(self, x, act):
    for linear in self.linears:
      x = linear(x)
    x = self.activationsact
    x = self.final(x)
    return x

dynamic_net = DynamicNet(3)
sample_input = torch.randn(4)
output = dynamic_net(sample_input, 'relu')

```


 对于任何给定的模块，其参数由其直接参数以及所有子模块的参数组成。这意味着调用 [`parameters()`](../generated/torch.nn.Module.html#torch.nn.Module.parameters "torch.nn.Module.parameters") 和 [`named_parameters()`](../generated/torch.nn.Module.html#torch.nn.Module.named_pa​​rameters "torch.nn. Module.named_pa​​rameters") 将递归地包含子参数，以便方便地优化网络内的所有参数：


```
for parameter in dynamic_net.named_parameters():
  print(parameter)
: ('linears.0.weight', Parameter containing:
tensor([[-1.2051,  0.7601,  1.1065,  0.1963],
        [ 3.0592,  0.4354,  1.6598,  0.9828],
        [-0.4446,  0.4628,  0.8774,  1.6848],
        [-0.1222,  1.5458,  1.1729,  1.4647]], requires_grad=True))
('linears.0.bias', Parameter containing:
tensor([ 1.5310,  1.0609, -2.0940,  1.1266], requires_grad=True))
('linears.1.weight', Parameter containing:
tensor([[ 2.1113, -0.0623, -1.0806,  0.3508],
        [-0.0550,  1.5317,  1.1064, -0.5562],
        [-0.4028, -0.6942,  1.5793, -1.0140],
        [-0.0329,  0.1160, -1.7183, -1.0434]], requires_grad=True))
('linears.1.bias', Parameter containing:
tensor([ 0.0361, -0.9768, -0.3889,  1.1613], requires_grad=True))
('linears.2.weight', Parameter containing:
tensor([[-2.6340, -0.3887, -0.9979,  0.0767],
        [-0.3526,  0.8756, -1.5847, -0.6016],
        [-0.3269, -0.1608,  0.2897, -2.0829],
        [ 2.6338,  0.9239,  0.6943, -1.5034]], requires_grad=True))
('linears.2.bias', Parameter containing:
tensor([ 1.0268,  0.4489, -0.9403,  0.1571], requires_grad=True))
('final.weight', Parameter containing:
tensor([[ 0.2509], [-0.5052], [ 0.3088], [-1.4951]], requires_grad=True))
('final.bias', Parameter containing:
tensor([0.3381], requires_grad=True))

```


 使用 [`to()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to "torch.nn.Module.to") 也可以轻松地将所有参数移动到不同的设备或更改其精度：


```
# Move all parameters to a CUDA device
dynamic_net.to(device='cuda')

# Change precision of all parameters
dynamic_net.to(dtype=torch.float64)

dynamic_net(torch.randn(5, device='cuda', dtype=torch.float64))
: tensor([6.5166], device='cuda:0', dtype=torch.float64, grad_fn=<AddBackward0>)

```

 更一般地，可以使用 [`apply()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply "torch.nn.Module.apply") 函数将任意函数递归地应用于模块及其子模块。 例如，要将自定义初始化应用于模块及其子模块的参数：


```
# Define a function to initialize Linear weights.
# Note that no_grad() is used here to avoid tracking this computation in the autograd graph.
@torch.no_grad()
def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_normal_(m.weight)
    m.bias.fill_(0.0)

# Apply the function recursively on the module and its submodules.
dynamic_net.apply(init_weights)

```


 这些例子展示了如何通过模块组合来形成复杂的神经网络并方便地操作。为了能够以最少的样板快速轻松地构建神经网络，PyTorch 在 [`torch.nn`](../nn.html#module-torch.nn "torch.nn") 命名空间内提供了一个大型的高性能模块库执行常见的神经网络操作，如池化、卷积、损失函数等。


 在下一节中，我们将给出训练神经网络的完整示例。


 欲了解更多信息，请查看：



* PyTorch 提供的模块库：[torch.nn](https://pytorch.org/docs/stable/nn.html)
* 定义神经网络模块：<https://pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html>


## [使用模块进行神经网络训练](#id6) [¶](#neural-network-training-with-modules "永久链接到此标题")


 网络构建完成后，必须对其进行训练，并且可以使用 [`torch.optim`](https://pytorch.org/docs/stable/optim.html#module-torch.optim "module-torch.optim") 中的 PyTorch 优化器之一轻松优化其参数：


```
# Create the network (from previous section) and optimizer
net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

# Run a sample training loop that "teaches" the network
# to output the constant zero function
for _ in range(10000):
  input = torch.randn(4)
  output = net(input)
  loss = torch.abs(output)
  net.zero_grad()
  loss.backward()
  optimizer.step()

# After training, switch the module to eval mode to do inference, compute performance metrics, etc.
# (see discussion below for a description of training and evaluation modes)
...
net.eval()
...

```


 在这个简化的示例中，网络学习简单地输出零，因为任何非零输出都会通过使用 [`torch.abs()`](https://pytorch.org/docs/stable/generated/torch.abs.html#torch.abs "torch.abs") 作为损失函数，根据其绝对值进行“惩罚”。 虽然这不是一项非常有趣的任务，但培训的关键部分是存在的：



* 创建一个网络。
* 创建一个优化器(在本例中为随机梯度下降优化器)，并且网络的参数与其关联。
* 训练循环...
  + 获取输入， 
  + 运行网络， 
  + 计算损失， 
  + 将网络参数的梯度归零， 
  + 调用loss.backward() 更新参数的梯度， 
  + 调用optimizer.step() 将梯度应用到参数。


 运行上述代码片段后，请注意网络参数已更改。特别是，检查 `l1` 的 `weight` 参数的值表明它的值现在更接近 0(正如预期的那样)：


```
print(net.l1.weight)
: Parameter containing:
tensor([[-0.0013],
        [ 0.0030],
        [-0.0008]], requires_grad=True)

```


 请注意，上述过程完全是在网络模块处于“训练模式”时完成的。模块默认为训练模式，可以使用 [`train()`](../generated/torch.nn.Module.html#torch.nn.Module.train "torch.nn.Module.train 在训练和评估模式之间切换") 和 [`eval()`](../generated/torch.nn.Module.html#torch.nn.Module.eval "torch.nn.Module.eval") 。它们的行为可能有所不同，具体取决于它们所处的模式。例如，“BatchNorm”模块在训练期间维护运行平均值和方差，而当模块处于评估模式时，这些平均值和方差不会更新。一般来说，模块在训练期间应处于训练模式，仅在进行推理或评估时才切换到评估模式。以下是自定义模块的示例，该模块在两种模式之间的行为有所不同：


```
class ModalModule(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    if self.training:
      # Add a constant only in training mode.
      return x + 1.
    else:
      return x


m = ModalModule()
x = torch.randn(4)

print('training mode output: {}'.format(m(x)))
: tensor([1.6614, 1.2669, 1.0617, 1.6213, 0.5481])

m.eval()
print('evaluation mode output: {}'.format(m(x)))
: tensor([ 0.6614,  0.2669,  0.0617,  0.6213, -0.4519])

```


 训练神经网络通常很棘手。欲了解更多信息，请查看：



* 使用优化器：<https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html>。
* 神经网络训练：<https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html>
* 简介自动毕业：<https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>


## [模块状态](#id7) [¶](#module-state "此标题的永久链接")


 在上一节中，我们演示了训练模块的“参数”，或者计算的可学习方面。 现在，如果我们想将训练后的模型保存到磁盘，我们可以通过保存其 state_dict (即“状态字典”)来实现：


```
# Save the module
torch.save(net.state_dict(), 'net.pt')

...

# Load the module later on
new_net = Net()
new_net.load_state_dict(torch.load('net.pt'))
: <All keys matched successfully>

```


 模块的“state_dict”包含影响其计算的状态。这包括但不限于模块的参数。对于某些模块，拥有超出影响模块计算的参数的状态可能很有用，但不可学习。对于这种情况，PyTorch 提供了“缓冲区”的概念，包括“持久”和“非持久”。以下是模块可以具有的各种状态类型的概述：



* **参数**：计算的可学习方面；包含在 `state_dict`
* **Buffers** 中：计算的不可学习方面



+ **持久**缓冲区：包含在 `state_dict` 中(即在保存和加载时序列化) 
+ **非持久**缓冲区：不包含在 `state_dict` 中(即不进行序列化)


 作为使用缓冲区的激励示例，请考虑一个维护运行平均值的简单模块。我们希望运行平均值的当前值被视为模块“state_dict”的一部分，以便在加载模块的序列化形式时它将被恢复，但我们不希望它是可学习的。此代码片段显示了如何使用 [`register_buffer()`](../generated/torch.nn.Module.html#torch.nn.Module.register_buffer "torch.nn.Module.register_buffer") 来完成此操作：


```
class RunningMean(nn.Module):
  def __init__(self, num_features, momentum=0.9):
    super().__init__()
    self.momentum = momentum
    self.register_buffer('mean', torch.zeros(num_features))
  def forward(self, x):
    self.mean = self.momentum * self.mean + (1.0 - self.momentum) * x
    return self.mean

```


 现在，运行平均值的当前值被视为模块“state_dict”的一部分，并且在从磁盘加载模块时将被正确恢复：


```
m = RunningMean(4)
for _ in range(10):
  input = torch.randn(4)
  m(input)

print(m.state_dict())
: OrderedDict([('mean', tensor([ 0.1041, -0.1113, -0.0647,  0.1515]))]))

# Serialized form will contain the 'mean' tensor
torch.save(m.state_dict(), 'mean.pt')

m_loaded = RunningMean(4)
m_loaded.load_state_dict(torch.load('mean.pt'))
assert(torch.all(m.mean == m_loaded.mean))

```


 如前所述，可以通过将缓冲区标记为非持久性来将其排除在模块的“state_dict”之外：


```
self.register_buffer('unserialized_thing', torch.randn(5), persistent=False)

```


 持久缓冲区和非持久缓冲区都会受到使用 [`to()`](../generated/torch.nn.Module.html#torch.nn.Module.to "torch. nn.模块.to") :


```
# Moves all module parameters and buffers to the specified device / dtype
m.to(device='cuda', dtype=torch.float64)

```


 模块的缓冲区可以使用 [`buffers()`](../generated/torch.nn.Module.html#torch.nn.Module.buffers "torch.nn.Module.buffers") 或 [`命名_buffers()`](../generated/torch.nn.Module.html#torch.nn.Module.named_buffers“torch.nn.Module.named_buffers”)。


```
for buffer in m.named_buffers():
  print(buffer)

```


 以下类演示了在模块内注册参数和缓冲区的各种方法：


```
class StatefulModule(nn.Module):
  def __init__(self):
    super().__init__()
    # Setting a nn.Parameter as an attribute of the module automatically registers the tensor
    # as a parameter of the module.
    self.param1 = nn.Parameter(torch.randn(2))

    # Alternative string-based way to register a parameter.
    self.register_parameter('param2', nn.Parameter(torch.randn(3)))

    # Reserves the "param3" attribute as a parameter, preventing it from being set to anything
    # except a parameter. "None" entries like this will not be present in the module's state_dict.
    self.register_parameter('param3', None)

    # Registers a list of parameters.
    self.param_list = nn.ParameterList([nn.Parameter(torch.randn(2)) for i in range(3)])

    # Registers a dictionary of parameters.
    self.param_dict = nn.ParameterDict({
      'foo': nn.Parameter(torch.randn(3)),
      'bar': nn.Parameter(torch.randn(4))
    })

    # Registers a persistent buffer (one that appears in the module's state_dict).
    self.register_buffer('buffer1', torch.randn(4), persistent=True)

    # Registers a non-persistent buffer (one that does not appear in the module's state_dict).
    self.register_buffer('buffer2', torch.randn(5), persistent=False)

    # Reserves the "buffer3" attribute as a buffer, preventing it from being set to anything
    # except a buffer. "None" entries like this will not be present in the module's state_dict.
    self.register_buffer('buffer3', None)

    # Adding a submodule registers its parameters as parameters of the module.
    self.linear = nn.Linear(2, 3)

m = StatefulModule()

# Save and load state_dict.
torch.save(m.state_dict(), 'state.pt')
m_loaded = StatefulModule()
m_loaded.load_state_dict(torch.load('state.pt'))

# Note that non-persistent buffer "buffer2" and reserved attributes "param3" and "buffer3" do
# not appear in the state_dict.
print(m_loaded.state_dict())
: OrderedDict([('param1', tensor([-0.0322,  0.9066])),
               ('param2', tensor([-0.4472,  0.1409,  0.4852])),
               ('buffer1', tensor([ 0.6949, -0.1944,  1.2911, -2.1044])),
               ('param_list.0', tensor([ 0.4202, -0.1953])),
               ('param_list.1', tensor([ 1.5299, -0.8747])),
               ('param_list.2', tensor([-1.6289,  1.4898])),
               ('param_dict.bar', tensor([-0.6434,  1.5187,  0.0346, -0.4077])),
               ('param_dict.foo', tensor([-0.0845, -1.4324,  0.7022])),
               ('linear.weight', tensor([[-0.3915, -0.6176],
                                         [ 0.6062, -0.5992],
                                         [ 0.4452, -0.2843]])),
               ('linear.bias', tensor([-0.3710, -0.0795, -0.3947]))])

```


 欲了解更多信息，请查看：



* 保存和加载：<https://pytorch.org/tutorials/beginner/saving_loading_models.html>
* 序列化语义：<https://pytorch.org/docs/main/notes/serialization.html>
* 什么是状态听写？ <https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html>


## [模块初始化](#id8) [¶](#module-initialization "永久链接到此标题")


 默认情况下，[`torch.nn`](../nn.html#module-torch.nn "torch.nn") 提供的模块的参数和浮点缓冲区在模块实例化期间初始化为 32 位浮点值在 CPU 上使用初始化方案确定该模块类型历史上表现良好。对于某些用例，可能需要使用不同的类型、设备(例如 GPU)或初始化技术进行初始化。


 例子：


```
# Initialize module directly onto GPU.
m = nn.Linear(5, 3, device='cuda')

# Initialize module with 16-bit floating point parameters.
m = nn.Linear(5, 3, dtype=torch.half)

# Skip default parameter initialization and perform custom (e.g. orthogonal) initialization.
m = torch.nn.utils.skip_init(nn.Linear, 5, 3)
nn.init.orthogonal_(m.weight)

```


 请注意，上面演示的设备和数据类型选项也适用于为模块注册的任何浮点缓冲区：


```
m = nn.BatchNorm2d(3, dtype=torch.half)
print(m.running_mean)
: tensor([0., 0., 0.], dtype=torch.float16)

```


 虽然模块编写者可以使用任何设备或 dtype 来初始化其自定义模块中的参数，但好的做法是默认使用“dtype=torch.float”和“device='cpu'”。或者，您可以通过遵守上面演示的所有 [`torch.nn`](../nn.html#module-torch.nn "torch.nn") 模块遵循的约定，为自定义模块在这些领域提供充分的灵活性:



* 提供适用于模块注册的任何参数/缓冲区的“device”构造函数 kwarg。
* 提供适用于模块注册的任何参数/浮点缓冲区的“dtype”构造函数 kwarg。
* 仅使用初始化函数(即函数来自“torch.nn.init”)模块构造函数内的参数和缓冲区。请注意，这仅需要使用 [`skip_init()`](../generated/torch.nn.utils.skip_init.html#torch.nn.utils.skip_init "torch.nn.utils.skip_init") ;请参阅[此页面](https://pytorch.org/tutorials/prototype/skip_param_init.html#updating-modules-to-support-skipping-initialization)以获取解释。


 欲了解更多信息，请查看：



* 跳过模块参数初始化：<https://pytorch.org/tutorials/prototype/skip_param_init.html>


## [模块挂钩](#id9) [¶](#module-hooks "此标题的永久链接")


 在[使用模块进行神经网络训练](#neural-network-training-with-modules)中，我们演示了模块的训练过程，该过程迭代地执行前向和后向传递，每次迭代更新模块参数。为了更好地控制这个过程，PyTorch 提供了“钩子”，可以在前向或后向传递过程中执行任意计算，甚至可以根据需要修改传递的完成方式。此功能的一些有用示例包括调试、可视化激活、深入检查梯度等。可以将 Hooks 添加到您自己尚未编写的模块中，这意味着此功能可以应用于第三方或 PyTorch 提供的模块。


 PyTorch 为模块提供了两种类型的钩子：



* **前向钩子** 在前向传递过程中被调用。可以使用 [`register_forward_pre_hook()`](../generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook "torch.nn.Module. register_forward_pre_hook") 和 [`register_forward_hook()`](../generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook "torch.nn.Module.register_forward_hook") 。这些钩子将分别在调用前向函数之前和调用之后调用。或者，可以使用类似的 [`register_module_forward_pre_hook()`](../generated/torch.nn.modules.module.register_module_forward_pre_hook.html#torch.nn.modules.module.register_module_forward_pre_hook "torch.nn.modules.module.register_module_forward_pre_hook") 和 [`register_module_forward_hook()`](../generated/torch.nn.modules.module.register_module_forward_hook.html#torch.nn.modules.module.register_module_forward_hook "torch.nn.modules.module.register_module_forward_hook") 函数。
* **向后钩子** 在向后传递期间被调用。 它们可以通过 [`register_full_backward_pre_hook()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook "torch.nn.Module.register_full_backward_pre_hook") 和 [`register_full_backward_hook()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook "torch.nn.Module.register_full_backward_hook") 安装。 当计算出该模块的后向时，将调用这些钩子。 [`register_full_backward_pre_hook()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook "torch.nn.Module.register_full_backward_pre_hook") 将允许用户访问输出的梯度，而 [`register_full_backward_hook()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook "torch.nn.Module.register_full_backward_hook") 将允许用户访问输入和输出的梯度。 或者，可以使用 [`register_module_full_backward_hook()`](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_hook.html#torch.nn.modules.module.register_module_full_backward_hook "torch.nn.modules.module.register_module_full_backward_hook") 和 [`register_module_full_backward_pre_hook()`](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_pre_hook.html#torch.nn.modules.module.register_module_full_backward_pre_hook "torch.nn.modules.module.register_module_full_backward_pre_hook") 为所有模块全局安装它们。


 所有钩子都允许用户返回将在剩余计算中使用的更新值。因此，这些钩子可用于沿着常规模块向前/向后执行任意代码或修改某些输入/输出，而无需更改模块的`前进()`函数。


 下面是一个演示前向和后向钩子用法的示例：


```
torch.manual_seed(1)

def forward_pre_hook(m, inputs):
  # Allows for examination and modification of the input before the forward pass.
  # Note that inputs are always wrapped in a tuple.
  input = inputs[0]
  return input + 1.

def forward_hook(m, inputs, output):
  # Allows for examination of inputs / outputs and modification of the outputs
  # after the forward pass. Note that inputs are always wrapped in a tuple while outputs
  # are passed as-is.

  # Residual computation a la ResNet.
  return output + inputs[0]

def backward_hook(m, grad_inputs, grad_outputs):
  # Allows for examination of grad_inputs / grad_outputs and modification of
  # grad_inputs used in the rest of the backwards pass. Note that grad_inputs and
  # grad_outputs are always wrapped in tuples.
  new_grad_inputs = [torch.ones_like(gi) * 42. for gi in grad_inputs]
  return new_grad_inputs

# Create sample module & input.
m = nn.Linear(3, 3)
x = torch.randn(2, 3, requires_grad=True)

# ==== Demonstrate forward hooks. ====
# Run input through module before and after adding hooks.
print('output with no forward hooks: {}'.format(m(x)))
: output with no forward hooks: tensor([[-0.5059, -0.8158,  0.2390],
                                        [-0.0043,  0.4724, -0.1714]], grad_fn=<AddmmBackward>)

# Note that the modified input results in a different output.
forward_pre_hook_handle = m.register_forward_pre_hook(forward_pre_hook)
print('output with forward pre hook: {}'.format(m(x)))
: output with forward pre hook: tensor([[-0.5752, -0.7421,  0.4942],
                                        [-0.0736,  0.5461,  0.0838]], grad_fn=<AddmmBackward>)

# Note the modified output.
forward_hook_handle = m.register_forward_hook(forward_hook)
print('output with both forward hooks: {}'.format(m(x)))
: output with both forward hooks: tensor([[-1.0980,  0.6396,  0.4666],
                                          [ 0.3634,  0.6538,  1.0256]], grad_fn=<AddBackward0>)

# Remove hooks; note that the output here matches the output before adding hooks.
forward_pre_hook_handle.remove()
forward_hook_handle.remove()
print('output after removing forward hooks: {}'.format(m(x)))
: output after removing forward hooks: tensor([[-0.5059, -0.8158,  0.2390],
                                               [-0.0043,  0.4724, -0.1714]], grad_fn=<AddmmBackward>)

# ==== Demonstrate backward hooks. ====
m(x).sum().backward()
print('x.grad with no backwards hook: {}'.format(x.grad))
: x.grad with no backwards hook: tensor([[ 0.4497, -0.5046,  0.3146],
                                         [ 0.4497, -0.5046,  0.3146]])

# Clear gradients before running backward pass again.
m.zero_grad()
x.grad.zero_()

m.register_full_backward_hook(backward_hook)
m(x).sum().backward()
print('x.grad with backwards hook: {}'.format(x.grad))
: x.grad with backwards hook: tensor([[42., 42., 42.],
                                      [42., 42., 42.]])

```


## [高级功能](#id10) [¶](#advanced-features "永久链接到此标题")


 PyTorch 还提供了一些旨在与模块配合使用的更高级功能。所有这些功能都可用于自定义编写的模块，但需要注意的是，某些功能可能需要模块符合特定的约束才能得到支持。有关这些功能和相应要求的深入讨论可以在下面的链接中找到。


### [分布式训练](#id11) [¶](#distributed-training "此标题的永久链接")


 PyTorch 中存在各种分布式训练方法，既可用于使用多个 GPU 进行扩展训练，也可用于跨多台机器进行训练。查看[分布式训练概述页面](https://pytorch.org/tutorials/beginner/dist_overview.html)，了解有关如何利用这些的详细信息。


### [分析性能](#id12) [¶](#profiling-performance "永久链接到此标题")


 [PyTorch Profiler](https://pytorch.org/tutorials/beginner/profiler.html) 对于识别模型中的性能瓶颈非常有用。它测量并输出内存使用情况和所用时间的性能特征。


### [通过量化提高性能](#id13) [¶](#improving-performance-with-quantization "永久链接到此标题")


 将量化技术应用于模块可以通过利用比浮点精度更低的位宽来提高性能和内存使用率。查看 PyTorch 提供的各种量化机制 [此处](https://pytorch.org/docs/stable/quantization.html)。


### [通过修剪改善内存使用](#id14) [¶](#improving-memory-usage-with-pruning "永久链接到此标题")


 大型深度学习模型通常过度参数化，导致内存使用率较高。为了解决这个问题，PyTorch 提供了模型修剪机制，这可以帮助减少内存使用，同时保持任务准确性。 [修剪教程](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) 描述了如何利用 PyTorch 提供的修剪技术或根据需要定义自定义修剪技术。


### [参数化](#id15) [¶](#参数化“此标题的永久链接”)


 对于某些应用，在模型训练期间约束参数空间可能是有益的。例如，强制学习参数的正交性可以提高 RNN 的收敛性。 PyTorch 提供了一种应用[参数化的机制](https://pytorch.org/tutorials/intermediate/parametrizations.html)，并进一步允许定义自定义约束。


### [使用 FX 转换模块](#id16) [¶](#transforming-modules-with-fx "永久链接到此标题")


 PyTorch 的 [FX](https://pytorch.org/docs/stable/fx.html) 组件提供了一种通过直接在模块计算图上操作来转换模块的灵活方法。这可用于以编程方式生成或操作各种用例的模块。要探索 FX，请查看这些使用 FX 进行[卷积 + 批标准化融合](https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html) 和 [CPU 性能分析](https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html) 。