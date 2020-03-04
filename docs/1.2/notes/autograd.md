

# 自动求导机制  

> 译者：[冯宝宝](https://github.com/PEGASUS1993)
>
> 校验：[AlexJakin](https://github.com/AlexJakin)

本说明将概述autograd(自动求导）如何工作并记录每一步操作。了解这些并不是绝对必要的，但我们建议您熟悉它，因为它将帮助你编写更高效，更清晰的程序，并可以帮助您进行调试。  

## 反向排除子图

每个张量都有一个标志：`requires_grad`，允许从梯度计算中细致地排除子图，并可以提高效率。    

### `requires_grad`   

只要有单个输入进行梯度计算操作，则其输出也需要梯度计算。相反，只有当所有输入都不需要计算梯度时，输出才不需要梯度计算。如果其中所有的张量都不需要进行梯度计算，后向计算不会在子图中执行。   


```py
>>> x = torch.randn(5, 5)  # requires_grad=False by default
>>> y = torch.randn(5, 5)  # requires_grad=False by default
>>> z = torch.randn((5, 5), requires_grad=True)
>>> a = x + y
>>> a.requires_grad
False
>>> b = a + z
>>> b.requires_grad
True

```  

当你想要冻结部分模型或者事先知道不会使用某些参数的梯度时，这个`requires_grad`标志非常有用。例如，如果要微调预训练的CNN，只需在冻结的基础中切换`requires_grad`标志就够了，并且直到计算到达最后一层，才会保存中间缓冲区，，其中仿射变换将使用所需要梯度的权重 ，网络的输出也需要它们。  


```py
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
model.fc = nn.Linear(512, 100)

# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

```  

## 自动求导是如何记录编码历史的   

自动求导是反向自动分化系统。从概念上讲，自动求导会记录一个图形，记录在执行操作时创建数据的所有操作，为您提供有向无环图，其叶子是输入张量，根节点是输出张量。通过从根到叶跟踪此图，您可以使用链法则自动计算梯度。   

在内部，autograd将此图表示为`Function`对象(实际表达式）的图形，可以用来计算评估图形的结果。 当计算前向传播时，自动求导同时执行所请求的计算并建立表示计算梯度的函数的图形(每个`torch.Tensor`的`.grad_fn`属性是该图的入口点）。当前向传播完成时，我们在后向传播中评估该图以计算梯度。

需要注意的一点是，在每次迭代时都会从头开始重新创建计算图，这正是允许使用任意Python控制流语句的原因，它可以在每次迭代时更改图形的整体形状和大小。 在开始训练之前，不必编码所有可能的路径 - 您运行的是您所区分的部分。  

## 使用autograd进行`in-place`操作  

在autograd中支持`in-place`操作是一件很难的事情，大多数情况下，我们不鼓励使用它们。Autograd积极的缓冲区释放和重用使其非常高效，实际上在`in-place`操作会大幅降低内存使用量的情况也非常少。除非在巨大的内存压力下运行，否则你可能永远不需要使用它们。  

限制`in-place`操作适用性的主要原因有两个：  

1. 这个操作可能会覆盖梯度计算所需的值。  
2. 实际上，每个`in-place`操作需要重写计算图。`out-of-place`版本只是分配新对象并保留对旧图的引用，而`in-place`操作则需要将所有输入的creator更改为表示此操作的`Function`。这就比较麻烦，特别是如果有许多变量引用同一存储(例如通过索引或转置创建的），并且如果被修改输入的存储被任何其他张量引用，这样的话，`in-place`函数会抛出错误。 

## In-place正确性检查  

每一个张量都有一个版本计算器，每次在任何操作中标记都会递增。 当`Function`保存任何用于后向传播的张量时，也会保存包含张量的版本计数器。一旦访问`self.saved_tensors`后，它将被检查，如果它大于保存的值，则会引发错误。这可以确保如果您使用`in-place`函数而没有看到任何错误，则可以确保计算出的梯度是正确的。
