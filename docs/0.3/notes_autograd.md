# 自动求导机制

> 译者： [@那伊抹微笑](https://github.com/wangyangting)
> 
> 校对者：[@Twinkle](https://github.com/kemingzeng)

本文将介绍 autograd (自动求导) 如何工作并记录操作. 理解这一切并不是必须的, 但我们建议您熟悉它, 因为它会帮助您编写出更高效, 更简洁的程序, 并且可以帮助您进行调试.

## 反向排除 subgraphs (子图)

每一个变量都有两个标记: `requires_grad` 和 `volatile`. 它们都允许从梯度计算中精细地排除 subgraphs (子图) , 并且可以提高效率.

### `requires_grad`

如果有一个单一的输入操作需要梯度, 则其输出也需要梯度. 相反, 只有当所有输入都不需要梯度时, 输出也才不需要它. 当所有的变量都不需要梯度时, 则反向计算不会在 subgraphs (子图) 中执行.

```py
>>> x = Variable(torch.randn(5, 5))
>>> y = Variable(torch.randn(5, 5))
>>> z = Variable(torch.randn(5, 5), requires_grad=True)
>>> a = x + y
>>> a.requires_grad
False
>>> b = a + z
>>> b.requires_grad
True

```

当您想要冻结模型的一部分, 或者您事先知道不会使用某些参数的梯度时, 这个标记是特别有用的. 例如, 如果要对预先训练的 CNN 进行微优化, 只需在冻结模型的基础上切换 `requires_grad` 标记就可以了, 直到计算到最后一层时, 才会保存中间缓冲区, 其中的 affine transform (仿射变换) 将使用需要梯度的权重, 并且网络的输出也将需要它们.

```py
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# 替换最后一个 fully-connected layer (全连接层)
# 新构造的模块默认情况下参数默认 requires_grad=True
model.fc = nn.Linear(512, 100)

# 仅用于分类器的优化器
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

```

### `volatile`

如果您确定不会调用 `.backward()`, 则推荐在纯粹的 inference mode (推断模式) 中使用 Volatile. 它比任何其它的 autograd (自动求导) 设置更高效 - 它将使用绝对最小量的内存来评估模型. `volatile` 也会确定 `require_grad 为 False`.

Volatile 不同于 `requires_grad` 的标记传播方式. 即使一个操作只有一个单一的 `volatile` 输入, 它的输出也将会是 `volatile` 这样的. Volatility 在整个图中比 `non-requiring` 梯度更容易传播 - 您只需要一个 **单个** `volatile` 叶子即可得到一个 `volatile` 输出, 相对的, 您需要 **所有** 叶子以 `non-requiring` 的方式, 来产生一个 `non-requiring` 的输出. 使用 `volatile` 标记, 您不需要更改模型参数的任何参数, 以便将其用于推断. 创建一个 `volatile` 输入已经足够了, 这种方式也将确保没有中间状态被保存.

```py
>>> regular_input = Variable(torch.randn(1, 3, 227, 227))
>>> volatile_input = Variable(torch.randn(1, 3, 227, 227), volatile=True)
>>> model = torchvision.models.resnet18(pretrained=True)
>>> model(regular_input).requires_grad
True
>>> model(volatile_input).requires_grad
False
>>> model(volatile_input).volatile
True
>>> model(volatile_input).grad_fn is None
True

```

## autograd (自动求导) 如何编码 history (历史信息)

Autograd (自动求导) 是一个反向自动微分的系统. 从概念上来说, autograd (自动求导) 记录一个 graph (图) , 它记录了在执行操作时创建数据的所有操作, 然后给出一个 DAG (有向无环图) , 其中 leaves (叶子) 是输入变量, roots (根) 是输出变量. 通过追踪这个从 roots (根) 到 leaves (叶子) 的 graph (图) , 您可以使用 chain rule (链式规则) 来自动计算梯度.

在其内部, autograd (自动求导) 将这个 graph (图) 形象的表示为 `Function` 对象 (真正的表达式) , 可以通过 `apply()` 方法来计算评估 graph (图) 的结果. 当计算 forwards pass (前向传递) 时, autograd (自动求导) 同时执行所需要的计算, 并且构建一个图以表示计算梯度的函数 ( 每个 `Variable` 类的 `.grad_fn` 属性是该 graph 的入口点) . 当 forwards pass (前向传递) 计算完成时, 我们通过 backwards pass (方向传递) 评估该 graph (图) 来计算梯度.

很重要的一点需要注意, 就是每次迭代都会重新创建一个 graph (图) , 这正是允许使用任意 `Python 控制流语句` 的原因, 这样可以在每次迭代中改变 graph (图) 的整体形状和大小. 在开始训练之前, 您不必编码所有可能的路径 - 您运行的即是您所微分的.

## 变量上的 In-place Operations (就地操作)

在 autograd (自动求导) 中支持 In-place Operations (就地操作) 是一件很难的事情, 我们不鼓励在大多数情况下使用它们. Autograd (自动求导) 主动的 `缓存区释放` 和 `重用` 使其非常高效, 而且 In-place Operations (就地操作) 实际上很少能降低大量的内存使用. 除非您在内存压力很大的情况下操作, 否则您可能永远不需要使用它们.

限制 In-place Operations (就地操作) 适用性的主要原因有两个:

1.  覆盖梯度计算所需的值. 这就是为什么变量不支持 `log_` 的原因. 它的梯度公式需要原始输入, 虽然可以通过计算反向操作可以重新创建它, 但它在数值上是不稳定的, 并且需要额外的工作, 这往往会使这些功能的使用得不偿失.
2.  每一个 in-place Operations (就地操作) 实际上都需要实现重写计算图. Out-of-place (当前通用的) 的版本只是简单的分配新的对象, 并保持旧图的引用, 而 in-place Operations (就地操作) 需要将所有输入的 `creator` 更改为表示此操作的 `Function`. 这可能会很棘手, 特别是如果有许多变量引用相同的存储 (例如通过索引或转置创建的) , 并且如果修改了输入的存储被任何其它的 `Variable` (变量) 所引用, 则 in-place Functions (就地函数) 实际上会抛出错误.

## In-place Operations (就地操作) 的正确性检查

每一个变量都保留有一个 version counter (版本计数器) , 每一次的任何操作被标记为 dirty 时候都会进行递增. 当一个 `Function` 保存了任何用于 backward (方向的) tensor 时, 还会保存其包含变量的 version counter (版本计数器) . 一旦您访问 `self.saved_tensors` 时它将被检查, 如果它大于已保存的值, 则会引起错误.