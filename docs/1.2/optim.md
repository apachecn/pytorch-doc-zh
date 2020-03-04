# torch.optim

> 译者：[ApacheCN](https://github.com/apachecn)

是一个实现各种优化算法的包。已经支持最常用的方法，并且界面足够通用，因此将来可以轻松集成更复杂的方法。

## 如何使用优化器

要使用，您必须构造一个优化器对象，该对象将保持当前状态并将根据计算的渐变更新参数。

### 构建它

要构造一个你必须给它一个包含参数的迭代(所有应该是`Variable` s）来优化。然后，您可以指定特定于优化程序的选项，例如学习率，重量衰减等。

注意

如果您需要通过`.cuda()`将模型移动到GPU，请在为其构建优化器之前执行此操作。 `.cuda()`之后的模型参数与调用之前的参数不同。

通常，在构造和使用优化程序时，应确保优化参数位于一致的位置。

例：

```
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr = 0.0001)

```

### 每个参数选项

s还支持指定每个参数选项。要做到这一点，不要传递一个可迭代的`Variable`，而是传递一个可迭代的s。它们中的每一个都将定义一个单独的参数组，并且应包含`params`键，其中包含属于它的参数列表。其他键应与优化程序接受的关键字参数匹配，并将用作此组的优化选项。

Note

您仍然可以将选项作为关键字参数传递。它们将在未覆盖它们的组中用作默认值。当您只想改变单个选项，同时保持参数组之间的所有其他选项保持一致时，这非常有用。

例如，当想要指定每层学习速率时，这非常有用：

```
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)

```

这意味着`model.base`的参数将使用`1e-2`的默认学习速率，`model.classifier`的参数将使用`1e-3`的学习速率，`0.9`的动量将用于所有参数

### 采取优化步骤

所有优化器都实现了一个更新参数的方法。它可以以两种方式使用：

#### `optimizer.step()`

这是大多数优化器支持的简化版本。一旦使用例如计算梯度，就可以调用该函数。 `backward()`。

Example:

```
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

```

#### `optimizer.step(closure)`

一些优化算法，例如Conjugate Gradient和LBFGS需要多次重新评估函数，因此您必须传入一个允许它们重新计算模型的闭包。闭合应清除梯度，计算损失并返回。

Example:

```
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)

```

## 算法

```
class torch.optim.Optimizer(params, defaults)
```

所有优化器的基类。

警告

需要将参数指定为具有在运行之间一致的确定性排序的集合。不满足这些属性的对象的示例是字典值的集合和迭代器。

参数：

*   **params**  (_iterable_ ) - s或s的可迭代。指定应优化的张量。
*   **默认值** - (dict）：包含优化选项默认值的dict(当参数组未指定它们时使用）。

```
add_param_group(param_group)
```

将参数组添加到s `param_groups`。

当微调预先训练的网络时，这可以是有用的，因为冻结层可以被训练并且被添加到训练进展中。

Parameters:

*   **param_group** (） - 指定应该与组一起优化的张量
*   **优化选项。** (_特异性_） -

```
load_state_dict(state_dict)
```

加载优化器状态。

| 参数： | **state_dict** (） - 优化器状态。应该是从调用返回的对象。 |
| --- | --- |

```
state_dict()
```

以...格式返回优化程序的状态。

它包含两个条目：

*   ```
    state - a dict holding current optimization state. Its content
    ```

    优化器类之间有所不同。

*   param_groups - 包含所有参数组的dict

```
step(closure)
```

执行单个优化步骤(参数更新）。

| Parameters: | **闭包**(_可调用_） - 一个重新评估模型并返回损失的闭包。大多数优化器都是可选的。 |
| --- | --- |

```
zero_grad()
```

清除所有优化s的渐变。

```
class torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
```

实现Adadelta算法。

已在 [ADADELTA中提出：自适应学习速率方法](https://arxiv.org/abs/1212.5701)。

Parameters:

*   **params**  (_iterable_ ) - 可迭代参数以优化或决定参数组
*   **rho** (_，_ _可选_） - 用于计算平方梯度运行平均值的系数(默认值：0.9）
*   **eps** (_，_ _可选_） - 术语加入分母以提高数值稳定性(默认值：1e-6）
*   **lr** (_，_ _可选_） - 在应用于参数之前缩放增量的系数(默认值：1.0）
*   **weight_decay** (_，_ _可选_） - 体重衰减(L2惩罚）(默认值：0）

```
step(closure=None)
```

执行单个优化步骤。

| Parameters: | **关闭**(_可调用_ _，_ _可选_） - 一个重新评估模型并返回损失的闭包。 |
| --- | --- |

```
class torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
```

实现Adagrad算法。

已经在[自适应子梯度方法中提出了在线学习和随机优化](http://jmlr.org/papers/v12/duchi11a.html)。

Parameters:

*   **params**  (_iterable_ ) - 可迭代参数以优化或决定参数组
*   **lr** (_，_ _可选_） - 学习率(默认值：1e-2）
*   **lr_decay** (_，_ _可选_） - 学习率衰减(默认值：0）
*   **weight_decay** (_，_ _可选_） - 体重衰减(L2惩罚）(默认值：0）

```
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

实现Adam算法。

已在 [Adam中提出：随机优化方法](https://arxiv.org/abs/1412.6980)。

Parameters:

*   **params**  (_iterable_ ) - 可迭代参数以优化或决定参数组
*   **lr** (_，_ _可选_） - 学习率(默认值：1e-3）
*   **beta** (_元组_ _ [，_ _] __，_ _任选_） - 用于计算运行平均值的系数渐变及其方形(默认值：(0.9,0.999））
*   **eps** (_，_ _可选_） - 术语加入分母以提高数值稳定性(默认值：1e-8）
*   **weight_decay** (_，_ _可选_） - 体重衰减(L2惩罚）(默认值：0）
*   **amsgrad** (_布尔_ _，_ _可选_） - 是否使用该算法的AMSGrad变体[关于亚当及其后的收敛](https://openreview.net/forum?id=ryQu7f-RZ)(默认值：False）

```
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```
class torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
```

实现适用于稀疏张量的懒惰版Adam算法。

在此变体中，只有渐变中显示的时刻才会更新，并且只有渐变的那些部分才会应用于参数。

Parameters:

*   **params**  (_iterable_ ) - 可迭代参数以优化或决定参数组
*   **lr** (_，_ _可选_） - 学习率(默认值：1e-3）
*   **beta** (_元组_ _ [，_ _] __，_ _任选_） - 用于计算运行平均值的系数渐变及其方形(默认值：(0.9,0.999））
*   **eps** (_，_ _可选_） - 术语加入分母以提高数值稳定性(默认值：1e-8）

```
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```
class torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```

实现Adamax算法(基于无穷大规范的Adam的变体）。

It has been proposed in [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).

Parameters:

*   **params**  (_iterable_ ) - 可迭代参数以优化或决定参数组
*   **lr** (_，_ _可选_） - 学习率(默认值：2e-3）
*   **beta** (_元组_ _ [，_ _] __，_ _任选_） - 用于计算运行平均值的系数渐变和它的正方形
*   **eps** (_，_ _可选_） - 术语加入分母以提高数值稳定性(默认值：1e-8）
*   **weight_decay** (_，_ _可选_） - 体重衰减(L2惩罚）(默认值：0）

```
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```
class torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
```

实现平均随机梯度下降。

已经在[中通过平均](http://dl.acm.org/citation.cfm?id=131098)来加速随机近似。

Parameters:

*   **params**  (_iterable_ ) - 可迭代参数以优化或决定参数组
*   **lr** (_，_ _可选_） - 学习率(默认值：1e-2）
*   **lambd** (_，_ _可选_） - 衰变期限(默认值：1e-4）
*   **alpha** (_，_ _可选_） - eta更新的权力(默认值：0.75）
*   **t0** (_，_ _可选_） - 开始平均的点(默认值：1e6）
*   **weight_decay** (_，_ _可选_） - 体重衰减(L2惩罚）(默认值：0）

```
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```
class torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
```

实现L-BFGS算法。

Warning

此优化器不支持每个参数选项和参数组(只能有一个）。

Warning

现在所有参数都必须在一台设备上。这将在未来得到改善。

Note

这是一个内存密集型优化器(它需要额外的`param_bytes * (history_size + 1)`字节）。如果它不适合内存尝试减少历史记录大小，或使用不同的算法。

Parameters:

*   **lr** (） - 学习率(默认值：1）
*   **max_iter** (） - 每个优化步骤的最大迭代次数(默认值：20）
*   **max_eval** (） - 每个优化步骤的最大函数评估数(默认值：max_iter * 1.25）。
*   **tolerance_grad** (） - 一阶最优性的终止容差(默认值：1e-5）。
*   **tolerance_change** (） - 功能值/参数更改的终止容差(默认值：1e-9）。
*   **history_size** (） - 更新历史记录大小(默认值：100）。

```
step(closure)
```

Performs a single optimization step.

| Parameters: | **闭包**(_可调用_） - 一个重新评估模型并返回损失的闭包。 |
| --- | --- |

```
class torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```

实现RMSprop算法。

G. Hinton在他的[课程](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)中提出的建议。

中心版本首先出现在[生成具有回归神经网络的序列](https://arxiv.org/pdf/1308.0850v5.pdf)中。

Parameters:

*   **params**  (_iterable_ ) - 可迭代参数以优化或决定参数组
*   **lr** (_，_ _可选_） - 学习率(默认值：1e-2）
*   **动量**(_，_ _可选_） - 动量因子(默认值：0）
*   **alpha** (_，_ _可选_） - 平滑常数(默认值：0.99）
*   **eps** (_，_ _可选_） - 术语加入分母以提高数值稳定性(默认值：1e-8）
*   **居中**(_，_ _可选_） - 如果`True`计算居中的RMSProp，则通过估计其方差对梯度进行归一化
*   **weight_decay** (_，_ _可选_） - 体重衰减(L2惩罚）(默认值：0）

```
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```
class torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
```

实现弹性反向传播算法。

Parameters:

*   **params**  (_iterable_ ) - 可迭代参数以优化或决定参数组
*   **lr** (_，_ _可选_） - 学习率(默认值：1e-2）
*   **etas**  (_Tuple_ _ [，_ _] __，_ _任选_） - 对(etaminus，etaplis） ，这是乘法增加和减少因子(默认值：(0.5,1.2））
*   **step_sizes**  (_Tuple_ _ [，_ _] __，_ _任选_） - 一对最小和最大允许步长(默认值：(1e-6,50））

```
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```
class torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

实现随机梯度下降(可选择带动量）。

Nesterov动量是基于[关于初始化和动量在深度学习](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)中的重要性的公式。

Parameters:

*   **params**  (_iterable_ ) - 可迭代参数以优化或决定参数组
*   **lr** (） - 学习率
*   **动量**(_，_ _可选_） - 动量因子(默认值：0）
*   **weight_decay** (_，_ _可选_） - 体重衰减(L2惩罚）(默认值：0）
*   **阻尼**(_，_ _可选_） - 抑制动量(默认值：0）
*   **nesterov** (_，_ _可选_） - 启用Nesterov动量(默认值：False）

例

```
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()

```

Note

使用Momentum / Nesterov实施SGD与Sutskever等有所不同。人。和其他一些框架中的实现。

考虑到Momentum的具体情况，更新可以写成

其中p，g，v分别表示参数，梯度，速度和动量。

这与Sutskever等人形成鲜明对比。人。和其他采用表格更新的框架

Nesterov版本经过类似修改。

```
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

## 如何调整学习率

`torch.optim.lr_scheduler`提供了几种根据时期数调整学习率的方法。允许基于一些验证测量来降低动态学习速率。

```
class torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
```

将每个参数组的学习速率设置为给定函数的初始lr倍。当last_epoch = -1时，将初始lr设置为lr。

Parameters:

*   **优化器**(） - 包装优化器。
*   **lr_lambda** (_函数_ _或_） - 一个函数，它计算给定整数参数时期的乘法因子，或这些函数的列表，优化器中每个组一个.param_groups。
*   **last_epoch** (） - 最后一个纪元的索引。默认值：-1。

Example

```
>>> # Assuming optimizer has two groups.
>>> lambda1 = lambda epoch: epoch // 30
>>> lambda2 = lambda epoch: 0.95 ** epoch
>>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)

```

```
load_state_dict(state_dict)
```

加载调度程序状态。

| Parameters: | **state_dict** (） - 调度程序状态。应该是从调用返回的对象。 |
| --- | --- |

```
state_dict()
```

将调度程序的状态作为a返回。

它包含自我中每个变量的条目。 **dict** 不是优化器。学习率lambda函数只有在它们是可调用对象时才会被保存，而不是它们是函数或lambdas。

```
class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

将每个参数组的学习速率设置为每个step_size epochs由gamma衰减的初始lr。当last_epoch = -1时，将初始lr设置为lr。

Parameters:

*   **优化器**(） - 包装优化器。
*   **step_size** (） - 学习率衰减的时期。
*   **gamma** (） - 学习率衰减的乘法因子。默认值：0.1。
*   **last_epoch** (） - 最后一个纪元的索引。默认值：-1。

Example

```
>>> # Assuming optimizer uses lr = 0.05 for all groups
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 60
>>> # lr = 0.0005   if 60 <= epoch < 90
>>> # ...
>>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)

```

```
class torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

一旦纪元数达到其中一个里程碑，将每个参数组的学习速率设置为由伽玛衰减的初始lr。当last_epoch = -1时，将初始lr设置为lr。

Parameters:

*   **优化器**(） - 包装优化器。
*   **里程碑**(） - 时代指数列表。必须增加。
*   **gamma** (） - 学习率衰减的乘法因子。默认值：0.1。
*   **last_epoch** (） - 最后一个纪元的索引。默认值：-1。

Example

```
>>> # Assuming optimizer uses lr = 0.05 for all groups
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 80
>>> # lr = 0.0005   if epoch >= 80
>>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)

```

```
class torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
```

将每个参数组的学习率设置为每个时期由伽玛衰减的初始lr。当last_epoch = -1时，将初始lr设置为lr。

Parameters:

*   **优化器**(） - 包装优化器。
*   **gamma** (） - 学习率衰减的乘法因子。
*   **last_epoch** (） - 最后一个纪元的索引。默认值：-1。

```
class torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
```

使用余弦退火计划设置每个参数组的学习速率，其中设置为初始lr，并且是自SGDR上次重启以来的纪元数：

当last_epoch = -1时，将初始lr设置为lr。

已在 [SGDR中提出：具有暖启动的随机梯度下降](https://arxiv.org/abs/1608.03983)。请注意，这仅实现SGDR的余弦退火部分，而不是重启。

Parameters:

*   **优化器**(） - 包装优化器。
*   **T_max** (） - 最大迭代次数。
*   **eta_min** (） - 最低学习率。默认值：0。
*   **last_epoch** (） - 最后一个纪元的索引。默认值：-1。

```
class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
```

当指标停止改进时降低学习率。一旦学习停滞，模型通常会将学习率降低2-10倍。该调度程序读取度量数量，如果“耐心”数量的时期没有看到改善，则学习速率降低。

Parameters:

*   **优化器**(） - 包装优化器。
*   **模式**(） - `min`，`max`之一。在`min`模式下，当监控量停止下降时，lr将减少;在`max`模式下，当监控量停止增加时，它将减少。默认值：'min'。
*   **factor** (） - 学习率降低的因素。 new_lr = lr * factor。默认值：0.1。
*   **耐心**(） - 没有改善的时期数，之后学习率会降低。例如，如果`patience = 2`，那么我们将忽略没有改进的前2个时期，并且如果损失仍然没有改善那么将仅在第3个时期之后减少LR。默认值：10。
*   **verbose** (） - 如果`True`，每次更新都会向stdout输出一条消息。默认值：`False`。
*   **阈值**(） - 测量新最佳值的阈值，仅关注重大变化。默认值：1e-4。
*   **threshold_mode** (） - `rel`，`abs`之一。在`rel`模式下，dynamic_threshold ='max'模式下的最佳*(1 +阈值）或`min`模式下的最佳*(1 - 阈值）。在`abs`模式下，dynamic_threshold = `max`模式下的最佳+阈值或`min`模式下的最佳阈值。默认值：'rel'。
*   **冷却时间**(） - 在减少lr之后恢复正常操作之前要等待的时期数。默认值：0。
*   **min_lr** (_或_） - 标量或标量列表。所有参数组或每组的学习率的下限。默认值：0。
*   **eps** (） - 应用于lr的最小衰减。如果新旧lr之间的差异小于eps，则忽略更新。默认值：1e-8。

Example

```
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = ReduceLROnPlateau(optimizer, 'min')
>>> for epoch in range(10):
>>>     train(...)
>>>     val_loss = validate(...)
>>>     # Note that step should be called after validate()
>>>     scheduler.step(val_loss)

```