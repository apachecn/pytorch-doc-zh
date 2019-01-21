# Automatic differentiation package - torch.autograd

`torch.autograd`提供了类和函数用来对任意标量函数进行求导。要想使用自动求导，只需要对已有的代码进行微小的改变。只需要将所有的`tensor`包含进`Variable`对象中即可。

### torch.autograd.backward(variables, grad_variables, retain_variables=False)
Computes the sum of gradients of given variables w.r.t. graph leaves.
给定图的叶子节点`variables`, 计算图中变量的梯度和。
计算图可以通过链式法则求导。如果`variables`中的任何一个`variable`是 非标量(`non-scalar`)的，且`requires_grad=True`。那么此函数需要指定`grad_variables`，它的长度应该和`variables`的长度匹配，里面保存了相关`variable`的梯度(对于不需要`gradient tensor`的`variable`，`None`是可取的)。

此函数累积`leaf variables`计算的梯度。你可能需要在调用此函数之前将`leaf variable`的梯度置零。

参数说明:

- variables (variable 列表) – 被求微分的叶子节点，即 `ys` 。

- grad_variables (`Tensor` 列表) – 对应`variable`的梯度。仅当`variable`不是标量且需要求梯度的时候使用。

- retain_variables (bool) – `True`,计算梯度时所需要的`buffer`在计算完梯度后不会被释放。如果想对一个子图多次求微分的话，需要设置为`True`。

## Variable
### API 兼容性

`Variable API` 几乎和 `Tensor API`一致 (除了一些`in-place`方法，这些`in-place`方法会修改 `required_grad=True`的 `input` 的值)。多数情况下，将`Tensor`替换为`Variable`，代码一样会正常的工作。由于这个原因，我们不会列出`Variable`的所有方法，你可以通过`torch.Tensor`的文档来获取相关知识。

### In-place operations on Variables
在`autograd`中支持`in-place operations`是非常困难的。同时在很多情况下，我们阻止使用`in-place operations`。`Autograd`的贪婪的 释放`buffer`和 复用使得它效率非常高。只有在非常少的情况下，使用`in-place operations`可以降低内存的使用。除非你面临很大的内存压力，否则不要使用`in-place operations`。

### In-place 正确性检查
所有的`Variable`都会记录用在他们身上的 `in-place operations`。如果`pytorch`检测到`variable`在一个`Function`中已经被保存用来`backward`，但是之后它又被`in-place operations`修改。当这种情况发生时，在`backward`的时候，`pytorch`就会报错。这种机制保证了，如果你用了`in-place operations`，但是在`backward`过程中没有报错，那么梯度的计算就是正确的。

### class torch.autograd.Variable [source]

包装一个`Tensor`,并记录用在它身上的`operations`。

`Variable`是`Tensor`对象的一个`thin wrapper`，它同时保存着`Variable`的梯度和创建这个`Variable`的`Function`的引用。这个引用可以用来追溯创建这个`Variable`的整条链。如果`Variable`是被用户所创建的，那么它的`creator`是`None`，我们称这种对象为 `leaf Variables`。

由于`autograd`只支持标量值的反向求导(即：`y`是标量)，梯度的大小总是和数据的大小匹配。同时，仅仅给`leaf variables`分配梯度，其他`Variable`的梯度总是为0.

**`变量：`**

- data – 包含的`Tensor`

- grad – 保存着`Variable`的梯度。这个属性是懒分配的，且不能被重新分配。

- requires_grad – 布尔值，指示这个`Variable`是否是被一个包含`Variable`的子图创建的。更多细节请看`Excluding subgraphs from backward`。只能改变`leaf variable`的这个标签。

- volatile – 布尔值，指示这个`Variable`是否被用于推断模式(即，不保存历史信息)。更多细节请看`Excluding subgraphs from backward`。只能改变`leaf variable`的这个标签。

- creator – 创建这个`Variable`的`Function`，对于`leaf variable`，这个属性为`None`。只读属性。

**`属性:`**

- data (any tensor class) – 被包含的`Tensor`

- requires_grad (bool) – `requires_grad`标记. 只能通过`keyword`传入.

- volatile (bool) – `volatile`标记. 只能通过`keyword`传入.

#### backward(gradient=None, retain_variables=False)[source]

当前`Variable`对`leaf variable`求偏导。

计算图可以通过链式法则求导。如果`Variable`是 非标量(`non-scalar`)的，且`requires_grad=True`。那么此函数需要指定`gradient`，它的形状应该和`Variable`的长度匹配，里面保存了`Variable`的梯度。

此函数累积`leaf variable`的梯度。你可能需要在调用此函数之前将`Variable`的梯度置零。

**`参数:`**

- gradient (Tensor) – 其他函数对于此`Variable`的导数。仅当`Variable`不是标量的时候使用，类型和位形状应该和`self.data`一致。
- retain_variables (bool) – `True`, 计算梯度所必要的`buffer`在经历过一次`backward`过程后不会被释放。如果你想多次计算某个子图的梯度的时候，设置为`True`。在某些情况下，使用`autograd.backward()`效率更高。

#### detach()[source]
Returns a new Variable, detached from the current graph.
返回一个新的`Variable`，从当前图中分离下来的。

返回的`Variable` `requires_grad=False`，如果输入 `volatile=True`，那么返回的`Variable` `volatile=True`。

**`注意：`**

返回的`Variable`和原始的`Variable`公用同一个`data tensor`。`in-place`修改会在两个`Variable`上同时体现(因为它们共享`data tensor`)，可能会导致错误。

#### detach_()[source]

将一个`Variable`从创建它的图中分离，并把它设置成`leaf variable`。

#### register_hook(hook)[source]

注册一个`backward`钩子。

每次`gradients`被计算的时候，这个`hook`都被调用。`hook`应该拥有以下签名：

`hook(grad) -> Variable or None`

`hook`不应该修改它的输入，但是它可以选择性的返回一个替代当前梯度的新梯度。

这个函数返回一个 句柄(`handle`)。它有一个方法 `handle.remove()`，可以用这个方法将`hook`从`module`移除。

Example
```python
v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
h = v.register_hook(lambda grad: grad * 2)  # double the gradient
v.backward(torch.Tensor([1, 1, 1]))
#先计算原始梯度，再进hook，获得一个新梯度。
print(v.grad.data)

 2
 2
 2
[torch.FloatTensor of size 3]
>>> h.remove()  # removes the hook
```
```python
def w_hook(grad):
    print("hello")
    return None
w1 = Variable(torch.FloatTensor([1, 1, 1]),requires_grad=True)

w1.register_hook(w_hook) # 如果hook返回的是None的话，那么梯度还是原来计算的梯度。

w1.backward(gradient=torch.FloatTensor([1, 1, 1]))
print(w1.grad)
```
```
hello
Variable containing:
 1
 1
 1
[torch.FloatTensor of size 3]
```

#### reinforce(reward)[source]

注册一个奖励，这个奖励是由一个随机过程得到的。

微分一个随机节点需要提供一个奖励值。如果你的计算图中包含随机 `operations`，你需要在他们的输出上调用这个函数。否则的话，会报错。

**`参数:`**

- reward (Tensor) – 每个元素的reward。必须和`Varaible`形状相同，并在同一个设备上。

### class torch.autograd.Function[source]
Records operation history and defines formulas for differentiating ops.
记录`operation`的历史，定义微分公式。
每个执行在`Varaibles`上的`operation`都会创建一个`Function`对象，这个`Function`对象执行计算工作，同时记录下来。这个历史以有向无环图的形式保存下来，有向图的节点为`functions`，有向图的边代表数据依赖关系(`input<-output`)。之后，当`backward`被调用的时候，计算图以拓扑顺序处理，通过调用每个`Function`对象的`backward()`，同时将返回的梯度传递给下一个`Function`。

通常情况下，用户能和`Functions`交互的唯一方法就是创建`Function`的子类，定义新的`operation`。这是扩展`torch.autograd`的推荐方法。

由于`Function`逻辑在很多脚本上都是热点，所有我们把几乎所有的`Function`都使用`C`实现，通过这种策略保证框架的开销是最小的。

每个`Function`只被使用一次(在forward过程中)。

**`变量:`**

- saved_tensors – 调用`forward()`时需要被保存的 `Tensors`的 `tuple`。

- needs_input_grad – 长度为 输入数量的 布尔值组成的 `tuple`。指示给定的`input`是否需要梯度。这个被用来优化用于`backward`过程中的`buffer`，忽略`backward`中的梯度计算。

- num_inputs – `forward` 的输入参数数量。

- num_outputs – `forward`返回的`Tensor`数量。

- requires_grad – 布尔值。指示`backward`以后会不会被调用。

- previous_functions – 长度为 `num_inputs`的 Tuple of (int, Function) pairs。`Tuple`中的每单元保存着创建 `input`的`Function`的引用，和索引。
#### backward(* grad_output)[source]

定义了`operation`的微分公式。

所有的`Function`子类都应该重写这个方法。

所有的参数都是`Tensor`。他必须接收和`forward`的输出 相同个数的参数。而且它需要返回和`forward`的输入参数相同个数的`Tensor`。
即：`backward`的输入参数是 此`operation`的输出的值的梯度。`backward`的返回值是此`operation`输入值的梯度。

#### forward(* input)[source]

执行`operation`。

所有的`Function`子类都需要重写这个方法。

可以接收和返回任意个数 `tensors`

#### mark_dirty(* args)[source]

将输入的 `tensors` 标记为被`in-place operation`修改过。

这个方法应当至多调用一次，仅仅用在 `forward`方法里，而且`mark_dirty`的实参只能是`forward`的实参。

每个在`forward`方法中被`in-place operations`修改的`tensor`都应该传递给这个方法。这样，可以保证检查的正确性。这个方法在`tensor`修改前后调用都可以。

#### mark_non_differentiable(* args)[source]
将输出标记为不可微。

这个方法至多只能被调用一次，只能在`forward`中调用，而且实参只能是`forward`的返回值。

这个方法会将输出标记成不可微，会增加`backward`过程中的效率。在`backward`中，你依旧需要接收`forward`输出值的梯度，但是这些梯度一直是`None`。

This is used e.g. for indices returned from a max Function.

#### mark_shared_storage(* pairs)[source]
将给定的`tensors pairs`标记为共享存储空间。

这个方法至多只能被调用一次，只能在`forward`中调用，而且所有的实参必须是`(input, output)`对。

如果一些 `inputs` 和 `outputs` 是共享存储空间的，所有的这样的 `(input, output)`对都应该传给这个函数，保证 `in-place operations` 检查的正确性。唯一的特例就是，当 `output`和`input`是同一个`tensor`(`in-place operations`的输入和输出)。这种情况下，就没必要指定它们之间的依赖关系，因为这个很容易就能推断出来。

这个函数在很多时候都用不到。主要是用在 索引 和 转置 这类的 `op` 中。

#### save_for_backward(* tensors)[source]

将传入的 `tensor` 保存起来，留着`backward`的时候用。

这个方法至多只能被调用一次，只能在`forward`中调用。

之后，被保存的`tensors`可以通过 `saved_tensors`属性获取。在返回这些`tensors`之前，`pytorch`做了一些检查，保证这些`tensor`没有被`in-place operations`修改过。

实参可以是`None`。
