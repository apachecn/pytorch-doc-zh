# torch.nn.init

```python
torch.nn.init.calculate_gain(nonlinearity,param=None)
```

对于给定的非线性函数，返回推荐的增益值。这些值如下所示：


| nonlinearity | gain                         |
| ------------ | ---------------------------- |
| linear       | 1                            |
| conv{1,2,3}d | 1                            |
| sigmoid      | 1                            |
| tanh         | 5/3                          |
| relu         | sqrt(2)                      |
| leaky_relu   | sqrt(2/(1+negative_slope^2)) |

**参数：**

- **nonlinearity** - 非线性函数(`nn.functional`名称）
- **param** - 非线性函数的可选参数

**例子：**

```python
>>> gain = nn.init.gain('leaky_relu')
```

```python
torch.nn.init.uniform(tensor, a=0, b=1)
```

从均匀分布U(a, b)中生成值，填充输入的张量或变量

**参数：**

- **tensor** - n维的torch.Tensor
- **a** - 均匀分布的下界
- **b** - 均匀分布的上界

**例子**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.uniform(w)
```

```python
torch.nn.init.normal(tensor, mean=0, std=1)
```

从给定均值和标准差的正态分布N(mean, std)中生成值，填充输入的张量或变量

**参数：**

- **tensor** – n维的torch.Tensor
- **mean** – 正态分布的均值
- **std** – 正态分布的标准差

**例子**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.normal(w)
```

```python
torch.nn.init.constant(tensor, val)
```

用*val*的值填充输入的张量或变量

**参数：**

- **tensor** – n维的torch.Tensor或autograd.Variable
- **val** – 用来填充张量的值

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.constant(w)
```

```python
torch.nn.init.eye(tensor)
```

用单位矩阵来填充2维输入张量或变量。在线性层尽可能多的保存输入特性。

**参数：**

- **tensor** – 2维的torch.Tensor或autograd.Variable

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.eye(w)
```

```python
torch.nn.init.dirac(tensor)
```

用Dirac $\delta$ 函数来填充{3, 4, 5}维输入张量或变量。在卷积层尽可能多的保存输入通道特性。

**参数：**

- **tensor** – {3, 4, 5}维的torch.Tensor或autograd.Variable

**例子：**

```python
>>> w = torch.Tensor(3, 16, 5, 5)
>>> nn.init.dirac(w)
```

```python
torch.nn.init.xavier_uniform(tensor, gain=1)
```

根据Glorot, X.和Bengio, Y.在“Understanding the difficulty of training deep feedforward neural networks”中描述的方法，用一个均匀分布生成值，填充输入的张量或变量。结果张量中的值采样自U(-a, a)，其中a= gain * sqrt( 2/(fan_in + fan_out))* sqrt(3). 该方法也被称为Glorot initialisation

**参数：**

- **tensor** – n维的torch.Tensor
- **gain** - 可选的缩放因子

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.xavier_uniform(w, gain=math.sqrt(2.0))
```

```python
torch.nn.init.xavier_normal(tensor, gain=1)
```

根据Glorot, X.和Bengio, Y. 于2010年在“Understanding the difficulty of training deep feedforward neural networks”中描述的方法，用一个正态分布生成值，填充输入的张量或变量。结果张量中的值采样自均值为0，标准差为gain * sqrt(2/(fan_in + fan_out))的正态分布。也被称为Glorot initialisation.

**参数：**

- **tensor** – n维的torch.Tensor
- **gain** - 可选的缩放因子

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.xavier_normal(w)
```

```python
torch.nn.init.kaiming_uniform(tensor, a=0, mode='fan_in')
```

根据He, K等人于2015年在“Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification”中描述的方法，用一个均匀分布生成值，填充输入的张量或变量。结果张量中的值采样自U(-bound, bound)，其中bound = sqrt(2/((1 + a^2) * fan_in)) * sqrt(3)。也被称为He initialisation.

**参数：**

- **tensor** – n维的torch.Tensor或autograd.Variable
- **a** -这层之后使用的rectifier的斜率系数(ReLU的默认值为0）
- **mode** -可以为“fan_in”(默认）或“fan_out”。“fan_in”保留前向传播时权值方差的量级，“fan_out”保留反向传播时的量级。

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.kaiming_uniform(w, mode='fan_in')
```

```python
torch.nn.init.kaiming_normal(tensor, a=0, mode='fan_in')
```

根据He, K等人在“Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification”中描述的方法，用一个正态分布生成值，填充输入的张量或变量。结果张量中的值采样自均值为0，标准差为sqrt(2/((1 + a^2) * fan_in))的正态分布。

**参数：**

- **tensor** – n维的torch.Tensor或 autograd.Variable
- **a** -这层之后使用的rectifier的斜率系数(ReLU的默认值为0）
- **mode** -可以为“fan_in”(默认）或“fan_out”。“fan_in”保留前向传播时权值方差的量级，“fan_out”保留反向传播时的量级。

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.kaiming_normal(w, mode='fan_out')
```

```python
torch.nn.init.orthogonal(tensor, gain=1)
```

用(半）正交矩阵填充输入的张量或变量。输入张量必须至少是2维的，对于更高维度的张量，超出的维度会被展平，视作行等于第一个维度，列等于稀疏矩阵乘积的2维表示。其中非零元素生成自均值为0，标准差为std的正态分布。

参考：Saxe, A等人(2013)的“Exact solutions to the nonlinear dynamics of learning in deep linear neural networks”

**参数：**

- **tensor** – n维的torch.Tensor或 autograd.Variable，其中n>=2
- **gain** -可选

**例子：**

```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.orthogonal(w)
```

```python
torch.nn.init.sparse(tensor, sparsity, std=0.01)
```

将2维的输入张量或变量当做稀疏矩阵填充，其中非零元素根据一个均值为0，标准差为std的正态分布生成。
参考Martens, J.(2010)的 “Deep learning via Hessian-free optimization”. 

**参数：**

- **tensor** – n维的torch.Tensor或autograd.Variable
- **sparsity** - 每列中需要被设置成零的元素比例
- **std** - 用于生成非零值的正态分布的标准差

**例子：**
```python
>>> w = torch.Tensor(3, 5)
>>> nn.init.sparse(w, sparsity=0.1)
```