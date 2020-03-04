# torch.nn.init

> 译者：[GeneZC](https://github.com/GeneZC)

```py
torch.nn.init.calculate_gain(nonlinearity, param=None)
```

返回给定非线性函数的推荐的增益值。对应关系如下表：

| 非线性函数 | 增益 |
| --- | --- |
| Linear / Identity | ![](http://latex.codecogs.com/gif.latex?1) |
| Conv{1,2,3}D | ![](http://latex.codecogs.com/gif.latex?1) |
| Sigmoid | ![](http://latex.codecogs.com/gif.latex?1) |
| Tanh | ![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B5%7D%7B3%7D) |
| ReLU | ![](http://latex.codecogs.com/gif.latex?%5Csqrt%7B2%7D) |
| Leaky Relu | ![](http://latex.codecogs.com/gif.latex?%5Csqrt%7B%5Cfrac%7B2%7D%7B1%20%2B%20%5Ctext%7Bnegative%5C_slope%7D%5E2%7D%7D) |

 
参数：

*   **nonlinearity** – 非线性函数 (`nn.functional` 中的名字)
*   **param** – 对应非线性函数的可选参数


例子

```py
>>> gain = nn.init.calculate_gain('leaky_relu')

```

```py
torch.nn.init.uniform_(tensor, a=0, b=1)
```

用均匀分布 ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BU%7D(a%2C%20b)) 初始化输入 `Tensor`。

 
参数： 

*   **tensor** – n 维 `torch.Tensor`
*   **a** – 均匀分布的下界
*   **b** – 均匀分布的上界


例子

```py
>>> w = torch.empty(3, 5)
>>> nn.init.uniform_(w)

```

```py
torch.nn.init.normal_(tensor, mean=0, std=1)
```

用正态分布 ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D(%5Ctext%7Bmean%7D%2C%20%5Ctext%7Bstd%7D)) 初始化输入 `Tensor`。

 
参数： 

*   **tensor** – n 维 `torch.Tensor`
*   **mean** – 正态分布的均值
*   **std** – 正态分布的标准差


例子

```py
>>> w = torch.empty(3, 5)
>>> nn.init.normal_(w)

```

```py
torch.nn.init.constant_(tensor, val)
```

用常数 ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bval%7D) 初始化输入 `Tensor`。

 
参数： 

*   **tensor** – n 维 `torch.Tensor`
*   **val** – 用以填入张量的常数


例子

```py
>>> w = torch.empty(3, 5)
>>> nn.init.constant_(w, 0.3)

```

```py
torch.nn.init.eye_(tensor)
```

用单位矩阵初始化 2 维输入 `Tensor`。 保持输入张量输入 `Linear` 时的独一性，并且越多越好.

 
参数：  

*   **tensor** – 2 维 `torch.Tensor` 


例子

```py
>>> w = torch.empty(3, 5)
>>> nn.init.eye_(w)

```

```py
torch.nn.init.dirac_(tensor)
```

用狄拉克δ函数初始化 {3, 4, 5} 维输入 `Tensor`。 保持输入张量输入 `Convolutional` 时的独一性，并且越多通道越好。

 
参数：  

*   **tensor** – {3, 4, 5} 维 `torch.Tensor` 

例子

```py
>>> w = torch.empty(3, 16, 5, 5)
>>> nn.init.dirac_(w)

```

```py
torch.nn.init.xavier_uniform_(tensor, gain=1)
```

用论文 “Understanding the difficulty of training deep feedforward neural networks” - Glorot, X. & Bengio, Y. (2010) 中提及的均匀分布初始化输入 `Tensor`。初始化后的张量中的值采样自 ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BU%7D(-a%2C%20a)) 且

![](http://latex.codecogs.com/gif.latex?%0D%0Aa%20%3D%20%5Ctext%7Bgain%7D%20%5Ctimes%20%5Csqrt%7B%5Cfrac%7B6%7D%7B%5Ctext%7Bfan%5C_in%7D%20%2B%20%5Ctext%7Bfan%5C_out%7D%7D%7D%0D%0A%0D%0A)

也被称作 Glorot 初始化。

 
参数： 

*   **tensor** – n 维 `torch.Tensor`
*   **gain** – 可选缩放因子


例子

```py
>>> w = torch.empty(3, 5)
>>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))

```

```py
torch.nn.init.xavier_normal_(tensor, gain=1)
```

用论文 “Understanding the difficulty of training deep feedforward neural networks” - Glorot, X. & Bengio, Y. (2010) 中提及的正态分布初始化输入 `Tensor`。初始化后的张量中的值采样自 ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D(0%2C%20%5Ctext%7Bstd%7D)) 且

![](http://latex.codecogs.com/gif.latex?%0D%0A%5Ctext%7Bstd%7D%20%3D%20%5Ctext%7Bgain%7D%20%5Ctimes%20%5Csqrt%7B%5Cfrac%7B2%7D%7B%5Ctext%7Bfan%5C_in%7D%20%2B%20%5Ctext%7Bfan%5C_out%7D%7D%7D%0D%0A%0D%0A)

也被称作 Glorot initialization。

 
参数： 

*   **tensor** – n 维 `torch.Tensor`
*   **gain** – 可选缩放因子


例子

```py
>>> w = torch.empty(3, 5)
>>> nn.init.xavier_normal_(w)

```

```py
torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
```

用论文 “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification” - He, K. et al. (2015) 中提及的均匀分布初始化输入 `Tensor`。初始化后的张量中的值采样自 ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BU%7D(-%5Ctext%7Bbound%7D%2C%20%5Ctext%7Bbound%7D)) 且

![](http://latex.codecogs.com/gif.latex?%0D%0A%5Ctext%7Bbound%7D%20%3D%20%5Csqrt%7B%5Cfrac%7B6%7D%7B(1%20%2B%20a%5E2)%20%5Ctimes%20%5Ctext%7Bfan%5C_in%7D%7D%7D%0D%0A%0D%0A)

也被称作 He initialization。

 
参数： 

*   **tensor** – n 维 `torch.Tensor`
*   **a** – 该层后面一层的整流函数中负的斜率 (默认为 0，此时为 Relu)
*   **mode** – ‘fan_in’ (default) 或者 ‘fan_out’。使用fan_in保持weights的方差在前向传播中不变；使用fan_out保持weights的方差在反向传播中不变。
*   **nonlinearity** – 非线性函数 (`nn.functional` 中的名字)，推荐只使用 ‘relu’ 或 ‘leaky_relu’ (default)。


例子

```py
>>> w = torch.empty(3, 5)
>>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')

```

```py
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
```

用论文 “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification” - He, K. et al. (2015) 中提及的正态分布初始化输入 `Tensor`。初始化后的张量中的值采样 ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D(0%2C%20%5Ctext%7Bstd%7D)) 且

![](http://latex.codecogs.com/gif.latex?%0D%0A%5Ctext%7Bstd%7D%20%3D%20%5Csqrt%7B%5Cfrac%7B2%7D%7B(1%20%2B%20a%5E2)%20%5Ctimes%20%5Ctext%7Bfan%5C_in%7D%7D%7D%0D%0A%0D%0A)

也被称作 He initialization。

 
参数： 

*   **tensor** – n 维 `torch.Tensor`
*   **a** – 该层后面一层的整流函数中负的斜率 (默认为 0，此时为 Relu)
*   **mode** – ‘fan_in’ (default) 或者 ‘fan_out’。使用fan_in保持weights的方差在前向传播中不变；使用fan_out保持weights的方差在反向传播中不变。
*   **nonlinearity** – 非线性函数 (`nn.functional` 中的名字)，推荐只使用 ‘relu’ 或 ‘leaky_relu’ (default)。


例子

```py
>>> w = torch.empty(3, 5)
>>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')

```

```py
torch.nn.init.orthogonal_(tensor, gain=1)
```

用论文 “Exact solutions to the nonlinear dynamics of learning in deep linear neural networks” - Saxe, A. et al. (2013) 中描述的(半）正定矩阵初始化输入 `Tensor`。输入张量必须至少有 2 维，如果输入张量的维度大于 2， 则对后续维度进行放平操作。

 
参数： 

*   **tensor** – n 维 `torch.Tensor`，且 ![](http://latex.codecogs.com/gif.latex?n%20%5Cgeq%202)
*   **gain** – 可选缩放因子


例子

```py
>>> w = torch.empty(3, 5)
>>> nn.init.orthogonal_(w)

```

```py
torch.nn.init.sparse_(tensor, sparsity, std=0.01)
```

用论文 “Deep learning via Hessian-free optimization” - Martens, J. (2010). 提及的稀疏矩阵初始化 2 维输入 `Tensor`，且使用正态分布 ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D(0%2C%200.01)) 初始化非零元素。

 
参数： 

*   **tensor** – n 维 `torch.Tensor`
*   **sparsity** – 每一行置零元素的比例
*   **std** – 初始化非零元素时使用正态分布的标准差


例子

```py
>>> w = torch.empty(3, 5)
>>> nn.init.sparse_(w, sparsity=0.1)

```
