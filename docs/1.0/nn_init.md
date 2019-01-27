# torch.nn.init

```py
torch.nn.init.calculate_gain(nonlinearity, param=None)
```

Return the recommended gain value for the given nonlinearity function. The values are as follows:

| nonlinearity | gain |
| --- | --- |
| Linear / Identity | ![](http://latex.codecogs.com/gif.latex?1) |
| Conv{1,2,3}D | ![](http://latex.codecogs.com/gif.latex?1) |
| Sigmoid | ![](http://latex.codecogs.com/gif.latex?1) |
| Tanh | ![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B5%7D%7B3%7D) |
| ReLU | ![](http://latex.codecogs.com/gif.latex?%5Csqrt%7B2%7D) |
| Leaky Relu | ![](http://latex.codecogs.com/gif.latex?%5Csqrt%7B%5Cfrac%7B2%7D%7B1%20%2B%20%5Ctext%7Bnegative%5C_slope%7D%5E2%7D%7D) |

 
| Parameters: | 

*   **nonlinearity** – the non-linear function (`nn.functional` name)
*   **param** – optional parameter for the non-linear function

 |
| --- | --- |

Examples

```py
>>> gain = nn.init.calculate_gain('leaky_relu')

```

```py
torch.nn.init.uniform_(tensor, a=0, b=1)
```

Fills the input Tensor with values drawn from the uniform distribution ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BU%7D(a%2C%20b)).

 
| Parameters: | 

*   **tensor** – an n-dimensional `torch.Tensor`
*   **a** – the lower bound of the uniform distribution
*   **b** – the upper bound of the uniform distribution

 |
| --- | --- |

Examples

```py
>>> w = torch.empty(3, 5)
>>> nn.init.uniform_(w)

```

```py
torch.nn.init.normal_(tensor, mean=0, std=1)
```

Fills the input Tensor with values drawn from the normal distribution ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D(%5Ctext%7Bmean%7D%2C%20%5Ctext%7Bstd%7D)).

 
| Parameters: | 

*   **tensor** – an n-dimensional `torch.Tensor`
*   **mean** – the mean of the normal distribution
*   **std** – the standard deviation of the normal distribution

 |
| --- | --- |

Examples

```py
>>> w = torch.empty(3, 5)
>>> nn.init.normal_(w)

```

```py
torch.nn.init.constant_(tensor, val)
```

Fills the input Tensor with the value ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bval%7D).

 
| Parameters: | 

*   **tensor** – an n-dimensional `torch.Tensor`
*   **val** – the value to fill the tensor with

 |
| --- | --- |

Examples

```py
>>> w = torch.empty(3, 5)
>>> nn.init.constant_(w, 0.3)

```

```py
torch.nn.init.eye_(tensor)
```

Fills the 2-dimensional input `Tensor` with the identity matrix. Preserves the identity of the inputs in `Linear` layers, where as many inputs are preserved as possible.

 
| Parameters: | **tensor** – a 2-dimensional `torch.Tensor` |
| --- | --- |

Examples

```py
>>> w = torch.empty(3, 5)
>>> nn.init.eye_(w)

```

```py
torch.nn.init.dirac_(tensor)
```

Fills the {3, 4, 5}-dimensional input `Tensor` with the Dirac delta function. Preserves the identity of the inputs in `Convolutional` layers, where as many input channels are preserved as possible.

 
| Parameters: | **tensor** – a {3, 4, 5}-dimensional `torch.Tensor` |
| --- | --- |

Examples

```py
>>> w = torch.empty(3, 16, 5, 5)
>>> nn.init.dirac_(w)

```

```py
torch.nn.init.xavier_uniform_(tensor, gain=1)
```

Fills the input `Tensor` with values according to the method described in “Understanding the difficulty of training deep feedforward neural networks” - Glorot, X. & Bengio, Y. (2010), using a uniform distribution. The resulting tensor will have values sampled from ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BU%7D(-a%2C%20a)) where

![](http://latex.codecogs.com/gif.latex?%0D%0Aa%20%3D%20%5Ctext%7Bgain%7D%20%5Ctimes%20%5Csqrt%7B%5Cfrac%7B6%7D%7B%5Ctext%7Bfan%5C_in%7D%20%2B%20%5Ctext%7Bfan%5C_out%7D%7D%7D%0D%0A%0D%0A)

Also known as Glorot initialization.

 
| Parameters: | 

*   **tensor** – an n-dimensional `torch.Tensor`
*   **gain** – an optional scaling factor

 |
| --- | --- |

Examples

```py
>>> w = torch.empty(3, 5)
>>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))

```

```py
torch.nn.init.xavier_normal_(tensor, gain=1)
```

Fills the input `Tensor` with values according to the method described in “Understanding the difficulty of training deep feedforward neural networks” - Glorot, X. & Bengio, Y. (2010), using a normal distribution. The resulting tensor will have values sampled from ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D(0%2C%20%5Ctext%7Bstd%7D)) where

![](http://latex.codecogs.com/gif.latex?%0D%0A%5Ctext%7Bstd%7D%20%3D%20%5Ctext%7Bgain%7D%20%5Ctimes%20%5Csqrt%7B%5Cfrac%7B2%7D%7B%5Ctext%7Bfan%5C_in%7D%20%2B%20%5Ctext%7Bfan%5C_out%7D%7D%7D%0D%0A%0D%0A)

Also known as Glorot initialization.

 
| Parameters: | 

*   **tensor** – an n-dimensional `torch.Tensor`
*   **gain** – an optional scaling factor

 |
| --- | --- |

Examples

```py
>>> w = torch.empty(3, 5)
>>> nn.init.xavier_normal_(w)

```

```py
torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
```

Fills the input `Tensor` with values according to the method described in “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification” - He, K. et al. (2015), using a uniform distribution. The resulting tensor will have values sampled from ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BU%7D(-%5Ctext%7Bbound%7D%2C%20%5Ctext%7Bbound%7D)) where

![](http://latex.codecogs.com/gif.latex?%0D%0A%5Ctext%7Bbound%7D%20%3D%20%5Csqrt%7B%5Cfrac%7B6%7D%7B(1%20%2B%20a%5E2)%20%5Ctimes%20%5Ctext%7Bfan%5C_in%7D%7D%7D%0D%0A%0D%0A)

Also known as He initialization.

 
| Parameters: | 

*   **tensor** – an n-dimensional `torch.Tensor`
*   **a** – the negative slope of the rectifier used after this layer (0 for ReLU by default)
*   **mode** – either ‘fan_in’ (default) or ‘fan_out’. Choosing `fan_in` preserves the magnitude of the variance of the weights in the forward pass. Choosing `fan_out` preserves the magnitudes in the backwards pass.
*   **nonlinearity** – the non-linear function (`nn.functional` name), recommended to use only with ‘relu’ or ‘leaky_relu’ (default).

 |
| --- | --- |

Examples

```py
>>> w = torch.empty(3, 5)
>>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')

```

```py
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
```

Fills the input `Tensor` with values according to the method described in “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification” - He, K. et al. (2015), using a normal distribution. The resulting tensor will have values sampled from ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D(0%2C%20%5Ctext%7Bstd%7D)) where

![](http://latex.codecogs.com/gif.latex?%0D%0A%5Ctext%7Bstd%7D%20%3D%20%5Csqrt%7B%5Cfrac%7B2%7D%7B(1%20%2B%20a%5E2)%20%5Ctimes%20%5Ctext%7Bfan%5C_in%7D%7D%7D%0D%0A%0D%0A)

Also known as He initialization.

 
| Parameters: | 

*   **tensor** – an n-dimensional `torch.Tensor`
*   **a** – the negative slope of the rectifier used after this layer (0 for ReLU by default)
*   **mode** – either ‘fan_in’ (default) or ‘fan_out’. Choosing `fan_in` preserves the magnitude of the variance of the weights in the forward pass. Choosing `fan_out` preserves the magnitudes in the backwards pass.
*   **nonlinearity** – the non-linear function (`nn.functional` name), recommended to use only with ‘relu’ or ‘leaky_relu’ (default).

 |
| --- | --- |

Examples

```py
>>> w = torch.empty(3, 5)
>>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')

```

```py
torch.nn.init.orthogonal_(tensor, gain=1)
```

Fills the input `Tensor` with a (semi) orthogonal matrix, as described in “Exact solutions to the nonlinear dynamics of learning in deep linear neural networks” - Saxe, A. et al. (2013). The input tensor must have at least 2 dimensions, and for tensors with more than 2 dimensions the trailing dimensions are flattened.

 
| Parameters: | 

*   **tensor** – an n-dimensional `torch.Tensor`, where ![](http://latex.codecogs.com/gif.latex?n%20%5Cgeq%202)
*   **gain** – optional scaling factor

 |
| --- | --- |

Examples

```py
>>> w = torch.empty(3, 5)
>>> nn.init.orthogonal_(w)

```

```py
torch.nn.init.sparse_(tensor, sparsity, std=0.01)
```

Fills the 2D input `Tensor` as a sparse matrix, where the non-zero elements will be drawn from the normal distribution ![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D(0%2C%200.01)), as described in “Deep learning via Hessian-free optimization” - Martens, J. (2010).

 
| Parameters: | 

*   **tensor** – an n-dimensional `torch.Tensor`
*   **sparsity** – The fraction of elements in each column to be set to zero
*   **std** – the standard deviation of the normal distribution used to generate the non-zero values

 |
| --- | --- |

Examples

```py
>>> w = torch.empty(3, 5)
>>> nn.init.sparse_(w, sparsity=0.1)

```