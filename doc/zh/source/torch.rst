torch
===================================
.. automodule:: torch

Tensors (张量) 
----------------------------------
.. autofunction:: is_tensor
.. autofunction:: is_storage
.. autofunction:: set_default_tensor_type
.. autofunction:: numel
.. autofunction:: set_printoptions


Creation Ops (创建操作) 
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: eye
.. autofunction:: from_numpy
.. autofunction:: linspace
.. autofunction:: logspace
.. autofunction:: ones
.. autofunction:: ones_like
.. autofunction:: arange
.. autofunction:: range
.. autofunction:: zeros
.. autofunction:: zeros_like

Indexing, Slicing, Joining, Mutating Ops (索引, 切片, 连接, 换位) 操作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: cat
.. autofunction:: chunk
.. autofunction:: gather
.. autofunction:: index_select
.. autofunction:: masked_select
.. autofunction:: nonzero
.. autofunction:: split
.. autofunction:: squeeze
.. autofunction:: stack
.. autofunction:: t
.. autofunction:: take
.. autofunction:: transpose
.. autofunction:: unbind
.. autofunction:: unsqueeze


Random sampling (随机采样) 
----------------------------------
.. autofunction:: manual_seed
.. autofunction:: initial_seed
.. autofunction:: get_rng_state
.. autofunction:: set_rng_state
.. autodata:: default_generator
.. autofunction:: bernoulli
.. autofunction:: multinomial
.. autofunction:: normal
.. autofunction:: rand
.. autofunction:: randn
.. autofunction:: randperm

In-place random sampling (直接随机采样) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在Tensors模块上还定义了许多 in-place 随机采样函数,可以点击参考它们的文档:

- :func:`torch.Tensor.bernoulli_` - 是 :func:`torch.bernoulli` 的 in-place 版本
- :func:`torch.Tensor.cauchy_` - 从柯西分布中抽取数字
- :func:`torch.Tensor.exponential_` - 从指数分布中抽取数字
- :func:`torch.Tensor.geometric_` - 从几何分布中抽取元素
- :func:`torch.Tensor.log_normal_` - 对数正态分布中的样本
- :func:`torch.Tensor.normal_` - 是 :func:`torch.normal` 的 in-place 版本
- :func:`torch.Tensor.random_` - 离散均匀分布中采样的数字
- :func:`torch.Tensor.uniform_` - 正态分布中采样的数字

Serialization (序列化) 
----------------------------------
.. autofunction:: save
.. autofunction:: load


Parallelism (并行化) 
----------------------------------
.. autofunction:: get_num_threads
.. autofunction:: set_num_threads


Math operations (数学操作) 
----------------------------------

Pointwise Ops (逐点操作) 
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: abs
.. autofunction:: acos
.. autofunction:: add
.. autofunction:: addcdiv
.. autofunction:: addcmul
.. autofunction:: asin
.. autofunction:: atan
.. autofunction:: atan2
.. autofunction:: ceil
.. autofunction:: clamp
.. autofunction:: cos
.. autofunction:: cosh
.. autofunction:: div
.. autofunction:: erf
.. autofunction:: erfinv
.. autofunction:: exp
.. autofunction:: floor
.. autofunction:: fmod
.. autofunction:: frac
.. autofunction:: lerp
.. autofunction:: log
.. autofunction:: log1p
.. autofunction:: mul
.. autofunction:: neg
.. autofunction:: pow
.. autofunction:: reciprocal
.. autofunction:: remainder
.. autofunction:: round
.. autofunction:: rsqrt
.. autofunction:: sigmoid
.. autofunction:: sign
.. autofunction:: sin
.. autofunction:: sinh
.. autofunction:: sqrt
.. autofunction:: tan
.. autofunction:: tanh
.. autofunction:: trunc


Reduction Ops (归约操作) 
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: cumprod
.. autofunction:: cumsum
.. autofunction:: dist
.. autofunction:: mean
.. autofunction:: median
.. autofunction:: mode
.. autofunction:: norm
.. autofunction:: prod
.. autofunction:: std
.. autofunction:: sum
.. autofunction:: var


Comparison Ops (比较操作) 
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: eq
.. autofunction:: equal
.. autofunction:: ge
.. autofunction:: gt
.. autofunction:: kthvalue
.. autofunction:: le
.. autofunction:: lt
.. autofunction:: max
.. autofunction:: min
.. autofunction:: ne
.. autofunction:: sort
.. autofunction:: topk


Other Operations (其它操作) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: cross
.. autofunction:: diag
.. autofunction:: histc
.. autofunction:: renorm
.. autofunction:: trace
.. autofunction:: tril
.. autofunction:: triu


BLAS and LAPACK Operations (BLAS和LAPACK操作)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: addbmm
.. autofunction:: addmm
.. autofunction:: addmv
.. autofunction:: addr
.. autofunction:: baddbmm
.. autofunction:: bmm
.. autofunction:: btrifact
.. autofunction:: btrisolve
.. autofunction:: dot
.. autofunction:: eig
.. autofunction:: gels
.. autofunction:: geqrf
.. autofunction:: ger
.. autofunction:: gesv
.. autofunction:: inverse
.. autofunction:: matmul
.. autofunction:: mm
.. autofunction:: mv
.. autofunction:: orgqr
.. autofunction:: ormqr
.. autofunction:: potrf
.. autofunction:: potri
.. autofunction:: potrs
.. autofunction:: pstrf
.. autofunction:: qr
.. autofunction:: svd
.. autofunction:: symeig
.. autofunction:: trtrs
