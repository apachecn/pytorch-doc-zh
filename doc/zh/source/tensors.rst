.. currentmodule:: torch

torch.Tensor
===================================

 :class:`torch.Tensor` 是一种包含单一数据类型元素的多维矩阵.

Torch 定义了七种 CPU tensor 类型和八种 GPU tensor 类型:

======================== ===========================   ================================
Data type                CPU tensor                    GPU tensor
======================== ===========================   ================================
32-bit floating point    :class:`torch.FloatTensor`    :class:`torch.cuda.FloatTensor`
64-bit floating point    :class:`torch.DoubleTensor`   :class:`torch.cuda.DoubleTensor`
16-bit floating point    :class:`torch.HalfTensor`     :class:`torch.cuda.HalfTensor`
8-bit integer (unsigned) :class:`torch.ByteTensor`     :class:`torch.cuda.ByteTensor`
8-bit integer (signed)   :class:`torch.CharTensor`     :class:`torch.cuda.CharTensor`
16-bit integer (signed)  :class:`torch.ShortTensor`    :class:`torch.cuda.ShortTensor`
32-bit integer (signed)  :class:`torch.IntTensor`      :class:`torch.cuda.IntTensor`
64-bit integer (signed)  :class:`torch.LongTensor`     :class:`torch.cuda.LongTensor`
======================== ===========================   ================================

 :class:`torch.Tensor` 是默认的 tensor 类型(:class:`torch.FloatTensor`)的简称.

一个 tensor 对象可以从 Python 的 :class:`list` 或者序列(sequence)构建:

::

    >>> torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    1  2  3
    4  5  6
    [torch.FloatTensor of size 2x3]

一个空的 tensor 对象可以通过所指定的大小来构建:

::

    >>> torch.IntTensor(2, 4).zero_()
    0  0  0  0
    0  0  0  0
    [torch.IntTensor of size 2x4]

可以通过 Python 的索引和切片方式来获取或修改 tensor 对象的内容:

::

    >>> x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    >>> print(x[1][2])
    6.0
    >>> x[0][1] = 8
    >>> print(x)
     1  8  3
     4  5  6
    [torch.FloatTensor of size 2x3]

每一个 tensor 对象都有一个相应的 :class:`torch.Storage` 用来保存数据. 
tensor 类提供了一个存储的多维的,  有 `跨度(strided) <https://en.wikipedia.org/wiki/Stride_of_an_array>`_ 的视图, 并且在视图上定义了数值运算.

.. note::
   会改变 tensor 对象的函数方法名, 其使用了一个下划线后缀作为标识. 
   比如, :func:`torch.FloatTensor.abs_` 会在原地(in-place)计算绝对值并返回改变后的 tensor. 而 :func:`torch.FloatTensor.abs` 会在一个新建的 tensor 中计算结果.

.. class:: Tensor()
           Tensor(*sizes)
           Tensor(size)
           Tensor(sequence)
           Tensor(ndarray)
           Tensor(tensor)
           Tensor(storage)

   可以通过提供大小或者数据来创建一个新的 tensor 对象.

   如果没有提供参数, 将返回一个空的零维的 tensor. 
   如果提供了 :class:`numpy.ndarray`, :class:`torch.Tensor`, 或者 :class:`torch.Storage` 作为参数, 其将返回一个与参数共享数据的 tensor 对象.
   如果提供一个 Python 序列(sequence)作为参数, 将返回从序列的副本中创建的一个新的 tensor 对象.

   .. automethod:: abs
   .. automethod:: abs_
   .. automethod:: acos
   .. automethod:: acos_
   .. automethod:: add
   .. automethod:: add_
   .. automethod:: addbmm
   .. automethod:: addbmm_
   .. automethod:: addcdiv
   .. automethod:: addcdiv_
   .. automethod:: addcmul
   .. automethod:: addcmul_
   .. automethod:: addmm
   .. automethod:: addmm_
   .. automethod:: addmv
   .. automethod:: addmv_
   .. automethod:: addr
   .. automethod:: addr_
   .. automethod:: apply_
   
   .. automethod:: asin
   .. automethod:: asin_
   .. automethod:: atan
   .. automethod:: atan2
   .. automethod:: atan2_
   .. automethod:: atan_
   .. automethod:: baddbmm
   .. automethod:: baddbmm_
   .. automethod:: bernoulli
   .. automethod:: bernoulli_
   .. automethod:: bmm
   .. automethod:: byte
   .. automethod:: cauchy_
   .. automethod:: ceil
   .. automethod:: ceil_
   .. automethod:: char
   .. automethod:: chunk
   .. automethod:: clamp
   .. automethod:: clamp_
   .. automethod:: clone
   .. automethod:: contiguous
   .. automethod:: copy_
   .. automethod:: cos
   .. automethod:: cos_
   .. automethod:: cosh
   .. automethod:: cosh_
   .. automethod:: cpu
   .. automethod:: cross
   .. automethod:: cuda
   .. automethod:: cumprod
   .. automethod:: cumsum
   .. automethod:: data_ptr
   .. automethod:: diag
   .. automethod:: dim
   .. automethod:: dist
   .. automethod:: div
   .. automethod:: div_
   .. automethod:: dot
   .. automethod:: double
   .. automethod:: eig
   .. automethod:: element_size
   .. automethod:: eq
   .. automethod:: eq_
   .. automethod:: equal
   .. automethod:: erf
   .. automethod:: erf_
   .. automethod:: erfinv
   .. automethod:: erfinv_
   .. automethod:: exp
   .. automethod:: exp_
   .. automethod:: expand
   .. automethod:: expand_as
   .. automethod:: exponential_
   .. automethod:: fill_
   .. automethod:: float
   .. automethod:: floor
   .. automethod:: floor_
   .. automethod:: fmod
   .. automethod:: fmod_
   .. automethod:: frac
   .. automethod:: frac_
   .. automethod:: gather
   .. automethod:: ge
   .. automethod:: ge_
   .. automethod:: gels
   .. automethod:: geometric_
   .. automethod:: geqrf
   .. automethod:: ger
   .. automethod:: gesv
   .. automethod:: gt
   .. automethod:: gt_
   .. automethod:: half
   .. automethod:: histc
   .. automethod:: index
   .. automethod:: index_add_
   .. automethod:: index_copy_
   .. automethod:: index_fill_
   .. automethod:: index_select
   .. automethod:: int
   .. automethod:: inverse
   .. automethod:: is_contiguous
   .. autoattribute:: is_cuda
      :annotation:
   .. automethod:: is_pinned
   .. automethod:: is_set_to
   .. automethod:: is_signed
   .. automethod:: kthvalue
   .. automethod:: le
   .. automethod:: le_
   .. automethod:: lerp
   .. automethod:: lerp_
   .. automethod:: log
   .. automethod:: log1p
   .. automethod:: log1p_
   .. automethod:: log_
   .. automethod:: log_normal_
   .. automethod:: long
   .. automethod:: lt
   .. automethod:: lt_
   .. automethod:: map_
   .. automethod:: masked_scatter_
   .. automethod:: masked_fill_
   .. automethod:: masked_select
   .. automethod:: matmul
   .. automethod:: max
   .. automethod:: mean
   .. automethod:: median
   .. automethod:: min
   .. automethod:: mm
   .. automethod:: mode
   .. automethod:: mul
   .. automethod:: mul_
   .. automethod:: multinomial
   .. automethod:: mv
   .. automethod:: narrow
   .. automethod:: ndimension
   .. automethod:: ne
   .. automethod:: ne_
   .. automethod:: neg
   .. automethod:: neg_
   .. automethod:: nelement
   .. automethod:: new
   .. automethod:: nonzero
   .. automethod:: norm
   .. automethod:: normal_
   .. automethod:: numel
   .. automethod:: numpy
   .. automethod:: orgqr
   .. automethod:: ormqr
   .. automethod:: permute
   .. automethod:: pin_memory
   .. automethod:: potrf
   .. automethod:: potri
   .. automethod:: potrs
   .. automethod:: pow
   .. automethod:: pow_
   .. automethod:: prod
   .. automethod:: pstrf
   .. automethod:: put_
   .. automethod:: qr
   .. automethod:: random_
   .. automethod:: reciprocal
   .. automethod:: reciprocal_
   .. automethod:: remainder
   .. automethod:: remainder_
   .. automethod:: renorm
   .. automethod:: renorm_
   .. automethod:: repeat
   .. automethod:: resize_
   .. automethod:: resize_as_
   .. automethod:: round
   .. automethod:: round_
   .. automethod:: rsqrt
   .. automethod:: rsqrt_
   .. automethod:: scatter_
   .. automethod:: select
   .. automethod:: set_
   .. automethod:: share_memory_
   .. automethod:: short
   .. automethod:: sigmoid
   .. automethod:: sigmoid_
   .. automethod:: sign
   .. automethod:: sign_
   .. automethod:: sin
   .. automethod:: sin_
   .. automethod:: sinh
   .. automethod:: sinh_
   .. automethod:: size
   .. automethod:: sort
   .. automethod:: split
   .. automethod:: sqrt
   .. automethod:: sqrt_
   .. automethod:: squeeze
   .. automethod:: squeeze_
   .. automethod:: std
   .. automethod:: storage
   .. automethod:: storage_offset
   .. automethod:: storage_type
   .. automethod:: stride
   .. automethod:: sub
   .. automethod:: sub_
   .. automethod:: sum
   .. automethod:: svd
   .. automethod:: symeig
   .. automethod:: t
   .. automethod:: t_
   .. automethod:: take
   .. automethod:: tan
   .. automethod:: tan_
   .. automethod:: tanh
   .. automethod:: tanh_
   .. automethod:: tolist
   .. automethod:: topk
   .. automethod:: trace
   .. automethod:: transpose
   .. automethod:: transpose_
   .. automethod:: tril
   .. automethod:: tril_
   .. automethod:: triu
   .. automethod:: triu_
   .. automethod:: trtrs
   .. automethod:: trunc
   .. automethod:: trunc_
   .. automethod:: type
   .. automethod:: type_as
   .. automethod:: unfold
   .. automethod:: uniform_
   .. automethod:: unsqueeze
   .. automethod:: unsqueeze_
   .. automethod:: var
   .. automethod:: view
   .. automethod:: view_as
   .. automethod:: zero_

.. class:: ByteTensor()

   下面这些函数方法只存在于 :class:`torch.ByteTensor`.

   .. automethod:: all
   .. automethod:: any
