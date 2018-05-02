"""Adds docstrings to Tensor functions"""

import torch._C
from torch._C import _add_docstr as add_docstr


tensor_classes = [
    'DoubleTensorBase',
    'FloatTensorBase',
    'LongTensorBase',
    'IntTensorBase',
    'ShortTensorBase',
    'CharTensorBase',
    'ByteTensorBase',
]


def add_docstr_all(method, docstr):
    for cls_name in tensor_classes:
        cls = getattr(torch._C, cls_name)
        try:
            add_docstr(getattr(cls, method), docstr)
        except AttributeError:
            pass


add_docstr_all('abs',
               """
abs() -> Tensor

请查看 :func:`torch.abs`
""")

add_docstr_all('abs_',
               """
abs_() -> Tensor

 :meth:`~Tensor.abs` 的 in-place 运算形式
""")

add_docstr_all('acos',
               """
acos() -> Tensor

请查看 :func:`torch.acos`
""")

add_docstr_all('acos_',
               """
acos_() -> Tensor

:meth:`~Tensor.acos` 的 in-place 运算形式
""")

add_docstr_all('add',
               """
add(value)

请查看 :func:`torch.add`
""")

add_docstr_all('add_',
               """
add_(value)

 :meth:`~Tensor.add` 的 in-place 运算形式
""")

add_docstr_all('addbmm',
               """
addbmm(beta=1, mat, alpha=1, batch1, batch2) -> Tensor

请查看 :func:`torch.addbmm`
""")

add_docstr_all('addbmm_',
               """
addbmm_(beta=1, mat, alpha=1, batch1, batch2) -> Tensor

 :meth:`~Tensor.addbmm` 的 in-place 运算形式
""")

add_docstr_all('addcdiv',
               """
addcdiv(value=1, tensor1, tensor2) -> Tensor

请查看 :func:`torch.addcdiv`
""")

add_docstr_all('addcdiv_',
               """
addcdiv_(value=1, tensor1, tensor2) -> Tensor

 :meth:`~Tensor.addcdiv` 的 in-place 运算形式
""")

add_docstr_all('addcmul',
               """
addcmul(value=1, tensor1, tensor2) -> Tensor

请查看 :func:`torch.addcmul`
""")

add_docstr_all('addcmul_',
               """
addcmul_(value=1, tensor1, tensor2) -> Tensor

 :meth:`~Tensor.addcmul` 的 in-place 运算形式
""")

add_docstr_all('addmm',
               """
addmm(beta=1, mat, alpha=1, mat1, mat2) -> Tensor

请查看 :func:`torch.addmm`
""")

add_docstr_all('addmm_',
               """
addmm_(beta=1, mat, alpha=1, mat1, mat2) -> Tensor

 :meth:`~Tensor.addmm` 的 in-place 运算形式
""")

add_docstr_all('addmv',
               """
addmv(beta=1, tensor, alpha=1, mat, vec) -> Tensor

请查看 :func:`torch.addmv`
""")

add_docstr_all('addmv_',
               """
addmv_(beta=1, tensor, alpha=1, mat, vec) -> Tensor

 :meth:`~Tensor.addmv` 的 in-place 运算形式
""")

add_docstr_all('addr',
               """
addr(beta=1, alpha=1, vec1, vec2) -> Tensor

请查看 :func:`torch.addr`
""")

add_docstr_all('addr_',
               """
addr_(beta=1, alpha=1, vec1, vec2) -> Tensor

 :meth:`~Tensor.addr` 的 in-place 运算形式
""")

add_docstr_all('all',
               """
all() -> bool

如果 tensor 里的所有元素都是非零的, 则返回 True, 否在返回 False.
""")

add_docstr_all('any',
               """
any() -> bool

如果 tensor 里的存在元素是非零的, 则返回 True, 否在返回 False.
""")

add_docstr_all('apply_',
               """
apply_(callable) -> Tensor

将函数 :attr:`callable` 作用于 tensor 的每一个元素, 并将每个元素用 :attr:`callable` 的返回值替换.

.. note::

    该函数只能在 CPU tensor 中使用, 并且不应该用在有较高性能的要求的代码块中.
""")

add_docstr_all('asin',
               """
asin() -> Tensor

请查看 :func:`torch.asin`
""")

add_docstr_all('asin_',
               """
asin_() -> Tensor

 :meth:`~Tensor.asin` 的 in-place 运算形式
""")

add_docstr_all('atan',
               """
atan() -> Tensor

请查看 :func:`torch.atan`
""")

add_docstr_all('atan2',
               """
atan2(other) -> Tensor

请查看 :func:`torch.atan2`
""")

add_docstr_all('atan2_',
               """
atan2_(other) -> Tensor

 :meth:`~Tensor.atan2` 的 in-place 运算形式
""")

add_docstr_all('atan_',
               """
atan_() -> Tensor

 :meth:`~Tensor.atan` 的 in-place 运算形式
""")

add_docstr_all('baddbmm',
               """
baddbmm(beta=1, alpha=1, batch1, batch2) -> Tensor

请查看 :func:`torch.baddbmm`
""")

add_docstr_all('baddbmm_',
               """
baddbmm_(beta=1, alpha=1, batch1, batch2) -> Tensor

 :meth:`~Tensor.baddbmm` 的 in-place 运算形式
""")

add_docstr_all('bernoulli',
               """
bernoulli() -> Tensor

请查看 :func:`torch.bernoulli`
""")

add_docstr_all('bernoulli_',
               """
bernoulli_() -> Tensor

 :meth:`~Tensor.bernoulli` 的 in-place 运算形式
""")

add_docstr_all('bmm',
               """
bmm(batch2) -> Tensor

请查看 :func:`torch.bmm`
""")

add_docstr_all('cauchy_',
               """
cauchy_(median=0, sigma=1, *, generator=None) -> Tensor

用柯西分布得到的数值来填充 tensor 中的元素:

.. math::

    P(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x - median)^2 + \sigma^2}
""")

add_docstr_all('ceil',
               """
ceil() -> Tensor

请查看 :func:`torch.ceil`
""")

add_docstr_all('ceil_',
               """
ceil_() -> Tensor

 :meth:`~Tensor.ceil` 的 in-place 运算形式
""")

add_docstr_all('clamp',
               """
clamp(min, max) -> Tensor

请查看 :func:`torch.clamp`
""")

add_docstr_all('clamp_',
               """
clamp_(min, max) -> Tensor

 :meth:`~Tensor.clamp` 的 in-place 运算形式
""")

add_docstr_all('clone',
               """
clone() -> Tensor

返回与原 tensor 具有相同大小和数据类型的 tensor.
""")

add_docstr_all('contiguous',
               """
contiguous() -> Tensor

返回一个内存连续的有相同数据的 tensor, 如果原 tensor 内存连续则返回原 tensor.
""")

add_docstr_all('copy_',
               """
copy_(src, async=False, broadcast=True) -> Tensor

将 :attr:`src` 中的元素复制到这个 tensor 中并返回这个 tensor

如果 :attr:`broadcast` 是 True, 源 tensor 一定和这个 tensor :ref:`broadcastable <broadcasting-semantics>`.
另外, 源 tensor 的元素数量应该和这个 tensor 的元素个数一致.
源 tensor 可以是另一种数据类型, 或者在别的的设备上.

Args:
    src (Tensor): 被复制的源 tensor
    async (bool): 如果值为 ``True`` 并且这个复制操作在 CPU 和 GPU 之间进行, 则拷贝的副本与源信息可能会出现异步(asynchronously). 对于其他类型的复制操作, 这个参数不起作用.
    broadcast (bool): 如果值为 ``True``, :attr:`src` 将广播基础的 tensor 的形状.
""")

add_docstr_all('cos',
               """
cos() -> Tensor

请查看 :func:`torch.cos`
""")

add_docstr_all('cos_',
               """
cos_() -> Tensor

 :meth:`~Tensor.cos` 的 in-place 运算形式
""")

add_docstr_all('cosh',
               """
cosh() -> Tensor

请查看 :func:`torch.cosh`
""")

add_docstr_all('cosh_',
               """
cosh_() -> Tensor

 :meth:`~Tensor.cosh` 的 in-place 运算形式
""")

add_docstr_all('cross',
               """
cross(other, dim=-1) -> Tensor

请查看 :func:`torch.cross`
""")

add_docstr_all('cumprod',
               """
cumprod(dim) -> Tensor

请查看 :func:`torch.cumprod`
""")

add_docstr_all('cumsum',
               """
cumsum(dim) -> Tensor

请查看 :func:`torch.cumsum`
""")

add_docstr_all('data_ptr',
               """
data_ptr() -> int

返回 tensor 第一个元素的地址.
""")

add_docstr_all('diag',
               """
diag(diagonal=0) -> Tensor

请查看 :func:`torch.diag`
""")

add_docstr_all('dim',
               """
dim() -> int

返回 tensor 的维数.
""")

add_docstr_all('dist',
               """
dist(other, p=2) -> float

请查看 :func:`torch.dist`
""")

add_docstr_all('div',
               """
div(value)

请查看 :func:`torch.div`
""")

add_docstr_all('div_',
               """
div_(value)

 :meth:`~Tensor.div` 的 in-place 运算形式
""")

add_docstr_all('dot',
               """
dot(tensor2) -> float

请查看 :func:`torch.dot`
""")

add_docstr_all('eig',
               """
eig(eigenvectors=False) -> (Tensor, Tensor)

请查看 :func:`torch.eig`
""")

add_docstr_all('element_size',
               """
element_size() -> int

返回单个元素的字节大小.

Example:
    >>> torch.FloatTensor().element_size()
    4
    >>> torch.ByteTensor().element_size()
    1
""")

add_docstr_all('eq',
               """
eq(other) -> Tensor

请查看 :func:`torch.eq`
""")

add_docstr_all('eq_',
               """
eq_(other) -> Tensor

 :meth:`~Tensor.eq` 的 in-place 运算形式
""")

add_docstr_all('equal',
               """
equal(other) -> bool

请查看 :func:`torch.equal`
""")

add_docstr_all('erf',
               """
erf() -> Tensor

请查看 :func:`torch.erf`
""")

add_docstr_all('erfinv',
               """
erfinv() -> Tensor

请查看 :func:`torch.erfinv`
""")

add_docstr_all('exp',
               """
exp() -> Tensor

请查看 :func:`torch.exp`
""")

add_docstr_all('exp_',
               """
exp_() -> Tensor

 :meth:`~Tensor.exp` 的 in-place 运算形式
""")

add_docstr_all('exponential_',
               """
exponential_(lambd=1, *, generator=None) -> Tensor

将该 tensor 用指数分布得到的元素填充:

.. math::

    P(x) = \lambda e^{-\lambda x}
""")

add_docstr_all('fill_',
               """
fill_(value) -> Tensor

将该 tensor 用指定的数值填充.
""")

add_docstr_all('floor',
               """
floor() -> Tensor

请查看 :func:`torch.floor`
""")

add_docstr_all('floor_',
               """
floor_() -> Tensor

 :meth:`~Tensor.floor` 的 in-place 运算形式
""")

add_docstr_all('fmod',
               """
fmod(divisor) -> Tensor

请查看 :func:`torch.fmod`
""")

add_docstr_all('fmod_',
               """
fmod_(divisor) -> Tensor

 :meth:`~Tensor.fmod` 的 in-place 运算形式
""")

add_docstr_all('frac',
               """
frac() -> Tensor

请查看 :func:`torch.frac`
""")

add_docstr_all('frac_',
               """
frac_() -> Tensor

 :meth:`~Tensor.frac` 的 in-place 运算形式
""")

add_docstr_all('gather',
               """
gather(dim, index) -> Tensor

请查看 :func:`torch.gather`
""")

add_docstr_all('ge',
               """
ge(other) -> Tensor

请查看 :func:`torch.ge`
""")

add_docstr_all('ge_',
               """
ge_(other) -> Tensor

 :meth:`~Tensor.ge` 的 in-place 运算形式
""")

add_docstr_all('gels',
               """
gels(A) -> Tensor

请查看 :func:`torch.gels`
""")

add_docstr_all('geometric_',
               """
geometric_(p, *, generator=None) -> Tensor

将该 tensor 用几何分布得到的元素填充:

.. math::

    P(X=k) = (1 - p)^{k - 1} p

""")

add_docstr_all('geqrf',
               """
geqrf() -> (Tensor, Tensor)

请查看 :func:`torch.geqrf`
""")

add_docstr_all('ger',
               """
ger(vec2) -> Tensor

请查看 :func:`torch.ger`
""")

add_docstr_all('gesv',
               """
gesv(A) -> Tensor, Tensor

请查看 :func:`torch.gesv`
""")

add_docstr_all('gt',
               """
gt(other) -> Tensor

请查看 :func:`torch.gt`
""")

add_docstr_all('gt_',
               """
gt_(other) -> Tensor

 :meth:`~Tensor.gt` 的 in-place 运算形式
""")

add_docstr_all('histc',
               """
histc(bins=100, min=0, max=0) -> Tensor

请查看 :func:`torch.histc`
""")

add_docstr_all('index',
               """
index(m) -> Tensor

用一个二进制的掩码或沿着一个给定的维度从 tensor 中选取元素. ``tensor.index(m)`` 等同于 ``tensor[m]``.

Args:
    m (int or ByteTensor or slice): 用来选取元素的维度或掩码
""")

add_docstr_all('index_add_',
               """
index_add_(dim, index, tensor) -> Tensor

按参数 index 给出的索引序列, 将参数 tensor 中的元素加到原来的 tensor 中.
参数 tensor 的尺寸必须严格地与原 tensor 匹配, 否则会发生错误.

Args:
    dim (int): 索引 index 所指向的维度
    index (LongTensor): 从参数 tensor 中选取数据的索引序列
    tensor (Tensor): 包含需要相加的元素的 tensor

Example:
    >>> x = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    >>> t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> index = torch.LongTensor([0, 2, 1])
    >>> x.index_add_(0, index, t)
    >>> x
      2   3   4
      8   9  10
      5   6   7
    [torch.FloatTensor of size 3x3]
""")

add_docstr_all('index_copy_',
               """
index_copy_(dim, index, tensor) -> Tensor

按参数 index 给出的索引序列, 将参数 tensor 中的元素复制到原来的 tensor 中.
参数 tensor 的尺寸必须严格地与原 tensor 匹配, 否则会发生错误. 

Args:
    dim (int): 索引 index 所指向的维度
    index (LongTensor): 从参数 tensor 中选取数据的索引序列
    tensor (Tensor): 包含需要复制的元素的 tensor

Example:
    >>> x = torch.Tensor(3, 3)
    >>> t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> index = torch.LongTensor([0, 2, 1])
    >>> x.index_copy_(0, index, t)
    >>> x
     1  2  3
     7  8  9
     4  5  6
    [torch.FloatTensor of size 3x3]
""")

add_docstr_all('index_fill_',
               """
index_fill_(dim, index, val) -> Tensor

按参数 index 给出的索引序列, 将原 tensor 中的元素用 :attr:`val` 填充.

Args:
    dim (int): 索引 index 所指向的维度
    index (LongTensor): 从参数 val 中选取数据的索引序列
    val (float): 用来填充的值

Example:
    >>> x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> index = torch.LongTensor([0, 2])
    >>> x.index_fill_(1, index, -1)
    >>> x
    -1  2 -1
    -1  5 -1
    -1  8 -1
    [torch.FloatTensor of size 3x3]
""")

add_docstr_all('index_select',
               """
index_select(dim, index) -> Tensor

请查看 :func:`torch.index_select`
""")

add_docstr_all('inverse',
               """
inverse() -> Tensor

请查看 :func:`torch.inverse`
""")

add_docstr_all('is_contiguous',
               """
is_contiguous() -> bool

以 C 语言的内存模型为原则, 如果该 tensor 在内如果该 tensor 在内存中连续的, 则返回 True.
""")

add_docstr_all('is_set_to',
               """
is_set_to(tensor) -> bool

如果此对象从 Torch C API 引用的 ``THTensor`` 对象与参数 tensor 引用的对象一致, 则返回True.
""")

add_docstr_all('kthvalue',
               """
kthvalue(k, dim=None, keepdim=False) -> (Tensor, LongTensor)

请查看 :func:`torch.kthvalue`
""")

add_docstr_all('le',
               """
le(other) -> Tensor

请查看 :func:`torch.le`
""")

add_docstr_all('le_',
               """
le_(other) -> Tensor

 :meth:`~Tensor.le` 的 in-place 运算形式
""")

add_docstr_all('lerp',
               """
lerp(start, end, weight)

请查看 :func:`torch.lerp`
""")

add_docstr_all('lerp_',
               """
lerp_(start, end, weight)

 :meth:`~Tensor.lerp` 的 in-place 运算形式
""")

add_docstr_all('log',
               """
log() -> Tensor

请查看 :func:`torch.log`
""")

add_docstr_all('log1p',
               """
log1p() -> Tensor

请查看 :func:`torch.log1p`
""")

add_docstr_all('log1p_',
               """
log1p_() -> Tensor

 :meth:`~Tensor.log1p` 的 in-place 运算形式
""")

add_docstr_all('log_', """
log_() -> Tensor

 :meth:`~Tensor.log` 的 in-place 运算形式
""")

add_docstr_all('log_normal_', u"""
log_normal_(mean=1, std=2, *, generator=None)

将该 tensor 用均值为 mean (\u00B5), 标准差为 std (\u03C3) 的对数正态分布得到的元素填充. 
要注意 :attr:`mean` 和 :attr:`stdv` 是基本正态分布的均值和标准差, 不是返回的分布:

.. math::

    P(x) = \\dfrac{1}{x \\sigma \\sqrt{2\\pi}} e^{-\\dfrac{(\\ln x - \\mu)^2}{2\\sigma^2}}
""")

add_docstr_all('lt',
               """
lt(other) -> Tensor

请查看 :func:`torch.lt`
""")

add_docstr_all('lt_',
               """
lt_(other) -> Tensor

 :meth:`~Tensor.lt` 的 in-place 运算形式
""")

add_docstr_all('map_',
               """
map_(tensor, callable)

将 :attr:`callable` 作用于本 tensor 和参数 tensor 中的每一个元素, 并将结果存放在本 tensor 中. 
本 tensor 和参数 tensor 都必须是 :ref:`broadcastable <broadcasting-semantics>`.

 :attr:`callable` 应该有下列标志::

    def callable(a, b) -> number
""")

add_docstr_all('masked_scatter_',
               """
masked_scatter_(mask, source)


复制 :attr:`source` 的元素到本 tensor 被:attr:`mask` 中值为 1 的元素标记的位置中.
 :attr:`mask` 的形状和本 tensor 的形状必须是可广播的 ( :ref:`broadcastable <broadcasting-semantics>` ).
 :attr:`source` 中元素的个数最少为 :attr:`mask` 中值为1的元素的个数.

Args:
    mask (ByteTensor): 二进制掩码
    source (Tensor): 复制的源 tensor

.. note::

     :attr:`mask` 作用于 :attr:`self` 自身的 tensor, 而不是参数 :attr:`source` 的 tensor.
""")

add_docstr_all('masked_fill_',
               """
masked_fill_(mask, value)

将本 tensor 被 :attr:`mask` 中值为 1 的元素标记的位置, 用 :attr:`value` 填充.
 :attr:`mask` 的形状和本 tensor 的形状必须是可广播的 (:ref:`broadcastable <broadcasting-semantics>`).
Fills elements of this tensor with :attr:`value` where :attr:`mask` is one.

Args:
    mask (ByteTensor): 二进制掩码
    value (float): 用来填充的值
""")

add_docstr_all('masked_select',
               """
masked_select(mask) -> Tensor

请查看 :func:`torch.masked_select`
""")

add_docstr_all('max',
               """
max(dim=None, keepdim=False) -> float or (Tensor, Tensor)

请查看 :func:`torch.max`
""")

add_docstr_all('mean',
               """
mean(dim=None, keepdim=False) -> float or (Tensor, Tensor)

请查看 :func:`torch.mean`
""")

add_docstr_all('median',
               """
median(dim=None, keepdim=False) -> (Tensor, LongTensor)

请查看 :func:`torch.median`
""")

add_docstr_all('min',
               """
min(dim=None, keepdim=False) -> float or (Tensor, Tensor)

请查看 :func:`torch.min`
""")

add_docstr_all('mm',
               """
mm(mat2) -> Tensor

请查看 :func:`torch.mm`
""")

add_docstr_all('mode',
               """
mode(dim=None, keepdim=False) -> (Tensor, LongTensor)

请查看 :func:`torch.mode`
""")

add_docstr_all('mul',
               """
mul(value) -> Tensor

请查看 :func:`torch.mul`
""")

add_docstr_all('mul_',
               """
mul_(value)

 :meth:`~Tensor.mul` 的 in-place 运算形式
""")

add_docstr_all('multinomial',
               """
multinomial(num_samples, replacement=False, *, generator=None)

请查看 :func:`torch.multinomial`
""")

add_docstr_all('mv',
               """
mv(vec) -> Tensor

请查看 :func:`torch.mv`
""")

add_docstr_all('narrow',
               """
narrow(dimension, start, length) -> Tensor

返回一个本 tensor 经过缩小后的 tensor.
维度 dim 缩小范围是 :attr:`start` 到 :attr:`start + length`.
原 tensor 与返回的 tensor 共享相同的底层存储.

Args:
    dimension (int): 需要缩小的维度
    start (int): 起始维度
    length (int):

Example:
    >>> x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> x.narrow(0, 0, 2)
     1  2  3
     4  5  6
    [torch.FloatTensor of size 2x3]
    >>> x.narrow(1, 1, 2)
     2  3
     5  6
     8  9
    [torch.FloatTensor of size 3x2]
""")

add_docstr_all('ndimension',
               """
ndimension() -> int

 :meth:`~Tensor.dim()` 的另一种表示
""")

add_docstr_all('ne',
               """
ne(other) -> Tensor

请查看 :func:`torch.ne`
""")

add_docstr_all('ne_',
               """
ne_(other) -> Tensor

 :meth:`~Tensor.ne` 的 in-place 运算形式
""")

add_docstr_all('neg',
               """
neg() -> Tensor

请查看 :func:`torch.neg`
""")

add_docstr_all('neg_',
               """
neg_() -> Tensor

 :meth:`~Tensor.neg` 的 in-place 运算形式
""")

add_docstr_all('nelement',
               """
nelement() -> int

 :meth:`~Tensor.numel` 的另一种表示
""")

add_docstr_all('nonzero',
               """
nonzero() -> LongTensor

请查看 :func:`torch.nonzero`
""")

add_docstr_all('norm',
               """
norm(p=2, dim=None, keepdim=False) -> float

请查看 :func:`torch.norm`
""")

add_docstr_all('normal_',
               """
normal_(mean=0, std=1, *, generator=None)

将 tensor 用均值为 :attr:`mean` 和标准差为 :attr:`std` 的正态分布填充.
""")

add_docstr_all('numel',
               """
numel() -> int

请查看 :func:`torch.numel`
""")

add_docstr_all('numpy',
               """
numpy() -> ndarray

将该 tensor 以 NumPy :class:`ndarray` 的形式返回. 两者共享相同的底层存储. 
原 tensor 的改变会影响到 :class:`ndarray`, 反之也一样.
""")

add_docstr_all('orgqr',
               """
orgqr(input2) -> Tensor

请查看 :func:`torch.orgqr`
""")

add_docstr_all('ormqr',
               """
ormqr(input2, input3, left=True, transpose=False) -> Tensor

请查看 :func:`torch.ormqr`
""")

add_docstr_all('potrf',
               """
potrf(upper=True) -> Tensor

请查看 :func:`torch.potrf`
""")

add_docstr_all('potri',
               """
potri(upper=True) -> Tensor

请查看 :func:`torch.potri`
""")

add_docstr_all('potrs',
               """
potrs(input2, upper=True) -> Tensor

请查看 :func:`torch.potrs`
""")

add_docstr_all('pow',
               """
pow(exponent)

请查看 :func:`torch.pow`
""")

add_docstr_all('pow_',
               """
pow_(exponent)

 :meth:`~Tensor.pow` 的 in-place 运算形式
""")

add_docstr_all('prod',
               """
prod(dim=None, keepdim=False) -> float

请查看 :func:`torch.prod`
""")

add_docstr_all('pstrf',
               """
pstrf(upper=True, tol=-1) -> (Tensor, IntTensor)

请查看 :func:`torch.pstrf`
""")

add_docstr_all('put_',
               """
put_(indices, tensor, accumulate=False) -> Tensor

复制 :attr:`tensor` 内的元素到 indices 指定的位置.
为了达到索引的目的, ``self`` tensor 被当做一维 (1D) 的 tensor.

如果 :attr:`accumulate` 是 ``True``, :attr:`tensor` 内的元素累加到 :attr:`self` 中.
如果 :attr:`accumulate` 是 ``False``, 在索引包含重复的值时, 行为未定义.

Args:
    indices (LongTensor): self 的索引
    tensor (Tensor): 包含需要复制值的 tensor
    accumulate (bool): 如果是 True, 元素累加到 self

Example::

    >>> src = torch.Tensor([[4, 3, 5],
    ...                     [6, 7, 8]])
    >>> src.put_(torch.LongTensor([1, 3]), torch.Tensor([9, 10]))
      4   9   5
     10   7   8
    [torch.FloatTensor of size 2x3]
""")

add_docstr_all('qr',
               """
qr() -> (Tensor, Tensor)

请查看 :func:`torch.qr`
""")

add_docstr_all('random_',
               """
random_(from=0, to=None, *, generator=None)

将 tensor 用在 [from, to - 1] 上的离散均匀分布进行填充.
如果没有特别说明, 填入的值由本 tensor 的数据类型限定范围.
但是, 对于浮点类型 (floating point types), 如果没有特别说明, 取值范围是[0, 2^mantissa](mantissa,小数部分的长度), 以确保每个数都是可表示的.
例如, `torch.DoubleTensor(1).random_()` 将均匀分布在[0, 2^53].
""")

add_docstr_all('reciprocal',
               """
reciprocal() -> Tensor

请查看 :func:`torch.reciprocal`
""")

add_docstr_all('reciprocal_',
               """
reciprocal_() -> Tensor

 :meth:`~Tensor.reciprocal` 的 in-place 运算形式
""")

add_docstr_all('remainder',
               """
remainder(divisor) -> Tensor

请查看 :func:`torch.remainder`
""")

add_docstr_all('remainder_',
               """
remainder_(divisor) -> Tensor

 :meth:`~Tensor.remainder` 的 in-place 运算形式
""")

add_docstr_all('renorm',
               """
renorm(p, dim, maxnorm) -> Tensor

请查看 :func:`torch.renorm`
""")

add_docstr_all('renorm_',
               """
renorm_(p, dim, maxnorm) -> Tensor

 :meth:`~Tensor.renorm` 的 in-place 运算形式
""")

add_docstr_all('resize_',
               """
resize_(*sizes)

将 tensor 的大小调整为指定的大小. 
如果元素个数比当前的内存大小大, 就将底层存储大小调整为与新元素数目一致的大小.
如果元素个数比当前内存小, 则底层存储不会被改变.
原来tensor中被保存下来的元素将保持不变, 但新内存将不会被初始化. 

Args:
    sizes (torch.Size or int...): 期望的大小

Example:
    >>> x = torch.Tensor([[1, 2], [3, 4], [5, 6]])
    >>> x.resize_(2, 2)
    >>> x
     1  2
     3  4
    [torch.FloatTensor of size 2x2]
""")

add_docstr_all('resize_as_',
               """
resize_as_(tensor)

将本 tensor 的大小调整为参数 tensor 的大小.
等效于::

    self.resize_(tensor.size())
""")

add_docstr_all('round',
               """
round() -> Tensor

请查看 :func:`torch.round`
""")

add_docstr_all('round_',
               """
round_() -> Tensor

 :meth:`~Tensor.round` 的 in-place 运算形式
""")

add_docstr_all('rsqrt',
               """
rsqrt() -> Tensor

请查看 :func:`torch.rsqrt`
""")

add_docstr_all('rsqrt_',
               """
rsqrt_() -> Tensor

 :meth:`~Tensor.rsqrt` 的 in-place 运算形式
""")

add_docstr_all('scatter_',
               """
scatter_(dim, index, src) -> Tensor

将 :attr:`src` 中的所有值按照 :attr:`index` 确定的索引顺序写入本 tensor 中.
给定的 dim 声明索引的维度, dim 按照 :meth:`~Tensor.gather` 中的描述的规则来确定.

注意, 关于 gather, index 的值必须是 `0` 到 `(self.size(dim) -1)` 区间,
而且, 属于同一维度的一行的值必须是唯一的.

Args:
    dim (int): 索引的轴向
    index (LongTensor):散射元素的索引指数
    src (Tensor or float): 散射的源元素

Example::

    >>> x = torch.rand(2, 5)
    >>> x

     0.4319  0.6500  0.4080  0.8760  0.2355
     0.2609  0.4711  0.8486  0.8573  0.1029
    [torch.FloatTensor of size 2x5]

    >>> torch.zeros(3, 5).scatter_(0, torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)

     0.4319  0.4711  0.8486  0.8760  0.2355
     0.0000  0.6500  0.0000  0.8573  0.0000
     0.2609  0.0000  0.4080  0.0000  0.1029
    [torch.FloatTensor of size 3x5]

    >>> z = torch.zeros(2, 4).scatter_(1, torch.LongTensor([[2], [3]]), 1.23)
    >>> z

     0.0000  0.0000  1.2300  0.0000
     0.0000  0.0000  0.0000  1.2300
    [torch.FloatTensor of size 2x4]

""")

add_docstr_all('select',
               """
select(dim, index) -> Tensor or number

沿着 dim 给定的维度, 按照 index 切片.
如果这个 tensor 是一维的, 返回一个数字. 否则, 返回一个给定维度已经被移除的 tensor.

Args:
    dim (int): 切片的维度
    index (int): 用来选取的索引

.. note::

    :meth:`select`等效于切片. 例如,
    ``tensor.select(0, index)`` 等效于 ``tensor[index]`` 和
    ``tensor.select(2, index)`` 等效于 ``tensor[:,:,index]``.
""")

add_docstr_all('set_',
               """
set_(source=None, storage_offset=0, size=None, stride=None)

设置底层存储, 大小, 和步长.
如果 :attr:`source` 是一个 tensor 对象, 本 tensor 和该 tensor 共享底层存储, 并且大小和步长一样.
在其中一个 tensor 中改变元素, 会音响到另一个 tensor.

如果 :attr:`source` 是一个 :class:`~torch.Storage`, 则将设置底层内存, 偏移量, 大小和步长.

Args:
    source (Tensor or Storage): 用到的 tensor 或 storage
    storage_offset (int): storage 的偏移量
    size (torch.Size): 期望的大小. 默认为源 tensor 的大小.
    stride (tuple): 期望的步长. 默认为 C 相邻内存的步长.
""")

add_docstr_all('sigmoid',
               """
sigmoid() -> Tensor

请查看 :func:`torch.sigmoid`
""")

add_docstr_all('sigmoid_',
               """
sigmoid_() -> Tensor

 :meth:`~Tensor.sigmoid` 的 in-place 运算形式
""")

add_docstr_all('sign',
               """
sign() -> Tensor

请查看 :func:`torch.sign`
""")

add_docstr_all('sign_',
               """
sign_() -> Tensor

 :meth:`~Tensor.sign` 的 in-place 运算形式
""")

add_docstr_all('sin',
               """
sin() -> Tensor

请查看 :func:`torch.sin`
""")

add_docstr_all('sin_',
               """
sin_() -> Tensor

 :meth:`~Tensor.sin` 的 in-place 运算形式
""")

add_docstr_all('sinh',
               """
sinh() -> Tensor

请查看 :func:`torch.sinh`
""")

add_docstr_all('sinh_',
               """
sinh_() -> Tensor

 :meth:`~Tensor.sinh` 的 in-place 运算形式
""")

add_docstr_all('size',
               """
size() -> torch.Size

返回 tensor 的大小. 返回的值是 :class:`tuple` 的子类. 

Example:
    >>> torch.Tensor(3, 4, 5).size()
    torch.Size([3, 4, 5])
""")

add_docstr_all('sort',
               """
sort(dim=None, descending=False) -> (Tensor, LongTensor)

请查看 :func:`torch.sort`
""")

add_docstr_all('sqrt',
               """
sqrt() -> Tensor

请查看 :func:`torch.sqrt`
""")

add_docstr_all('sqrt_',
               """
sqrt_() -> Tensor

 :meth:`~Tensor.sqrt` 的 in-place 运算形式
""")

add_docstr_all('squeeze',
               """
squeeze(dim=None)

请查看 :func:`torch.squeeze`
""")

add_docstr_all('squeeze_',
               """
squeeze_(dim=None)

 :meth:`~Tensor.squeeze` 的 in-place 运算形式
""")

add_docstr_all('std',
               """
std(dim=None, unbiased=True, keepdim=False) -> float

请查看 :func:`torch.std`
""")

add_docstr_all('storage',
               """
storage() -> torch.Storage

返回底层存储
""")

add_docstr_all('storage_offset',
               """
storage_offset() -> int

按照储存元素个数的偏移返回 tensor 在底层存储中的偏移量(不是按照字节计算).

Example:
    >>> x = torch.Tensor([1, 2, 3, 4, 5])
    >>> x.storage_offset()
    0
    >>> x[3:].storage_offset()
    3
""")

add_docstr_all('stride',
               """
stride(dim) -> tuple or int

返回 tesnor 的步长. 
步长是指按照 dim 指定的维度, 从一个元素到下一个元素需要跳跃的距离.
当没有指定维度, 会计算所有维度的步长, 并返回一个 tuple.
当给定维度时, 返回这个维度的步长.

Args:
    dim (int): 期望的需要计算步长的维度.

Example:
    >>> x = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> x.stride()
    (5, 1)
    >>>x.stride(0)
    5
    >>> x.stride(-1)
    1
""")

add_docstr_all('sub',
               """
sub(value, other) -> Tensor

从 tensor 中抽取一个标量或张量. 
如果 :attr:`value` 和 :attr:`other` 都是给定的, 则在使用之前 :attr:`other` 的每一个元素都会被 :attr:`value` 缩放.

如果 :attr:`other` 是一个tensor,  :attr:`other` 的形状必须于基础 tensor 的形状是可广播的 ( :ref:`broadcastable <broadcasting-semantics>` ).

""")

add_docstr_all('sub_',
               """
sub_(x) -> Tensor

 :meth:`~Tensor.sub` 的 in-place 运算形式
""")

add_docstr_all('sum',
               """
sum(dim=None, keepdim=False) -> float

请查看 :func:`torch.sum`
""")

add_docstr_all('svd',
               """
svd(some=True) -> (Tensor, Tensor, Tensor)

请查看 :func:`torch.svd`
""")

add_docstr_all('symeig',
               """
symeig(eigenvectors=False, upper=True) -> (Tensor, Tensor)

请查看 :func:`torch.symeig`
""")

add_docstr_all('t',
               """
t() -> Tensor

请查看 :func:`torch.t`
""")

add_docstr_all('t_',
               """
t_() -> Tensor

 :meth:`~Tensor.t` 的 in-place 运算形式
""")

add_docstr_all('take',
               """
take(indices) -> Tensor

请查看 :func:`torch.take`
""")

add_docstr_all('tan_',
               """
tan_() -> Tensor

 :meth:`~Tensor.tan` 的 in-place 运算形式
""")

add_docstr_all('tanh',
               """
tanh() -> Tensor

请查看 :func:`torch.tanh`
""")

add_docstr_all('tanh_',
               """
tanh_() -> Tensor

 :meth:`~Tensor.tanh` 的 in-place 运算形式
""")

add_docstr_all('topk',
               """
topk(k, dim=None, largest=True, sorted=True) -> (Tensor, LongTensor)

请查看 :func:`torch.topk`
""")

add_docstr_all('trace',
               """
trace() -> float

请查看 :func:`torch.trace`
""")

add_docstr_all('transpose',
               """
transpose(dim0, dim1) -> Tensor

请查看 :func:`torch.transpose`
""")

add_docstr_all('transpose_',
               """
transpose_(dim0, dim1) -> Tensor

 :meth:`~Tensor.transpose` 的 in-place 运算形式
""")

add_docstr_all('tril',
               """
tril(k=0) -> Tensor

请查看 :func:`torch.tril`
""")

add_docstr_all('tril_',
               """
tril_(k=0) -> Tensor

 :meth:`~Tensor.tril`
""")

add_docstr_all('triu',
               """
triu(k=0) -> Tensor

请查看 :func:`torch.triu`
""")

add_docstr_all('triu_',
               """
triu_(k=0) -> Tensor

 :meth:`~Tensor.triu` 的 in-place 运算形式
""")

add_docstr_all('trtrs',
               """
trtrs(A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)

请查看 :func:`torch.trtrs`
""")

add_docstr_all('trunc',
               """
trunc() -> Tensor

请查看 :func:`torch.trunc`
""")

add_docstr_all('trunc_',
               """
trunc_() -> Tensor

 :meth:`~Tensor.trunc` 的 in-place 运算形式
""")

add_docstr_all('unfold',
               """
unfold(dim, size, step) -> Tensor

返回一个在 :attr:`dim` 维度上包含所有 :attr:`size` 大小切片的 tensor.

 :attr:`step` 说明两个切片之间的步长.

如果 `sizedim` 是原tensor在 dim 维度原来的大小, 则返回的 tensor 在 `dim` 维度的大小是 `(sizedim - size) / step + 1`

一个额外的切片大小的维度已经添加在返回的 tensor 中.

Args:
    dim (int): 需要展开的维度
    size (int): 每一个分片需要展开的大小
    step (int): 相邻分片之间的步长

Example::

    >>> x = torch.arange(1, 8)
    >>> x

     1
     2
     3
     4
     5
     6
     7
    [torch.FloatTensor of size 7]

    >>> x.unfold(0, 2, 1)

     1  2
     2  3
     3  4
     4  5
     5  6
     6  7
    [torch.FloatTensor of size 6x2]

    >>> x.unfold(0, 2, 2)

     1  2
     3  4
     5  6
    [torch.FloatTensor of size 3x2]

""")

add_docstr_all('uniform_',
               """
uniform_(from=0, to=1) -> Tensor

将 tensor 用从均匀分布中抽样得到的值填充:

.. math:

    P(x) = \dfrac{1}{to - from}
""")

add_docstr_all('unsqueeze',
               """
unsqueeze(dim)

请查看 :func:`torch.unsqueeze`
""")

add_docstr_all('unsqueeze_',
               """
unsqueeze_(dim)

 :meth:`~Tensor.unsqueeze` 的 in-place 运算形式
""")

add_docstr_all('var',
               """
var(dim=None, unbiased=True, keepdim=False) -> float

请查看 :func:`torch.var`
""")

add_docstr_all('view',
               """
view(*args) -> Tensor

返回一个有相同数据但大小不同的新的 tensor.

返回的 tensor 与原 tensor 共享相同的数据, 一定有相同数目的元素, 但大小不同.
一个 tensor 必须是连续的 ( :func:`contiguous` ) 才能被查看.

Args:
    args (torch.Size or int...): 期望的大小

Example:
    >>> x = torch.randn(4, 4)
    >>> x.size()
    torch.Size([4, 4])
    >>> y = x.view(16)
    >>> y.size()
    torch.Size([16])
    >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    >>> z.size()
    torch.Size([2, 8])
""")

add_docstr_all('expand',
               """
expand(*sizes) -> Tensor

返回 tensor 单个维度扩展到大的一个新的视图.

传递 -1 作为一个维度的大小, 表示这个维度的大小不做改变.

Tensor 也可以扩展到一个很大的维数, 新添加的维度将放在前面. (对于新的维度, 大小不能设置为 -1 .)

扩展一个 tensor 不是分配一个新的内存, 而只是在这个存在的 tensor 上, 通过设置 ``stride`` 为 0, 创建一个新的某个维度从 1 扩展到很大的视图.
任何大小为 1 的维度, 在不用重新分配内存的情况下, 可以扩展到随意任何一个值.

Args:
    *sizes (torch.Size or int...): 期望扩展的大小

Example:
    >>> x = torch.Tensor([[1], [2], [3]])
    >>> x.size()
    torch.Size([3, 1])
    >>> x.expand(3, 4)
     1  1  1  1
     2  2  2  2
     3  3  3  3
    [torch.FloatTensor of size 3x4]
    >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
     1  1  1  1
     2  2  2  2
     3  3  3  3
    [torch.FloatTensor of size 3x4]
""")

add_docstr_all('zero_',
               """
zero_()

用0填充该 tensor.
""")
