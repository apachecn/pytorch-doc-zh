import sys
import torch
import torch._C as _C
from collections import OrderedDict
import torch.sparse as sparse
import torch.utils.hooks as hooks
import warnings
import weakref
from torch._six import imap


class Variable(_C._VariableBase):
    """封装一个张量用来各种操作.

    变量是张量对象周围的轻包装,能够拥有导数等数据, 这个引用允许回溯整个操作链创建数据.
    如果变量已经由用户创建, 它的 grad_fn
    为 ``None`` 我们称之为叶子节点.

    由于 autograd 只支持标量值函数微分, grad 大小始终与数据大小匹配. 此外,导数通常只分配
    叶变量,否则将始终为零.

    参数说明:
    *    data: 包裹任何类型的张量.
    *    grad: 变量保持类型和位置匹配的变量 ``.data``. 这个属性是懒惰的分配,不能被重新分配.
    *    requires_grad: 指示变量是否已被使用的布尔值由包含任何变量的子图创建,需要它.
        有关更多详细信息,请参阅 :ref:`excluded-subgraphs`.只能在叶变量上进行更改.
    *    volatile: 布尔值表示应该使用变量推理模式,即不保存历史. 查看 :ref:`excluding-subgraphs` 更多细节.
        只能在叶变量上进行更改.
    *    is_leaf: 指示是否为叶子节点,即是否由用户创建的节点.
    *    grad_fn: 导数函数跟踪.

    Args:
    *    data (any tensor class): 用来包装的张量.
    *    requires_grad (bool): 指示是否要被求导. **Keyword only.**
    *    volatile (bool): 指示是否可变. **Keyword only.**
    """

    _fallthrough_methods = {
        'size',
        'stride',
        'nelement',
        'ndimension',
        'element_size',
        'is_contiguous',
        'is_set_to',
        'is_signed',
        'numel',
        'dim',
        'get_device',
        'is_cuda',
        'shape'
    }

    def __getattr__(self, name):
        if name in self._fallthrough_methods:
            return getattr(self.data, name)
        return object.__getattribute__(self, name)

    def __getitem__(self, key):
        if torch.is_tensor(key):
            key = Variable(key)  # auto-wrap tensors
        if isinstance(key, Variable):
            if type(key.data).__name__ == 'ByteTensor':
                return MaskedSelect.apply(self, key)
            elif type(key.data).__name__ == 'LongTensor':
                return IndexSelect.apply(self, 0, key)
            # else fall through and raise an error in Index
        return Index.apply(self, key)

    def __setitem__(self, key, value):
        if isinstance(key, Variable) and type(key.data).__name__ == 'ByteTensor':
            if isinstance(value, Variable):
                return MaskedScatter.apply(self, key, value, True)
            else:
                return MaskedFill.apply(self, key, value, True)
        else:
            return SetItem.apply(self, key, value)

    def __deepcopy__(self, memo):
        if not self.is_leaf:
            raise RuntimeError("Only Variables created explicitly by the user "
                               "(graph leaves) support the deepcopy protocol at the moment")
        result = type(self)(self.data.clone())
        result.requires_grad = self.requires_grad
        result.volatile = self.volatile
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto):
        state = (self.requires_grad, self.volatile, self._backward_hooks)
        if proto > 1:
            return type(self), (self.data,), state
        if sys.version_info[0] == 2:
            from copy_reg import __newobj__
        else:
            from copyreg import __newobj__
        return __newobj__, (type(self), self.data), state

    def __setstate__(self, state):
        if len(state) == 5:
            # legacy serialization of Variable
            self.data = state[0]
            state = (state[3], state[4], state[2])
        if not self.is_leaf:
            raise RuntimeError('__setstate__ can be only called on leaf variables')
        self.requires_grad, self.volatile, self._backward_hooks = state

    def __repr__(self):
        return 'Variable containing:' + self.data.__repr__()

    def __bool__(self):
        if self.data.numel() == 0:
            return False
        raise RuntimeError("bool value of Variable objects containing non-empty " +
                           torch.typename(self.data) + " is ambiguous")

    __nonzero__ = __bool__

    def __int__(self):
        return int(self.data)

    def __long__(self):
        return long(self.data)

    def __float__(self):
        return float(self.data)

    def backward(self, gradient=None, retain_graph=None, create_graph=None, retain_variables=None):
        """给定图叶子节点计算导数.

       该图使用链式规则进行计算. 如果变量是非标量（即其数据具有多个元素）并且需要
       改变,该功能另外需要指定“梯度”.它应该是一个包含匹配类型和位置的张量
       微分函数的梯度w.r.t. ``self`` .

        这个功能在叶子上累积渐变 - 你可能需要调用之前将它们置零.

        Args:
        *    gradient (Tensor, Variable or None): 计算变量的梯度. 如果是张量,则会自动转换
            到一个变量,这是挥发性的,除非 ``create_graph`` 为真.没有值可以被指定为标量变量或那些
            不要求毕业. 如果一个None值是可以接受的这个参数是可选的.
        *    retain_graph (bool, 可选): 如果 “False” ,则用于计算的图形导数将被释放. 请注意,在几
            乎所有情况下设置这个选项为 True 是不需要的,通常可以解决在一个更有效的方式. 默认值为
            ``create_graph``.
        *    create_graph (bool, optional): 如果“真”,派生图将会被构造,允许计算更高阶的导数.
            默认为 ``False``,除非 ``gradient`` 是一个volatile变量.
        """
        torch.autograd.backward(self, gradient, retain_graph, create_graph, retain_variables)

    def register_hook(self, hook):
        """注册一个backward钩子.

        每次gradients被计算的时候,这个 hook 都被调用 .hook 应该拥有以下签名:

            hook(grad) -> Variable or None

        hook不应该修改它的输入,但是它可以选择性的返回一个替代当前梯度的新梯度.

        这个函数返回一个 句柄 (handle).它有一个方法 handle.remove(),可以用这个方法将 hook 从 module 移除.

        Example:
            >>> v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
            >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
            >>> v.backward(torch.Tensor([1, 1, 1]))
            >>> v.grad.data
             2
             2
             2
            [torch.FloatTensor of size 3]
            >>> h.remove()  # removes the hook
        """
        if self.volatile:
            raise RuntimeError("cannot register a hook on a volatile variable")
        if not self.requires_grad:
            raise RuntimeError("cannot register a hook on a variable that "
                               "doesn't require gradient")
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            if self.grad_fn is not None:
                self.grad_fn._register_hook_dict(self)
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def reinforce(self, reward):
        def trim(str):
            return '\n'.join([line.strip() for line in str.split('\n')])

        raise RuntimeError(trim(r"""reinforce() was removed.
            Use torch.distributions instead.
            See http://pytorch.org/docs/master/distributions.html

            Instead of:

            probs = policy_network(state)
            action = probs.multinomial()
            next_state, reward = env.step(action)
            action.reinforce(reward)
            action.backward()

            Use:

            probs = policy_network(state)
            # NOTE: categorical is equivalent to what used to be called multinomial
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            next_state, reward = env.step(action)
            loss = -m.log_prob(action) * reward
            loss.backward()
        """))

    def detach(self):
        """将一个Variable从创建它的图中分离,并把它设置成 leaf variable.


        .. 注意::

        返回变量使用与原始数据张量相同的数据张量,其中任何一个的就地修改都将被看到,并可能触发
        错误在正确性检查.
        """
        result = NoGrad()(self)  # this is needed, because it merges version counters
        result._grad_fn = None
        return result

    def detach_(self):
        """将一个 Variable 从创建它的图中分离,并把它设置成 leaf variable.
        """
        self._grad_fn = None
        self.requires_grad = False

    def retain_grad(self):
        """为非叶变量启用 .grad 属性."""
        if self.grad_fn is None:  # no-op for leaves
            return
        if not self.requires_grad:
            raise RuntimeError("can't retain_grad on Variable that has requires_grad=False")
        if hasattr(self, 'retains_grad'):
            return
        weak_self = weakref.ref(self)

        def retain_grad_hook(grad):
            var = weak_self()
            if var is None:
                return
            if var._grad is None:
                var._grad = grad.clone()
            else:
                var._grad = var._grad + grad

        self.register_hook(retain_grad_hook)
        self.retains_grad = True

    def contiguous(self):
        self.data = self.data.contiguous()
        return self

    def type(self, t):
        if t != type(self.data):
            return Type.apply(self, t)
        return self

    def type_as(self, t):
        if isinstance(t, Variable):
            t = t.data
        return self.type(type(t))

    def _get_type(self, name):
        module = torch._import_dotted_name(self.data.__module__)
        return getattr(module, name)

    def cuda(self, device=None, async=False):
        return CudaTransfer.apply(self, device, async)

    def cpu(self):
        return self.type(getattr(torch, type(self.data).__name__))

    def double(self):
        return self.type(self._get_type('DoubleTensor'))

    def float(self):
        return self.type(self._get_type('FloatTensor'))

    def half(self):
        return self.type(self._get_type('HalfTensor'))

    def long(self):
        return self.type(self._get_type('LongTensor'))

    def int(self):
        return self.type(self._get_type('IntTensor'))

    def short(self):
        return self.type(self._get_type('ShortTensor'))

    def char(self):
        return self.type(self._get_type('CharTensor'))

    def byte(self):
        return self.type(self._get_type('ByteTensor'))

    def clamp(self, min=None, max=None):
        if min is None and max is None:
            raise ValueError("clamp requires specifying at least one of "
                             "min and max arguments")
        elif min is None and max is not None:
            return CminConstant.apply(self, max)
        elif min is not None and max is None:
            return CmaxConstant.apply(self, min)
        else:
            return Clamp.apply(self, min, max)

    def prod(self, dim=None, keepdim=None):
        return Prod.apply(self, dim, keepdim)

    def view_as(self, tensor):
        return self.view(tensor.size())

    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], torch.Size):
            repeats = repeats[0]
        else:
            repeats = torch.Size(repeats)
        return Repeat.apply(self, repeats)

    def cumsum(self, dim):
        return Cumsum.apply(self, dim)

    def cumprod(self, dim):
        return Cumprod.apply(self, dim)

    def var(self, dim=None, keepdim=False, unbiased=True):
        if dim is None:
            mean = self.mean().view(*(1 for s in self.size()))
        else:
            mean = self.mean(dim, keepdim)
            # we could just set keepdim to True, but this preserves some fidelity
            if keepdim is False and self.dim() != 1:
                mean = mean.unsqueeze(dim)
        mean_expanded = mean.expand_as(self)
        zero_centered = self.sub(mean_expanded)
        if dim is None:
            var = zero_centered.mul(zero_centered).sum()
        else:
            var = zero_centered.mul(zero_centered).sum(dim, keepdim=keepdim)
        numel = self.numel() if dim is None else self.size(dim)
        return var.div(numel - int(unbiased))

    def std(self, dim=None, keepdim=False, unbiased=True):
        return self.var(dim, keepdim, unbiased).sqrt()

    def renorm(self, p, dim, maxnorm):
        t = self.transpose(dim, 0)
        flat = t.contiguous().view(self.size(0), -1)
        norms = flat.norm(p, 1, True)
        norms = norms.clamp(max=maxnorm).div(norms.add(1e-7))
        flat_out = flat.mul(norms.expand_as(flat))
        return flat_out.view(t.size()).transpose(dim, 0)

    def matmul(self, other):
        return torch.matmul(self, other)

    def resize(self, *sizes):
        return Resize.apply(self, sizes)

    def resize_as(self, variable):
        return Resize.apply(self, variable.size())

    def norm(self, p=2, dim=None, keepdim=False):
        if dim is None:
            return super(Variable, self).norm(p)
        else:
            return super(Variable, self).norm(p, dim, keepdim)

    def index_add(self, dim, index, tensor):
        return self.clone().index_add_(dim, index, tensor)

    def _advanced_index_add(self, index, tensor):
        return AdvancedIndexAdd.apply(self, index, tensor)

    def index_copy(self, dim, index, tensor):
        return self.clone().index_copy_(dim, index, tensor)

    def index_fill(self, dim, index, value):
        return self.clone().index_fill_(dim, index, value)

    def scatter(self, dim, index, source):
        return self.clone().scatter_(dim, index, source)

    def scatter_add(self, dim, index, source):
        return self.clone().scatter_add_(dim, index, source)

    def masked_copy(self, mask, variable):
        warnings.warn("masked_copy is deprecated and renamed to masked_scatter, and will be removed in v0.3")
        return self.masked_scatter(mask, variable)

    def masked_copy_(self, mask, variable):
        warnings.warn("masked_copy_ is deprecated and renamed to masked_scatter_, and will be removed in v0.3")
        return self.masked_scatter_(mask, variable)

    def masked_scatter(self, mask, variable):
        return self.clone().masked_scatter_(mask, variable)

    def masked_fill(self, mask, value):
        return self.clone().masked_fill_(mask, value)

    def expand_as(self, tensor):
        return self.expand(tensor.size())

    def multinomial(self, num_samples=1, replacement=False):
        return Categorical.apply(self, num_samples, replacement)

    def bernoulli(self):
        return Bernoulli.apply(self)

    def __rsub__(self, other):
        return -self + other

    def __matmul__(self, other):
        if not isinstance(other, Variable):
            return NotImplemented
        return self.matmul(other)

    def __rdiv__(self, other):
        return self.reciprocal() * other
    __rtruediv__ = __rdiv__

    __pow__ = _C._VariableBase.pow

    def __ipow__(self, other):
        raise NotImplementedError("in-place pow not implemented")

    def __rpow__(self, other):
        return PowConstant.apply(other, self)

    __neg__ = _C._VariableBase.neg

    __eq__ = _C._VariableBase.eq
    __ne__ = _C._VariableBase.ne
    __lt__ = _C._VariableBase.lt
    __le__ = _C._VariableBase.le
    __gt__ = _C._VariableBase.gt
    __ge__ = _C._VariableBase.ge

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # NB: we use 'imap' and not 'map' here, so that in Python 2 we get a
        # generator and don't eagerly perform all the indexes.  This could
        # save us work, and also helps keep trace ordering deterministic
        # (e.g., if you zip(*hiddens), the eager map will force all the
        # indexes of hiddens[0] before hiddens[1], while the generator
        # map will interleave them.)
        return iter(imap(lambda i: self[i], range(self.size(0))))

    def __hash__(self):
        return id(self)

    class _torch(object):
        @staticmethod
        def normal(means, std=1):
            return Normal.apply(means, std)


for method in dir(Variable):
    # This will also wrap some methods that normally aren't part of the
    # functional interface, but we don't care, as they won't ever be used
    if method.startswith('_') or method.endswith('_'):
        continue
    if hasattr(Variable._torch, method):
        continue
    as_static = staticmethod(getattr(Variable, method))
    setattr(Variable._torch, method, as_static)


from ._functions import *
from torch._C import _ImperativeEngine as ImperativeEngine
Variable._execution_engine = ImperativeEngine()
