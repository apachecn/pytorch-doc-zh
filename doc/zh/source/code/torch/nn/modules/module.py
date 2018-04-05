from collections import OrderedDict
import functools

import torch
from ..backends.thnn import backend as thnn_backend
from ..parameter import Parameter
from torch.autograd import Variable
import torch.utils.hooks as hooks


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Module(object):
    r"""所有神经网络的基类.

    你的模型应该也是该类的子类.

    Modules 也可以包含其它 Modules, 允许使用树结构嵌入它们.
    你可以将子模块赋值给模型属性 ::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

    以这种方式分配的子模块将被注册, 并且在调用 .cuda() 等等方法时也将转换它们的参数.
    """

    dump_patches = False

    def __init__(self):
        self._backend = thnn_backend
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

    def forward(self, *input):
        """定义每次调用时执行的计算.

        应该被所有的子类重写.

        .. note::
            尽管需要在此函数中定义正向传递的方式,
            但是应该事后尽量调用 :class:`Module` 实例,
            因为前者负责运行已注册的钩子, 而后者静默的忽略它们.
        """
        raise NotImplementedError

    def register_buffer(self, name, tensor):
        """给模块添加一个持久化的 buffer.

        持久化的 buffer 通常被用在这么一种情况: 我们需要保存一个状态, 但是这个状态不能看作成为模型参数.
        例如: BatchNorm 的 ``running_mean`` 不是一个 parameter, 但是它也是需要保存的状态之一.

        Buffers 可以使用指定的 name 作为属性访问.

        Args:
            name (string): buffer 的名称. 可以使用指定的 name 从该模块访问 buffer
            tensor (Tensor): 被注册的 buffer.

        Example:
            >>> self.register_buffer('running_mean', torch.zeros(num_features))
        """
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))

        self._buffers[name] = tensor

    def register_parameter(self, name, param):
        """添加一个参数到模块中.

        可以使用指定的 name 属性来访问参数.

        Args:
            name (string): 参数名. 可以使用指定的 name 来从该模块中访问参数
            parameter (Parameter): 要被添加到模块的参数.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        if hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(torch.typename(param), name))
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Variable to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another variable, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parameters[name] = param

    def add_module(self, name, module):
        """添加一个 child module（子模块）到当前的 module（模块）中.

        被添加的 module 还可以通过指定的 name 属性来获取它.

        Args:
            name (string): 子模块的名称. 可以使用指定的 name 从该模块访问子模块
            parameter (Module): 被添加到模块的子模块.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                torch.typename(module)))
        if hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        self._modules[name] = module

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def apply(self, fn):
        """将 ``fn`` 函数递归的应用到每一个子模块 (由 ``.children()`` 方法所返回的)
        以及 self. 典型的用于包括初始化模型的参数 (也可参阅 :ref:`torch-nn-init`).

        Args:
            fn (:class:`Module` -> None): 要被应用到每一个子模块上的函数

        Returns:
            Module: self

        Example:
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.data.fill_(1.0)
            >>>         print(m.weight)
            >>>
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear (2 -> 2)
            Parameter containing:
             1  1
             1  1
            [torch.FloatTensor of size 2x2]
            Linear (2 -> 2)
            Parameter containing:
             1  1
             1  1
            [torch.FloatTensor of size 2x2]
            Sequential (
              (0): Linear (2 -> 2)
              (1): Linear (2 -> 2)
            )
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def cuda(self, device=None):
        """将所有的模型参数和缓冲区移动到 GPU.

        这将会关联一些参数并且缓存不同的对象.
        所以在构建优化器之前应该调用它, 如果模块在优化的情况下会生存在 GPU 上.

        Arguments:
            device (int, optional): 如果指定, 所有参数将被复制到指定的设备上

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        """将所有的模型参数和缓冲区移动到 CPU.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cpu())

    def type(self, dst_type):
        """转换所有参数和缓冲区为 dst_type.

        Arguments:
            dst_type (type or string): 理想的类型

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.type(dst_type))

    def float(self):
        """将所有的 parameters 和 buffers 的数据类型转换成float.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.float())

    def double(self):
        """将所有的 parameters 和 buffers 的数据类型转换成 double.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.double())

    def half(self):
        """将所有的 parameters 和 buffers 的数据类型转换成 half.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.half())

    def register_backward_hook(self, hook):
        """在模块上注册一个 backward hook（反向钩子）.

        每次计算关于模块输入的梯度时, 都会调用该钩子.
        钩子应该有以下结构::

            hook(module, grad_input, grad_output) -> Tensor or None

        如果 module 有多个输入或输出的话, 那么 :attr:`grad_input` 和 :attr:`grad_output` 将会是个 tuple.
        hook 不应该修改它的参数, 但是它可以选择性地返回一个新的关于输入的梯度, 这个返回的梯度在后续的计算中会替代 :attr:`grad_input`.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                通过调用 ``handle.remove()`` 方法可以删除添加钩子的句柄
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def register_forward_pre_hook(self, hook):
        """在模块上注册一个预前向钩子.

        每一次在调用 :func:`forward` 函数前都会调用该钩子.
        它应该有以下结构::

            hook(module, input) -> None

        该钩子不应该修改输入.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                通过调用 ``handle.remove()`` 方法可以删除添加钩子的句柄
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        return handle

    def register_forward_hook(self, hook):
        r"""在模块上注册一个 forward hook（前向钩子）.

        每一次 :func:`forward` 函数计算出一个输出后, 该钩子将会被调用.
        它应该具有以下结构 ::

            hook(module, input, output) -> None

        该钩子应该不会修改输入或输出.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                通过调用 ``handle.remove()`` 方法可以删除添加钩子的句柄
        """
        handle = hooks.RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle

    def __call__(self, *input, **kwargs):
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        result = self.forward(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                raise RuntimeError(
                    "forward hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, Variable):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, Variable)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        return result

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_forward_pre_hooks' not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not torch.is_tensor(value):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """返回一个字典, 它包含整个模块的状态.

        包括参数和持久化的缓冲区 (例如. 运行中的平均值).
        Keys 是与之对应的参数和缓冲区的 name.

        当 keep_vars 为 ``True`` 时, 它为每一个参数（而不是一个张量）返回一个 Variable.

        Args:
            destination (dict, optional):
                如果不是 None, 该返回的字典应该被存储到 destination 中.
                Default: None
            prefix (string, optional):
                向结果字典中的每个参数和缓冲区的 key（名称）添加一个前缀.
                Default: ''
            keep_vars (bool, optional):
                如果为 ``True``, 为每一个参数返回一个 Variable.
                如果为 ``False``, 为每一个参数返回一个 Tensor.
                Default: ``False``

        Returns:
            dict:
                包含模块整体状态的字典

        Example:
            >>> module.state_dict().keys()
            ['bias', 'weight']
        """
        if destination is None:
            destination = OrderedDict()
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
        return destination

    def load_state_dict(self, state_dict, strict=True):
        """将 :attr:`state_dict` 中的 parameters 和 buffers 复制到此模块和它的子后代中.
        如果 :attr:`strict` 为 ``True``,  则 :attr:`state_dict` 的 key 必须和模块的 :func:`state_dict()` 函数返回的 key 一致.

        Arguments:
            state_dict (dict): 一个包含 parameters 和 persistent buffers（持久化缓存的）字典.
            strict (bool): 严格的强制 :attr:`state_dict` 属性中的 key 与该模块的函数 `:func:`state_dict()` 返回的 keys 相匹配.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def parameters(self):
        """返回一个模块参数的迭代器.

        这通常传递给优化器.

        Yields:
            Parameter: 模型参数

        Example:
            >>> for param in model.parameters():
            >>>     print(type(param.data), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
        """
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self, memo=None, prefix=''):
        """返回模块参数的迭代器, 产生参数的名称以及参数本身

        Yields:
            (string, Parameter): Tuple 包含名称很参数的 Tuple（元组）

        Example:
            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())
        """
        if memo is None:
            memo = set()
        for name, p in self._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_parameters(memo, submodule_prefix):
                yield name, p

    def _all_buffers(self, memo=None):
        if memo is None:
            memo = set()
        for name, b in self._buffers.items():
            if b is not None and b not in memo:
                memo.add(b)
                yield b
        for module in self.children():
            for b in module._all_buffers(memo):
                yield b

    def children(self):
        """返回一个最近子模块的 iterator（迭代器）.

        Yields:
            Module: 一个子模块
        """
        for name, module in self.named_children():
            yield module

    def named_children(self):
        """返回一个 iterator（迭代器）, 而不是最接近的子模块, 产生模块的 name 以及模块本身.

        Yields:
            (string, Module): 包含名称和子模块的 Tuple（元组）

        Example:
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self):
        """返回一个覆盖神经网络中所有模块的 iterator（迭代器）.

        Yields:
            Module: a module in the network

        Note:
            重复的模块只返回一次. 在下面的例子中, ``1`` 只会被返回一次.
            example, ``l`` will be returned only once.

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            >>>     print(idx, '->', m)
            0 -> Sequential (
              (0): Linear (2 -> 2)
              (1): Linear (2 -> 2)
            )
            1 -> Linear (2 -> 2)
        """
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):
        """返回一个神经网络中所有模块的 iterator（迭代器）, 产生模块的 name 以及模块本身.

        Yields:
            (string, Module): 名字和模块的 Tuple（元组）

        Note:
            重复的模块只返回一次. 在下面的例子中, ``1`` 只会被返回一次.

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            >>>     print(idx, '->', m)
            0 -> ('', Sequential (
              (0): Linear (2 -> 2)
              (1): Linear (2 -> 2)
            ))
            1 -> ('0', Linear (2 -> 2))
        """

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def train(self, mode=True):
        """设置模块为训练模式.

        这只对诸如 Dropout 或 BatchNorm 等模块时才会有影响.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        """将模块设置为评估模式.

        这种方式只对 Dropout 或 BatchNorm 等模块有效.
        """
        return self.train(False)

    def zero_grad(self):
        """将所有模型参数的梯度设置为零."""
        for p in self.parameters():
            if p.grad is not None:
                if p.grad.volatile:
                    p.grad.data.zero_()
                else:
                    data = p.grad.data
                    p.grad = Variable(data.new().resize_as_(data).zero_())

    def share_memory(self):
        return self._apply(lambda t: t.share_memory_())

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers
        return sorted(keys)
