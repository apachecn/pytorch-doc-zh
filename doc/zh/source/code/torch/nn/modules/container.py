from collections import OrderedDict
import string
import torch
import warnings
from .module import Module


class Container(Module):

    def __init__(self, **kwargs):
        super(Container, self).__init__()
        # DeprecationWarning is ignored by default <sigh>
        warnings.warn("nn.Container is deprecated. All of it's functionality "
                      "is now implemented in nn.Module. Subclass that instead.")
        for key, value in kwargs.items():
            self.add_module(key, value)


class Sequential(Module):
    r"""一个顺序的容器.
    模块将按照它们在构造函数中传递的顺序添加到它. 或者, 也可以传入模块的有序字典.

    为了更容易理解, 列举小例来说明 ::

        # 使用 Sequential 的例子
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # 与 OrderedDict 一起使用 Sequential 的例子
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class ModuleList(Module):
    r"""将子模块放入一个 list 中.

    ModuleList 可以像普通的 Python list 一样被索引, 但是它包含的模块已经被正确的注册了, 并且所有的 Module 方法都是可见的.

    Arguments:
        modules (list, optional): 要添加的模块列表

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            self += modules

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return self._modules[str(idx)]

    def __setitem__(self, idx, module):
        return setattr(self, str(idx), module)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def append(self, module):
        r"""添加一个指定的模块到 list 尾部.

        Arguments:
            module (nn.Module): 要被添加的模块
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        r"""在最后添加 Python list 中的模块.

        Arguments:
            modules (list): 要被添加的模块列表
        """
        if not isinstance(modules, list):
            raise TypeError("ModuleList.extend should be called with a "
                            "list, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


class ParameterList(Module):
    r"""保存 list 中的 parameter.

    ParameterList 可以像普通的 Python list 那样被索引, 但是它所包含的参数被正确的注册了, 并且所有的 Module 方法都可见的.

    Arguments:
        modules (list, optional): 要被添加的 :class:`~torch.nn.Parameter` 列表

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

            def forward(self, x):
                # ModuleList 可以充当 iterable（迭代器）, 或者可以使用整数进行索引
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, parameters=None):
        super(ParameterList, self).__init__()
        if parameters is not None:
            self += parameters

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return self._parameters[str(idx)]

    def __setitem__(self, idx, param):
        return self.register_parameter(str(idx), param)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def append(self, parameter):
        """添加一个指定的参数到 list 尾部.

        Arguments:
            parameter (nn.Parameter): parameter to append
        """
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self, parameters):
        """在最后添加 Python list 中的参数.

        Arguments:
            parameters (list): list of parameters to append
        """
        if not isinstance(parameters, list):
            raise TypeError("ParameterList.extend should be called with a "
                            "list, but got " + type(parameters).__name__)
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self
