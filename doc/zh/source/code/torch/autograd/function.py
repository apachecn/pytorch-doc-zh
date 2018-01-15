import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._six import with_metaclass
import functools
from collections import OrderedDict


class _ContextMethodMixin(object):

    def save_for_backward(self, *tensors):
        """将传入的 tensor 保存起来供函数 :func:`~Function.backward` 使用.

        ** 这个方法至多只能被调用一次, 只能在 **
        :func:`forward` **method.** 中使用.

        之后,被保存的张量可以通过 :attr:`saved_tensors` 
        属性获取; 或者, 如果变量还需要被使用 (比如二次求导), 能够被参数 :attr:`saved_variables` 获取.
        保证这些 tensor 没有被 in-place operations 修改过.

        参数可以被设置为 ``None``.
        """
        self.to_save = tensors

    def mark_dirty(self, *args):
        """将输入的 tensors 标记为被 in-place operation 修改过.
        **这个方法应当至多调用一次, 只能在方法 **
        :func:`forward` ** 中用, 实参只能是 forward 的实参.**

        每个在 forward 方法中被 in-place operations 修改的 tensor 都应该传递给这个方法.
        这样,可以保证检查的正确性.这个方法在 tensor 修改前后调用都可以.
        """
        self.dirty_tensors = args

    def mark_shared_storage(self, *pairs):
        """Marks that given pairs of distinct tensors are sharing storage.

        **这个方法应当至多调用一次, 只能在方法 **
        :func:`forward` ** 中用,所有的参数应该是元祖形式
         (input, output).**

        如果一些 inputs 和 outputs 是共享存储空间的,所有的这样的 (input, output)对都应该传给这个函数,
        保证 in-place operations 检查的正确性.唯一的特例就是,当 output 和 input 是同一个tensor(in-place operations 的输入和输出).
        这种情况下,就没必要指定它们之间的依赖关系,因为这个很容易就能推断出来.

        这个函数在很多时候都用不到,主要是用在索引 和 转置 这类的 op 中.
        """
        self.shared_pairs = pairs

    def mark_non_differentiable(self, *args):
        """将输出标记为不可微.

        **这个方法至多只能被调用一次,只能在方法**
        :func:`forward` ** 中用, 而且实参只能是 forward 的返回值.**

        这个方法会将输出标记成不可微,会增加 backward 过程中的效率.在 backward 中,
        你依旧需要接收 forward 输出值的梯度,但是这些梯度一直是 None.

        这用于例如对于从最大值返回的索引 :class:`Function`.
        """
        self.non_differentiable = args


class _HookMixin(object):

    @staticmethod
    def _register_hook(backward_hooks, hook):
        if backward_hooks is None:
            backward_hooks = OrderedDict()
        handle = hooks.RemovableHandle(backward_hooks)
        backward_hooks[handle.id] = hook
        return backward_hooks, handle


class BackwardCFunction(_C._FunctionBase, _ContextMethodMixin, _HookMixin):
    _is_legacy = False

    def apply(self, *args):
        return self._forward_cls.backward(self, *args)


class FunctionMeta(type):
    """Function metaclass.

    This metaclass sets up the following properties:
        _is_legacy: True if forward is not defined as a static method.
        _backward_cls: The Function class corresponding to the differentiated
            version of this function (which is generated on the fly by this
            metaclass).
    """

    def __init__(cls, name, bases, attrs):
        for super_cls in cls.mro():
            forward = super_cls.__dict__.get('forward')
            if forward is not None:
                has_static_forward = isinstance(forward, staticmethod) or isinstance(forward, classmethod)
                break

        setattr(cls, '_is_legacy', not has_static_forward)

        # old-style functions
        if not has_static_forward:
            return super(FunctionMeta, cls).__init__(name, bases, attrs)

        backward_fn = type(name + 'Backward', (BackwardCFunction,), {'_forward_cls': cls})
        setattr(cls, '_backward_cls', backward_fn)

        return super(FunctionMeta, cls).__init__(name, bases, attrs)


class Function(with_metaclass(FunctionMeta, _C._FunctionBase, _ContextMethodMixin, _HookMixin)):
    """记录操作历史记录并定义区分操作的方法.

     每个执行在 Varaibles 上的 operation 都会创建一个 Function 对象,这个 Function 对象执行计算工作,同时记录下来.这个历史以有向无环图的形式保存下来,
     有向图的节点为 functions ,有向图的边代表数据依赖关系 (input<-output).之后,当 backward 被调用的时候,计算图以拓扑顺序处理,通过调用每个 Function 对象的 backward(),
     同时将返回的梯度传递给下一个 Function.

    通常情况下,用户能和 Functions 交互的唯一方法就是创建 Function 的子类,定义新的 operation. 这是扩展 torch.autograd 的推荐方法.

    每个 Function 只被使用一次(在forward过程中).

    参数说明:
        requires_grad: 布尔类型依赖于方法 :func:`backward` 会不会还会被使用.

    比如::

        >>> class Exp(Function):
        >>>
        >>>     @staticmethod
        >>>     def forward(ctx, i):
        >>>         result = i.exp()
        >>>         ctx.save_for_backward(result)
        >>>         return result
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         result, = ctx.saved_variables
        >>>         return grad_output * result
    """

    # only for backward compatibility
    __call__ = _C._FunctionBase._do_forward

    # for the tracer
    is_traceable = False

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """进行操作.

        这个方法将会被继承他的所有子类覆盖.

        第一个参数为上下文参数,接下来可以输入任何张量或变量 (张量或其他类型).

        上下文可以用来存储可以在回传期间检索的变量.
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        """定义反向传播操作

        这个方法将会被继承他的所有子类覆盖.

        第一个参数为上下文参数, 接下来可以输入任何张量或变量 (张量或其他类型), 并且有多个返回值,
        并且为函数 :func:`forward` 的输入. 每个参数都是给定输出的导数, 并且每一个输出都是输入的导数.

        上下文可以用来检索转发过程中保存的变量.
        """
        raise NotImplementedError


def once_differentiable(fn):
    from .variable import Variable

    @functools.wraps(fn)
    def wrapper(ctx, *args):
        tensor_args = [arg.data if isinstance(arg, Variable) else arg
                       for arg in args]
        outputs = fn(ctx, *tensor_args)
        # XXX: this is only an approximation of these flags - there's no way
        # to figure out if fn didn't use ctx.saved_variables and as a result
        # some Variables might require grad, even if no args do.
        # Unfortunately, this leads to unexpected error messages ("no nodes
        # require computing gradients"), but I don't have a better idea.
        # These functions would raise an error in backward anyway.
        volatile = any(arg.volatile if isinstance(arg, Variable) else False
                       for arg in args)
        requires_grad = any(arg.requires_grad if isinstance(arg, Variable) else False
                            for arg in args)
        if volatile:
            def err_fn(*args):
                return args
            kwargs = {'volatile': True}
        else:
            err_fn = torch._C._functions.DelayedError(
                b"trying to differentiate twice a function that was marked"
                b"with @once_differentiable")
            kwargs = {'requires_grad': requires_grad}
        if not isinstance(outputs, tuple):
            var = Variable(outputs, **kwargs) if outputs is not None else None
            return err_fn(var)
        return err_fn(*[Variable(o, **kwargs) if o is not None else None
                      for o in outputs])
    return wrapper


def traceable(fn_cls):
    """标记函数为 JIT (即时编译).

    可追踪函数有其他限制 - 它们不能传递任何依赖于数据的值到后向（例如 Prod 传递输出,这使得它不可追踪）,
    并且它们的导数应该完全在 autograd 变量的所有情况下执行 （即使导数是不稳定的）.

    不要用这种装饰器. IT IS FOR INTERNAL USE ONLY AND SHOULD BE HANDLED WITH
    CARE (or can give incorrect results otherwise).
    """
    fn_cls.is_traceable = True
    return fn_cls


class InplaceFunction(Function):

    def __init__(self, inplace=False):
        super(InplaceFunction, self).__init__()
        self.inplace = inplace


def _nested_map(condition, fn):
    def _map(obj):
        if condition(obj):
            return fn(obj)
        elif obj is None:
            return None
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_map(x) for x in obj)
        else:
            raise ValueError("NestedIOFunction doesn't know how to process "
                             "an input object of type " + torch.typename(obj))
    return _map


def _iter_filter(condition):
    def _iter(obj):
        if condition(obj):
            yield obj
        elif obj is None:
            return
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                for var in _iter(o):
                    yield var
        else:
            raise ValueError("NestedIOFunction doesn't know how to process "
                             "an input object of type " + torch.typename(obj))
    return _iter


def _unflatten(input, proto):
    # unflatten a list or tuple input into a nested list/tuple structure
    # specified by proto
    def unflatten_helper(input, proto):
        res = []
        if not isinstance(proto, (list, tuple)):
            return input[0], input[1:]
        for e in proto:
            res_e, input = unflatten_helper(input, e)
            res.append(res_e)
        return type(proto)(res), input

    return unflatten_helper(input, proto)[0]


# Return suitable 'prototype' that doesn't hold
# references possibly big options from 'obj'
def _to_proto(obj):
    def helper(obj):
        if isinstance(obj, torch.autograd.Variable):
            return "HOLE"
        elif obj is None:
            return None
        elif isinstance(obj, (list, tuple)):
            type_ = type(obj)
            return type_(helper(o) for o in obj)
        else:
            raise ValueError("NestedIOFunction doesn't know how to process "
                             "an input object of type " + torch.typename(obj))
    return helper(obj)


_iter_variables = _iter_filter(lambda o: isinstance(o, torch.autograd.Variable))
_iter_tensors = _iter_filter(torch.is_tensor)
_iter_None_tensors = _iter_filter(lambda o: o is None or torch.is_tensor(o))
_map_variable_tensor = _nested_map(lambda o: isinstance(o, torch.autograd.Variable), lambda o: o.data)


class NestedIOFunction(Function):

    def _do_forward(self, *input):
        self._nested_input = input
        flat_input = tuple(_iter_variables(input))
        flat_output = super(NestedIOFunction, self)._do_forward(*flat_input)
        nested_output = self._nested_output
        nested_variables = _unflatten(flat_output, self._nested_output)
        return nested_variables

    def _do_backward(self, gradients, retain_variables):
        self.retain_variables = retain_variables
        result = super(NestedIOFunction, self)._do_backward(gradients, retain_variables)
        if not retain_variables:
            del self._nested_output
            del self._to_save_nested
        return result

    def backward(self, *gradients):
        nested_gradients = _unflatten(gradients, self._nested_output)
        result = self.backward_extended(*nested_gradients)
        return tuple(_iter_None_tensors(result))

    __call__ = _do_forward

    def forward(self, *args):
        nested_tensors = _map_variable_tensor(self._nested_input)
        result = self.forward_extended(*nested_tensors)
        del self._nested_input
        self._nested_output = result
        return tuple(_iter_tensors(result))

    def save_for_backward(self, *args):
        self.to_save = tuple(_iter_tensors(args))
        self._to_save_nested = args

    @property
    def saved_tensors(self):
        flat_tensors = super(NestedIOFunction, self).saved_tensors
        return _unflatten(flat_tensors, self._to_save_nested)

    def mark_dirty(self, *args, **kwargs):
        self.dirty_tensors = tuple(_iter_tensors((args, kwargs)))

    def mark_non_differentiable(self, *args, **kwargs):
        self.non_differentiable = tuple(_iter_tensors((args, kwargs)))

    def forward_extended(self, *input):
        raise NotImplementedError

    def backward_extended(self, *grad_output):
        raise NotImplementedError
