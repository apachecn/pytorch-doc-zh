"""
torch.autograd 提供了类和函数用来对任意标量函数进行求导.只需要对已有的代码进行微小的改变-只需要将所有的 tensors 包含在
 :class:`.Variable` 对象中即可.
"""
import torch
import warnings

from .variable import Variable
from .function import Function, NestedIOFunction
from .stochastic_function import StochasticFunction
from .gradcheck import gradcheck
from . import profiler

__all__ = ['Variable', 'Function', 'StochasticFunction', 'backward']


def _make_grads(outputs, grads, user_create_graph):
    if user_create_graph is not None:
        create_graph = user_create_graph
    else:
        create_graph = any(isinstance(grad, Variable) and not grad.volatile
                           for grad in grads)

    new_grads = []
    for out, grad in zip(outputs, grads):
        if isinstance(grad, Variable):
            new_grads.append(grad)
        elif torch.is_tensor(grad):
            new_grads.append(Variable(grad, volatile=not create_graph))
        elif grad is None:
            if out.requires_grad:
                if out.numel() != 1:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                data = out.data
                new_grads.append(
                    Variable(data.new().resize_as_(data).fill_(1), volatile=not create_graph))
            else:
                new_grads.append(None)
        else:
            raise TypeError("gradients can be either Tensors, Variables or None, but got " +
                            type(grad).__name__)
    return tuple(new_grads), create_graph


def backward(variables, grad_variables=None, retain_graph=None, create_graph=None, retain_variables=None):
    """给定图某一个的节点变量variables,计算对该变量求导的梯度和.

    计算图可以通过链式法则求导.如果任何 ``variables``
    都是非标量(比如 他们的 data 属性中有多个元素)并且需要求导, 那么此函数需要指定 ``grad_variables``.
    它的长度应该和variables的长度匹配,里面保存了相关 variable 的梯度 (对于不需要 gradient tensor 的 variable, 应制定为 None).

    此函数累积叶子节点 variables 计算的梯度 - 调用此函数之前应先将叶子节点 variables 梯度置零.

    参数说明:
    *    variables(Variable 列表): 被求微分的叶子节点.
    *    grad_variables ((Tensor,Variable or None)列表):对应 variable 的梯度. 任何张量将自动转换为变量除非
         ``create_graph`` 是 ``True``. 没有值可以被指定为标量变量或者不需要被求导. 如果没有值被所有的grad_variables接受, 那么该参数是可以被省略的.
    *    retain_graph (bool, 可选): 如果是 ``False``, 该图计算过的梯度被释放掉.注意的是, 几乎所有情况都设置为 ``True``.
        并不是必须的并且能够高效的计算. 将该 ``create_graph`` 参数值设置为默认即可.
    *    create_graph (bool, 可选): 如果是 ``True``, 将会建立一个梯度图, 用来求解高阶导数.
        默认为 ``False``, 除非 ``grad_variables`` 拥有不止一个易变的 Variable.
    """
    variables = (variables,) if isinstance(variables, Variable) else tuple(variables)

    if grad_variables is None:
        grad_variables = [None] * len(variables)
    elif isinstance(grad_variables, Variable) or torch.is_tensor(grad_variables):
        grad_variables = [grad_variables]
    else:
        grad_variables = list(grad_variables)

    grad_variables, create_graph = _make_grads(variables, grad_variables, create_graph)

    if retain_variables is not None:
        if retain_graph is not None:
            raise ValueError("only one of retain_graph and retain_variables can be specified")
        retain_graph = retain_variables
        warnings.warn("retain_variables option is deprecated and will be removed in 0.3. "
                      "Use retain_graph instead.")
    elif retain_graph is None:
        retain_graph = create_graph

    Variable._execution_engine.run_backward(
        variables, grad_variables, retain_graph)


def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=None,
         only_inputs=True, allow_unused=False):
    """计算并返回给定值的梯度的和.

    ``grad_outputs`` 是一个列表同时长度与 ``output`` 一样, 
    存放了预先计算 input 的梯度的和. 如果
    output 不需要被求导, 那么梯度将为 ``None``).
    当不需要派生图时,可以将梯度作为张量,或者作为变量,在这种情况下,图将被创建.

    如果参数 ``only_inputs`` 为 ``True``, 该方法将会返回给定输入的梯度值列表.如果为 ``False``, 那么遗留下来的所有叶子节点的梯度都会被计算, 被且会被列加到 ``.grad``
    参数中.

    参数说明:
    *    outputs (变量序列): 梯度函数的返回值.
    *    inputs (变量序列): 需要计算的梯度的输入 (并且不会被累加到 ``.grad`` 参数中).
    *    grad_outputs (张量或变量序列): 每一个输出的梯度.
            所有的张量都会变成变量并且是可变的除非参数 ``create_graph`` 为 ``True``. 没有值可以被指定为标量变量或者不需要变化的值.
            如果所有 grad_variabls 都可以接受 None 值,那么这个参数是可选的.
    *    retain_graph (bool, 可选): 如果是 ``False``, 用于计算 grad 的图将被释放. 几乎所有情况都设置为 ``True``.
            并不是必须的并且能够高效地运行. 默认与 ``create_graph`` 参数一样.
    *    create_graph (bool, 可选): 如果是 ``True``, 梯度图将会被建立,用来求解高阶导数.
            默认为 ``False``, 除非参数 ``grad_variables`` 包含不只一个变量.
    *    only_inputs (bool, 可选): 如果是 ``True``, 叶子节点的导数将会在图中, 但是不会出现在参数 ``inputs`` 也不会被计算以及累加. 默认为 ``True``.
    *    allow_unused (bool, 可选): 如果是 ``False``, 指定计算输出时未使用的输入（因此它们的 grad 始终为零）是错误的. 默认为 ``False``.
    """

    outputs = (outputs,) if isinstance(outputs, Variable) else tuple(outputs)
    inputs = (inputs,) if isinstance(inputs, Variable) else tuple(inputs)
    if grad_outputs is None:
        grad_outputs = [None] * len(outputs)
    elif isinstance(grad_outputs, Variable) or torch.is_tensor(grad_outputs):
        grad_outputs = [grad_outputs]
    else:
        grad_outputs = list(grad_outputs)

    grad_outputs, create_graph = _make_grads(outputs, grad_outputs, create_graph)
    if retain_graph is None:
        retain_graph = create_graph

    return Variable._execution_engine.run_backward(
        outputs, grad_outputs, retain_graph,
        inputs, only_inputs, allow_unused)

if not torch._C._autograd_init():
    raise RuntimeError("autograd initialization failed")
