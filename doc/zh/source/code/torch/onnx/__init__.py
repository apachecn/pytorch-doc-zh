# -*- coding: UTF-8 -*-
"""
torch.onnx 模块可以将模型导出成 ONNX IR 形式.被导出的模型可以通过 ONNX 库被重新导入,
然后转化为可以在其它的深度学习框架上运行的模型.
"""

import torch
import torch.jit
import torch.autograd
import torch.serialization
import re
import collections
import string
import json
import math
import contextlib
import numbers
import warnings
from torch._utils import _range
from torch._six import string_classes


@contextlib.contextmanager
def set_training(model, mode):
    """
    A context manager to temporarily set the training mode of 'model'
    to 'mode', resetting it when we exit the with-block.  A no-op if
    mode is None.
    """
    if mode is None:
        yield
        return
    old_mode = model.training
    if old_mode != mode:
        model.train(mode)
    try:
        yield
    finally:
        if old_mode != mode:
            model.train(old_mode)


def export(model, args, f, export_params=True, verbose=False, training=False):
    """
    将一个模型导出成 ONNX 格式.这个导出器为了得到模型运行的轨迹,会运行一次你的模型.同时,它不支持动态模型（如 RNN.）

    也可参考: :ref:`onnx-export`

    Args:
        model (torch.nn.Module): 将被导出模型.  
        args (tuple of arguments): 模型的输入, ``model(*args)`` 必须是对模型的有效调用.任何非变量参数将被硬编码到导出的模型中.任何变量参数都将按照它们在参数中出现的顺序,成为输出模型的输入.如果 args 是一个变量,相当于用该变量的一个元组来调用它.（注意:目前还不支持将关键参数传递给模型,如果需要,请联系我们.）  
        f: 一个类文件对象（必须实现返回文件描述的fileno）或一个包含文件名的字符串。一个二进制 Protobuf 将被写入这个文件.
        export_params (bool, default True): 如果指定,所有参数将被导出.如果要导出未经训练的模型,请将其设置为 False.在这种情况下, 导出的模型将首先将其所有参数作为参数, 顺序由 ``model.state_dict().values()`` 指定.  
        verbose (bool, default False): 如果指定,会打印出正在导出轨迹的调式描述. 
        training (bool, default False): 在训练模式下输出模型.目前, ONNX 只是作为导出模型的接口,所以你通常不需要将其设为 True.
    """
    _export(model, args, f, export_params, verbose, training)


def _optimize_trace(trace):
    torch._C._jit_pass_peephole(trace)
    torch._C._jit_pass_lint(trace)
    torch._C._jit_pass_onnx(trace)
    torch._C._jit_pass_lint(trace)
    torch._C._jit_pass_onnx_peephole(trace)
    torch._C._jit_pass_lint(trace)
    torch._C._jit_pass_dce(trace)
    torch._C._jit_pass_lint(trace)


def _trace(func, args, return_outs=False):
    # Special case for common case of passing a single Variable
    if isinstance(args, torch.autograd.Variable):
        args = (args, )

    trace, torch_out = torch.jit.trace(func, args)
    _optimize_trace(trace)
    if return_outs:
        return trace, torch_out
    return trace


def _export(model, args, f, export_params=True, verbose=False, training=False):
    # Special case for common case of passing a single Variable
    if isinstance(args, torch.autograd.Variable):
        args = (args, )

    # A basic sanity check: make sure the state_dict keys are the same
    # before and after running the model.  Fail fast!
    orig_state_dict_keys = model.state_dict().keys()

    # By default, training=False, which is good because running a model in
    # training mode could result in internal buffers getting updated, dropout
    # getting applied, etc.  If you really know what you're doing, you
    # can turn training=True (or None, to preserve whatever the original
    # training mode was.)
    with set_training(model, training):
        trace, torch_out = torch.jit.trace(model, args)

    if orig_state_dict_keys != model.state_dict().keys():
        raise RuntimeError("state_dict changed after running the tracer; "
                           "something weird is happening in your model!")

    _optimize_trace(trace)
    if verbose:
        print(trace)

    # TODO: Don't allocate a in-memory string for the protobuf
    from torch.onnx.symbolic import _onnx_opset_version
    if export_params:
        # NB: OrderedDict values is not actually a list, but trace.export is
        # not duck-typed and expects an actual list.
        proto = trace.export(list(model.state_dict().values()), _onnx_opset_version)
    else:
        proto = trace.export([], _onnx_opset_version)

    torch.serialization._with_file_like(f, "wb", lambda f: f.write(proto))
    return torch_out


attr_pattern = re.compile("^(.+)_([ifstgz])$")


def _run_symbolic_method(op_name, symbolic_fn, args):
    """
    This trampoline function gets invoked for every symbolic method
    call from C++.
    """
    try:
        return symbolic_fn(*args)
    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch
        # to symbolic_fn.  Otherwise, the backtrace will have the clues
        # you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name), )
        raise


def _add_attribute(node, key, value):
    """ initializes the right attribute based on type of value """
    m = attr_pattern.match(key)
    if m is None:
        raise IndexError((
            "Invalid attribute specifier '{}' names " +
            " must be suffixed with type, e.g. 'dim_i' or 'dims_i'").format(key))
    name, kind = m.group(1), m.group(2)
    if not isinstance(value, string_classes) and not torch.is_tensor(value) and isinstance(value, collections.Iterable):
        kind += "s"
    return getattr(node, kind + '_')(name, value)


def _newNode(g, opname, *args, **kwargs):
    n = g.create(opname, args)
    for k, v in sorted(kwargs.items()):
        _add_attribute(n, k, v)
    return n


def _graph_op(g, opname, *raw_args, **kwargs):
    """
    Create an ONNX operator 'opname', taking 'args' as inputs and attributes
    'kwargs'; returning the node representing the single output of this operator
    (see the `outputs` keyword argument for multi-return nodes).

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    This function is monkey-patched onto Graph.

    Arguments:
        opname (string): The ONNX operator name, e.g., `Abs` or `Add`.
        args (Node...): The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        kwargs: The attributes of the ONNX operator, with keys named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).
        outputs (int, optional):  The number of outputs this operator returns;
            by default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Node`, representing each output of the ONNX operator
            in positional.
    """
    outputs = kwargs.pop('outputs', 1)

    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    kwargs = dict((k, v) for k, v in kwargs.items() if v is not None)

    def const_if_tensor(arg):
        if isinstance(arg, torch._C.Node):
            return arg
        else:
            return g.op("Constant", value_z=arg)

    args = list(const_if_tensor(arg) for arg in raw_args)
    n = g.appendNode(_newNode(g, opname, *args, **kwargs))
    if outputs == 1:
        return n
    return tuple(g.appendNode(g.createSelect(n, i)) for i in _range(outputs))


# Note [Export inplace]
# ~~~~~~~~~~~~~~~~~~~~~
# In abstract, it would be better for us to export inplace annotations,
# than to not export them, since it is useful information that can
# help the target of an ONNX export export more efficiently.  However,
# ONNX doesn't currently formalize inplace.  Fortunately, it's sound to drop
# inplace annotations, but we are losing information this way.


def _run_symbolic_function(g, n, inputs):
    import torch.onnx.symbolic

    try:
        # See Note [Export inplace]
        if n.kind().endswith('_'):
            op_name = n.kind()[:-1]
        else:
            op_name = n.kind()
        if not hasattr(torch.onnx.symbolic, op_name):
            warnings.warn("ONNX export failed on {} because torch.onnx.symbolic.{} does not exist"
                          .format(op_name, op_name))
            return None
        fn = getattr(torch.onnx.symbolic, op_name)
        attrs = {k: n[k] for k in n.attributeNames()}
        return fn(g, *inputs, **attrs)

    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch.
        # Otherwise, the backtrace will have the clues you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name), )
        raise


def _graph_at(g, opname, *args, **kwargs):
    return g.op("ATen", *args, operator_s=opname, **kwargs)


# This helper function can create either constant tensor or constant scalar.
# If dims is None or 0 or [0], generate a 0-d tensor (scalar).
#
# TODO: We might not need this anymore, since most scalars now show up
# as tensors
def _graph_constant(g, value, dims, type, *args, **kwargs):
    assert isinstance(value, numbers.Number)
    assert type is not None
    isscalar = False
    if dims is None or dims == 0 or set(dims) == set([0]):
        dims = [1]
        isscalar = True
    type = type.lower()
    if type == "char":
        tensor = torch.CharTensor(*dims)
    elif type == "short":
        tensor = torch.ShortTensor(*dims)
    elif type == "int":
        tensor = torch.IntTensor(*dims)
    elif type == "long":
        tensor = torch.LongTensor(*dims)
    elif type == "half":
        tensor = torch.HalfTensor(*dims)
    elif type == "float":
        tensor = torch.FloatTensor(*dims)
    elif type == "double":
        tensor = torch.DoubleTensor(*dims)
    else:
        raise ValueError("Unknown type, type should be one of the following strings: "
                         "char, short, int, long, half, float, double")
    tensor.fill_(value)
    if isscalar:
        return g.op("Constant", *args, value_z=tensor, **kwargs)
    return g.op("Constant", *args, value_t=tensor, **kwargs)


def _node_getitem(self, k):
    """
    Accessor for attributes of a node which is polymorphic over
    return type.

    NB: This is monkey-patched onto Node.
    """
    sel = self.kindOf(k)
    return getattr(self, sel)(k)


torch._C.Graph.op = _graph_op
torch._C.Graph.at = _graph_at
torch._C.Graph.constant = _graph_constant
torch._C.Node.__getitem__ = _node_getitem
