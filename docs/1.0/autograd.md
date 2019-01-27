

# Automatic differentiation package - torch.autograd

`torch.autograd` provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions. It requires minimal changes to the existing code - you only need to declare `Tensor` s for which gradients should be computed with the `requires_grad=True` keyword.

```py
torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None)
```

Computes the sum of gradients of given tensors w.r.t. graph leaves.

The graph is differentiated using the chain rule. If any of `tensors` are non-scalar (i.e. their data has more than one element) and require gradient, the function additionally requires specifying `grad_tensors`. It should be a sequence of matching length, that contains gradient of the differentiated function w.r.t. corresponding tensors (`None` is an acceptable value for all tensors that don’t need gradient tensors).

This function accumulates gradients in the leaves - you might need to zero them before calling it.

| Parameters: | 

*   **tensors** (_sequence of Tensor_) – Tensors of which the derivative will be computed.
*   **grad_tensors** (_sequence of_ _(_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_None_](https://docs.python.org/3/library/constants.html#None "(in Python v3.7)")_)_) – Gradients w.r.t. each element of corresponding tensors. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable for all grad_tensors, then this argument is optional.
*   **retain_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If `False`, the graph used to compute the grad will be freed. Note that in nearly all cases setting this option to `True` is not needed and often can be worked around in a much more efficient way. Defaults to the value of `create_graph`.
*   **create_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If `True`, graph of the derivative will be constructed, allowing to compute higher order derivative products. Defaults to `False`.

 |
| --- | --- |

```py
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)
```

Computes and returns the sum of gradients of outputs w.r.t. the inputs.

`grad_outputs` should be a sequence of length matching `output` containing the pre-computed gradients w.r.t. each of the outputs. If an output doesn’t require_grad, then the gradient can be `None`).

If `only_inputs` is `True`, the function will only return a list of gradients w.r.t the specified inputs. If it’s `False`, then gradient w.r.t. all remaining leaves will still be computed, and will be accumulated into their `.grad` attribute.

| Parameters: | 

*   **outputs** (_sequence of Tensor_) – outputs of the differentiated function.
*   **inputs** (_sequence of Tensor_) – Inputs w.r.t. which the gradient will be returned (and not accumulated into `.grad`).
*   **grad_outputs** (_sequence of Tensor_) – Gradients w.r.t. each output. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable for all grad_tensors, then this argument is optional. Default: None.
*   **retain_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If `False`, the graph used to compute the grad will be freed. Note that in nearly all cases setting this option to `True` is not needed and often can be worked around in a much more efficient way. Defaults to the value of `create_graph`.
*   **create_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If `True`, graph of the derivative will be constructed, allowing to compute higher order derivative products. Default: `False`.
*   **allow_unused** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If `False`, specifying inputs that were not used when computing outputs (and therefore their grad is always zero) is an error. Defaults to `False`.

 |
| --- | --- |

## Locally disabling gradient computation

```py
class torch.autograd.no_grad
```

Context-manager that disabled gradient calculation.

Disabling gradient calculation is useful for inference, when you are sure that you will not call `Tensor.backward()`. It will reduce memory consumption for computations that would otherwise have &lt;cite&gt;requires_grad=True&lt;/cite&gt;. In this mode, the result of every computation will have &lt;cite&gt;requires_grad=False&lt;/cite&gt;, even when the inputs have &lt;cite&gt;requires_grad=True&lt;/cite&gt;.

Also functions as a decorator.

Example:

```py
>>> x = torch.tensor([1], requires_grad=True)
>>> with torch.no_grad():
...   y = x * 2
>>> y.requires_grad
False
>>> @torch.no_grad()
... def doubler(x):
...     return x * 2
>>> z = doubler(x)
>>> z.requires_grad
False

```

```py
class torch.autograd.enable_grad
```

Context-manager that enables gradient calculation.

Enables gradient calculation inside a [`no_grad`](#torch.autograd.no_grad "torch.autograd.no_grad") context. This has no effect outside of [`no_grad`](#torch.autograd.no_grad "torch.autograd.no_grad").

Also functions as a decorator.

Example:

```py
>>> x = torch.tensor([1], requires_grad=True)
>>> with torch.no_grad():
...   with torch.enable_grad():
...     y = x * 2
>>> y.requires_grad
True
>>> y.backward()
>>> x.grad
>>> @torch.enable_grad()
... def doubler(x):
...     return x * 2
>>> with torch.no_grad():
...     z = doubler(x)
>>> z.requires_grad
True

```

```py
class torch.autograd.set_grad_enabled(mode)
```

Context-manager that sets gradient calculation to on or off.

`set_grad_enabled` will enable or disable grads based on its argument `mode`. It can be used as a context-manager or as a function.

| Parameters: | **mode** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – Flag whether to enable grad (`True`), or disable (`False`). This can be used to conditionally enable gradients. |
| --- | --- |

Example:

```py
>>> x = torch.tensor([1], requires_grad=True)
>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...   y = x * 2
>>> y.requires_grad
False
>>> torch.set_grad_enabled(True)
>>> y = x * 2
>>> y.requires_grad
True
>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False

```

## In-place operations on Tensors

Supporting in-place operations in autograd is a hard matter, and we discourage their use in most cases. Autograd’s aggressive buffer freeing and reuse makes it very efficient and there are very few occasions when in-place operations actually lower memory usage by any significant amount. Unless you’re operating under heavy memory pressure, you might never need to use them.

### In-place correctness checks

All `Tensor` s keep track of in-place operations applied to them, and if the implementation detects that a tensor was saved for backward in one of the functions, but it was modified in-place afterwards, an error will be raised once backward pass is started. This ensures that if you’re using in-place functions and not seeing any errors, you can be sure that the computed gradients are correct.

## Variable (deprecated)

Warning

The Variable API has been deprecated: Variables are no longer necessary to use autograd with tensors. Autograd automatically supports Tensors with `requires_grad` set to `True`. Below please find a quick guide on what has changed:

*   `Variable(tensor)` and `Variable(tensor, requires_grad)` still work as expected, but they return Tensors instead of Variables.
*   `var.data` is the same thing as `tensor.data`.
*   Methods such as `var.backward(), var.detach(), var.register_hook()` now work on tensors with the same method names.

In addition, one can now create tensors with `requires_grad=True` using factory methods such as [`torch.randn()`](torch.html#torch.randn "torch.randn"), [`torch.zeros()`](torch.html#torch.zeros "torch.zeros"), [`torch.ones()`](torch.html#torch.ones "torch.ones"), and others like the following:

`autograd_tensor = torch.randn((2, 3, 4), requires_grad=True)`

## Tensor autograd functions

```py
class torch.Tensor
```

```py
backward(gradient=None, retain_graph=None, create_graph=False)
```

Computes the gradient of current tensor w.r.t. graph leaves.

The graph is differentiated using the chain rule. If the tensor is non-scalar (i.e. its data has more than one element) and requires gradient, the function additionally requires specifying `gradient`. It should be a tensor of matching type and location, that contains the gradient of the differentiated function w.r.t. `self`.

This function accumulates gradients in the leaves - you might need to zero them before calling it.

| Parameters: | 

*   **gradient** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_None_](https://docs.python.org/3/library/constants.html#None "(in Python v3.7)")) – Gradient w.r.t. the tensor. If it is a tensor, it will be automatically converted to a Tensor that does not require grad unless `create_graph` is True. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable then this argument is optional.
*   **retain_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If `False`, the graph used to compute the grads will be freed. Note that in nearly all cases setting this option to True is not needed and often can be worked around in a much more efficient way. Defaults to the value of `create_graph`.
*   **create_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If `True`, graph of the derivative will be constructed, allowing to compute higher order derivative products. Defaults to `False`.

 |
| --- | --- |

```py
detach()
```

Returns a new Tensor, detached from the current graph.

The result will never require gradient.

Note

Returned Tensor uses the same data tensor as the original one. In-place modifications on either of them will be seen, and may trigger errors in correctness checks.

```py
detach_()
```

Detaches the Tensor from the graph that created it, making it a leaf. Views cannot be detached in-place.

```py
grad
```

This attribute is `None` by default and becomes a Tensor the first time a call to [`backward()`](#torch.Tensor.backward "torch.Tensor.backward") computes gradients for `self`. The attribute will then contain the gradients computed and future calls to [`backward()`](#torch.Tensor.backward "torch.Tensor.backward") will accumulate (add) gradients into it.

```py
is_leaf
```

All Tensors that have [`requires_grad`](#torch.Tensor.requires_grad "torch.Tensor.requires_grad") which is `False` will be leaf Tensors by convention.

For Tensors that have [`requires_grad`](#torch.Tensor.requires_grad "torch.Tensor.requires_grad") which is `True`, they will be leaf Tensors if they were created by the user. This means that they are not the result of an operation and so `grad_fn` is None.

Only leaf Tensors will have their [`grad`](#torch.Tensor.grad "torch.Tensor.grad") populated during a call to [`backward()`](#torch.Tensor.backward "torch.Tensor.backward"). To get [`grad`](#torch.Tensor.grad "torch.Tensor.grad") populated for non-leaf Tensors, you can use [`retain_grad()`](#torch.Tensor.retain_grad "torch.Tensor.retain_grad").

Example:

```py
>>> a = torch.rand(10, requires_grad=True)
>>> a.is_leaf
True
>>> b = torch.rand(10, requires_grad=True).cuda()
>>> b.is_leaf
False
# b was created by the operation that cast a cpu Tensor into a cuda Tensor
>>> c = torch.rand(10, requires_grad=True) + 2
>>> c.is_leaf
False
# c was created by the addition operation
>>> d = torch.rand(10).cuda()
>>> d.is_leaf
True
# d does not require gradients and so has no operation creating it (that is tracked by the autograd engine)
>>> e = torch.rand(10).cuda().requires_grad_()
>>> e.is_leaf
True
# e requires gradients and has no operations creating it
>>> f = torch.rand(10, requires_grad=True, device="cuda")
>>> f.is_leaf
True
# f requires grad, has not operation creating it

```

```py
register_hook(hook)
```

Registers a backward hook.

The hook will be called every time a gradient with respect to the Tensor is computed. The hook should have the following signature:

```py
hook(grad) -> Tensor or None

```

The hook should not modify its argument, but it can optionally return a new gradient which will be used in place of [`grad`](#torch.Tensor.grad "torch.Tensor.grad").

This function returns a handle with a method `handle.remove()` that removes the hook from the module.

Example:

```py
>>> v = torch.tensor([0., 0., 0.], requires_grad=True)
>>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
>>> v.backward(torch.tensor([1., 2., 3.]))
>>> v.grad

 2
 4
 6
[torch.FloatTensor of size (3,)]

>>> h.remove()  # removes the hook

```

```py
requires_grad
```

Is `True` if gradients need to be computed for this Tensor, `False` otherwise.

Note

The fact that gradients need to be computed for a Tensor do not mean that the [`grad`](#torch.Tensor.grad "torch.Tensor.grad") attribute will be populated, see [`is_leaf`](#torch.Tensor.is_leaf "torch.Tensor.is_leaf") for more details.

```py
retain_grad()
```

Enables .grad attribute for non-leaf Tensors.

## Function

```py
class torch.autograd.Function
```

Records operation history and defines formulas for differentiating ops.

Every operation performed on `Tensor` s creates a new function object, that performs the computation, and records that it happened. The history is retained in the form of a DAG of functions, with edges denoting data dependencies (`input &lt;- output`). Then, when backward is called, the graph is processed in the topological ordering, by calling [`backward()`](#torch.autograd.backward "torch.autograd.backward") methods of each [`Function`](#torch.autograd.Function "torch.autograd.Function") object, and passing returned gradients on to next [`Function`](#torch.autograd.Function "torch.autograd.Function") s.

Normally, the only way users interact with functions is by creating subclasses and defining new operations. This is a recommended way of extending torch.autograd.

Each function object is meant to be used only once (in the forward pass).

Examples:

```py
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
>>>         result, = ctx.saved_tensors
>>>         return grad_output * result

```

```py
static backward(ctx, *grad_outputs)
```

Defines a formula for differentiating the operation.

This function is to be overridden by all subclasses.

It must accept a context `ctx` as the first argument, followed by as many outputs did [`forward()`](#torch.autograd.Function.forward "torch.autograd.Function.forward") return, and it should return as many tensors, as there were inputs to [`forward()`](#torch.autograd.Function.forward "torch.autograd.Function.forward"). Each argument is the gradient w.r.t the given output, and each returned value should be the gradient w.r.t. the corresponding input.

The context can be used to retrieve tensors saved during the forward pass. It also has an attribute `ctx.needs_input_grad` as a tuple of booleans representing whether each input needs gradient. E.g., [`backward()`](#torch.autograd.backward "torch.autograd.backward") will have `ctx.needs_input_grad[0] = True` if the first input to [`forward()`](#torch.autograd.Function.forward "torch.autograd.Function.forward") needs gradient computated w.r.t. the output.

```py
static forward(ctx, *args, **kwargs)
```

Performs the operation.

This function is to be overridden by all subclasses.

It must accept a context ctx as the first argument, followed by any number of arguments (tensors or other types).

The context can be used to store tensors that can be then retrieved during the backward pass.

## Numerical gradient checking

```py
torch.autograd.gradcheck(func, inputs, eps=1e-06, atol=1e-05, rtol=0.001, raise_exception=True)
```

Check gradients computed via small finite differences against analytical gradients w.r.t. tensors in `inputs` that are of floating point type and with `requires_grad=True`.

The check between numerical and analytical gradients uses [`allclose()`](torch.html#torch.allclose "torch.allclose").

Note

The default values are designed for `input` of double precision. This check will likely fail if `input` is of less precision, e.g., `FloatTensor`.

Warning

If any checked tensor in `input` has overlapping memory, i.e., different indices pointing to the same memory address (e.g., from `torch.expand()`), this check will likely fail because the numerical gradients computed by point perturbation at such indices will change values at all other indices that share the same memory address.

| Parameters: | 

*   **func** (_function_) – a Python function that takes Tensor inputs and returns a Tensor or a tuple of Tensors
*   **inputs** (_tuple of Tensor_ _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – inputs to the function
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – perturbation for finite differences
*   **atol** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – absolute tolerance
*   **rtol** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – relative tolerance
*   **raise_exception** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – indicating whether to raise an exception if the check fails. The exception gives more information about the exact nature of the failure. This is helpful when debugging gradchecks.

 |
| --- | --- |
| Returns: | True if all differences satisfy allclose condition |
| --- | --- |

```py
torch.autograd.gradgradcheck(func, inputs, grad_outputs=None, eps=1e-06, atol=1e-05, rtol=0.001, gen_non_contig_grad_outputs=False, raise_exception=True)
```

Check gradients of gradients computed via small finite differences against analytical gradients w.r.t. tensors in `inputs` and `grad_outputs` that are of floating point type and with `requires_grad=True`.

This function checks that backpropagating through the gradients computed to the given `grad_outputs` are correct.

The check between numerical and analytical gradients uses [`allclose()`](torch.html#torch.allclose "torch.allclose").

Note

The default values are designed for `input` and `grad_outputs` of double precision. This check will likely fail if they are of less precision, e.g., `FloatTensor`.

Warning

If any checked tensor in `input` and `grad_outputs` has overlapping memory, i.e., different indices pointing to the same memory address (e.g., from `torch.expand()`), this check will likely fail because the numerical gradients computed by point perturbation at such indices will change values at all other indices that share the same memory address.

| Parameters: | 

*   **func** (_function_) – a Python function that takes Tensor inputs and returns a Tensor or a tuple of Tensors
*   **inputs** (_tuple of Tensor_ _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – inputs to the function
*   **grad_outputs** (_tuple of Tensor_ _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – The gradients with respect to the function’s outputs.
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – perturbation for finite differences
*   **atol** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – absolute tolerance
*   **rtol** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – relative tolerance
*   **gen_non_contig_grad_outputs** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – if `grad_outputs` is `None` and `gen_non_contig_grad_outputs` is `True`, the randomly generated gradient outputs are made to be noncontiguous
*   **raise_exception** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – indicating whether to raise an exception if the check fails. The exception gives more information about the exact nature of the failure. This is helpful when debugging gradchecks.

 |
| --- | --- |
| Returns: | True if all differences satisfy allclose condition |
| --- | --- |

## Profiler

Autograd includes a profiler that lets you inspect the cost of different operators inside your model - both on the CPU and GPU. There are two modes implemented at the moment - CPU-only using [`profile`](#torch.autograd.profiler.profile "torch.autograd.profiler.profile"). and nvprof based (registers both CPU and GPU activity) using [`emit_nvtx`](#torch.autograd.profiler.emit_nvtx "torch.autograd.profiler.emit_nvtx").

```py
class torch.autograd.profiler.profile(enabled=True, use_cuda=False)
```

Context manager that manages autograd profiler state and holds a summary of results.

| Parameters: | 

*   **enabled** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – Setting this to False makes this context manager a no-op. Default: `True`.
*   **use_cuda** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – Enables timing of CUDA events as well using the cudaEvent API. Adds approximately 4us of overhead to each tensor operation. Default: `False`

 |
| --- | --- |

Example

```py
>>> x = torch.randn((1, 1), requires_grad=True)
>>> with torch.autograd.profiler.profile() as prof:
...     y = x ** 2
...     y.backward()
>>> # NOTE: some columns were removed for brevity
... print(prof)
-------------------------------------  ---------------  ---------------
Name                                          CPU time        CUDA time
-------------------------------------  ---------------  ---------------
PowConstant                                  142.036us          0.000us
N5torch8autograd9GraphRootE                   63.524us          0.000us
PowConstantBackward                          184.228us          0.000us
MulConstant                                   50.288us          0.000us
PowConstant                                   28.439us          0.000us
Mul                                           20.154us          0.000us
N5torch8autograd14AccumulateGradE             13.790us          0.000us
N5torch8autograd5CloneE                        4.088us          0.000us

```

```py
export_chrome_trace(path)
```

Exports an EventList as a Chrome tracing tools file.

The checkpoint can be later loaded and inspected under `chrome://tracing` URL.

| Parameters: | **path** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – Path where the trace will be written. |
| --- | --- |

```py
key_averages()
```

Averages all function events over their keys.

| Returns: | An EventList containing FunctionEventAvg objects. |
| --- | --- |

```py
table(sort_by=None)
```

Prints an EventList as a nicely formatted table.

| Parameters: | **sort_by** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")_,_ _optional_) – Attribute used to sort entries. By default they are printed in the same order as they were registered. Valid keys include: `cpu_time`, `cuda_time`, `cpu_time_total`, `cuda_time_total`, `count`. |
| --- | --- |
| Returns: | A string containing the table. |
| --- | --- |

```py
total_average()
```

Averages all events.

| Returns: | A FunctionEventAvg object. |
| --- | --- |

```py
class torch.autograd.profiler.emit_nvtx(enabled=True)
```

Context manager that makes every autograd operation emit an NVTX range.

It is useful when running the program under nvprof:

```py
nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

```

Unfortunately, there’s no way to force nvprof to flush the data it collected to disk, so for CUDA profiling one has to use this context manager to annotate nvprof traces and wait for the process to exit before inspecting them. Then, either NVIDIA Visual Profiler (nvvp) can be used to visualize the timeline, or [`torch.autograd.profiler.load_nvprof()`](#torch.autograd.profiler.load_nvprof "torch.autograd.profiler.load_nvprof") can load the results for inspection e.g. in Python REPL.

| Parameters: | **enabled** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – Setting this to False makes this context manager a no-op. Default: `True`. |
| --- | --- |

Example

```py
>>> with torch.cuda.profiler.profile():
...     model(x) # Warmup CUDA memory allocator and profiler
...     with torch.autograd.profiler.emit_nvtx():
...         model(x)

```

**Forward-backward correlation**

When viewing a profile created using [`emit_nvtx`](#torch.autograd.profiler.emit_nvtx "torch.autograd.profiler.emit_nvtx") in the Nvidia Visual Profiler, correlating each backward-pass op with the corresponding forward-pass op can be difficult. To ease this task, [`emit_nvtx`](#torch.autograd.profiler.emit_nvtx "torch.autograd.profiler.emit_nvtx") appends sequence number information to the ranges it generates.

During the forward pass, each function range is decorated with `seq=&lt;N&gt;`. `seq` is a running counter, incremented each time a new backward Function object is created and stashed for backward. Thus, the &lt;cite&gt;seq=&lt;N&gt;&lt;/cite&gt; annotation associated with each forward function range tells you that if a backward Function object is created by this forward function, the backward object will receive sequence number N. During the backward pass, the top-level range wrapping each C++ backward Function’s `apply()` call is decorated with `stashed seq=&lt;M&gt;`. `M` is the sequence number that the backward object was created with. By comparing `stashed seq` numbers in backward with `seq` numbers in forward, you can track down which forward op created each backward Function.

Any functions executed during the backward pass are also decorated with `seq=&lt;N&gt;`. During default backward (with `create_graph=False`) this information is irrelevant, and in fact, `N` may simply be 0 for all such functions. Only the top-level ranges associated with backward Function objects’ `apply()` methods are useful, as a way to correlate these Function objects with the earlier forward pass.

**Double-backward**

If, on the other hand, a backward pass with `create_graph=True` is underway (in other words, if you are setting up for a double-backward), each function’s execution during backward is given a nonzero, useful `seq=&lt;N&gt;`. Those functions may themselves create Function objects to be executed later during double-backward, just as the original functions in the forward pass did. The relationship between backward and double-backward is conceptually the same as the relationship between forward and backward: The functions still emit current-sequence-number-tagged ranges, the Function objects they create still stash those sequence numbers, and during the eventual double-backward, the Function objects’ `apply()` ranges are still tagged with `stashed seq` numbers, which can be compared to &lt;cite&gt;seq&lt;/cite&gt; numbers from the backward pass.

```py
torch.autograd.profiler.load_nvprof(path)
```

Opens an nvprof trace file and parses autograd annotations.

| Parameters: | **path** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – path to nvprof trace |
| --- | --- |

## Anomaly detection

```py
class torch.autograd.detect_anomaly
```

Context-manager that enable anomaly detection for the autograd engine.

This does two things: - Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function. - Any backward computation that generate “nan” value will raise an error.

Example

```py
>>> import torch
>>> from torch import autograd
>>> class MyFunc(autograd.Function):
...     @staticmethod
...     def forward(ctx, inp):
...         return inp.clone()
...     @staticmethod
...     def backward(ctx, gO):
...         # Error during the backward pass
...         raise RuntimeError("Some error in backward")
...         return gO.clone()
>>> def run_fn(a):
...     out = MyFunc.apply(a)
...     return out.sum()
>>> inp = torch.rand(10, 10, requires_grad=True)
>>> out = run_fn(inp)
>>> out.backward()
 Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "/your/pytorch/install/torch/tensor.py", line 93, in backward
 torch.autograd.backward(self, gradient, retain_graph, create_graph)
 File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
 allow_unreachable=True)  # allow_unreachable flag
 File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
 return self._forward_cls.backward(self, *args)
 File "<stdin>", line 8, in backward
 RuntimeError: Some error in backward
>>> with autograd.detect_anomaly():
...     inp = torch.rand(10, 10, requires_grad=True)
...     out = run_fn(inp)
...     out.backward()
 Traceback of forward call that caused the error:
 File "tmp.py", line 53, in <module>
 out = run_fn(inp)
 File "tmp.py", line 44, in run_fn
 out = MyFunc.apply(a)
 Traceback (most recent call last):
 File "<stdin>", line 4, in <module>
 File "/your/pytorch/install/torch/tensor.py", line 93, in backward
 torch.autograd.backward(self, gradient, retain_graph, create_graph)
 File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
 allow_unreachable=True)  # allow_unreachable flag
 File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
 return self._forward_cls.backward(self, *args)
 File "<stdin>", line 8, in backward
 RuntimeError: Some error in backward

```

```py
class torch.autograd.set_detect_anomaly(mode)
```

Context-manager that sets the anomaly detection for the autograd engine on or off.

`set_detect_anomaly` will enable or disable the autograd anomaly detection based on its argument `mode`. It can be used as a context-manager or as a function.

See `detect_anomaly` above for details of the anomaly detection behaviour.

| Parameters: | **mode** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – Flag whether to enable anomaly detection (`True`), or disable (`False`). |
| --- | --- |

