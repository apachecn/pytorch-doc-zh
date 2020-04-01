# torch.random

> torch.random.fork_rng(devices=None, enabled=True, _caller='fork_rng', _devices_kw='devices')[[source]](_modules/torch/random.html#fork_rng)

    
福克斯的RNG，所以，当你返回时，RNG复位的状态，这是以前英寸

  参数

    * **devices** (可迭代CUDA编号） - CUDA设备针对其叉的RNG。 CPU RNG状态始终分叉。默认情况下， `fork_rng(） `运行在所有设备上，但会发出警告，如果你的机器有很多的设备，因为该功能将运行非常缓慢在这种情况下。如果您明确指定的设备，这个警告将被抑制

    * **enabled** ([bool](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果`假 `时，RNG没有分叉。这是很容易禁用上下文管理，而不必删除它，并在它之下取消缩进Python代码便利的说法。


`torch.random.get_rng_state()`[[source]](_modules/torch/random.html#get_rng_state)

返回随机数发生器状态作为 torch.ByteTensor 。


`torch.random.initial_seed()`[[source]](_modules/torch/random.html#initial_seed)

返回初始种子用于产生随机数作为一个Python 长。


`torch.random.manual_seed(seed)`[[source]](_modules/torch/random.html#manual_seed)

设置生成随机数种子。返回 torch.Generator 对象。

  参数

    **seed** ([int](https://docs.python.org/3/library/functions.html#int） - 所需的种子。

`torch.random.seed()`[[source]](_modules/torch/random.html#seed)

设置用于产生随机数，以非确定性的随机数种子。返回用于播种RNG一个64位的数。


`torch.random.set_rng_state(new_state)`[[source]](_modules/torch/random.html#set_rng_state)

设置随机数生成器的状态。

  参数

    **new_state**(torch.ByteTensor) - 期望状态


## 随机数发生器

`torch.random.get_rng_state()`[[source]](_modules/torch/random.html#get_rng_state)

Returns the random number generator state as a torch.ByteTensor.


`torch.random.set_rng_state(new_state)`[[source]](_modules/torch/random.html#set_rng_state)

Sets the random number generator state.

  Parameters

    **new_state** ( _torch.ByteTensor_ ) – The desired state


`torch.random.manual_seed(seed)`[[source]](_modules/torch/random.html#manual_seed)

Sets the seed for generating random numbers. Returns a torch.Generator object.

  Parameters
    **seed** ([ _int_](https://docs.python.org/3/library/functions.html#int)) – The desired seed.

`torch.random.seed()`[[source]](_modules/torch/random.html#seed)

Sets the seed for generating random numbers to a non-deterministic random
number. Returns a 64 bit number used to seed the RNG.

`torch.random.initial_seed()`[[source]](_modules/torch/random.html#initial_seed)


Returns the initial seed for generating random numbers as a Python long.

`torch.random.fork_rng(devices=None, enabled=True, _caller='fork_rng', _devices_kw='devices')`[[source]](_modules/torch/random.html#fork_rng)

Forks the RNG, so that when you return, the RNG is reset to the state that it
was previously in.

  Parameters
    * **devices** ( _iterable of CUDA IDs_ ) – CUDA devices for which to fork the RNG. CPU RNG state is always forked. By default, `fork_rng()`operates on all devices, but will emit a warning if your machine has a lot of devices, since this function will run very slowly in that case. If you explicitly specify devices, this warning will be suppressed

    * **enabled** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – if `False`, the RNG is not forked. This is a convenience argument for easily disabling the context manager without having to delete it and unindent your Python code under it.
