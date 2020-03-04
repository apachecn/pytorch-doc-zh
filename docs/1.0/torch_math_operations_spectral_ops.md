

### 光谱行动

> 译者：[ApacheCN](https://github.com/apachecn)

```py
torch.fft(input, signal_ndim, normalized=False) → Tensor
```

复杂到复杂的离散傅立叶变换

该方法计算复数到复数的离散傅立叶变换。忽略批量维度，它计算以下表达式：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5562db7e8cbd1caa591b84aef7d65178.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5562db7e8cbd1caa591b84aef7d65178.jpg)

其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg) = `signal_ndim`是信号的维数， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/4236d8cccece7d17f3a004865adbf94d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/4236d8cccece7d17f3a004865adbf94d.jpg) 是信号维数 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/31df9c730e19ca29b59dce64b99d98c1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/31df9c730e19ca29b59dce64b99d98c1.jpg) 的大小。

该方法支持1D，2D和3D复杂到复合变换，由`signal_ndim`表示。 `input`必须是最后一个尺寸为2的张量，表示复数的实部和虚部，并且至少应具有`signal_ndim + 1`尺寸和任意数量的前导批量尺寸。如果`normalized`设置为`True`，则通过将其除以 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/eee43fa49e4959712077ced4d4c25da3.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/eee43fa49e4959712077ced4d4c25da3.jpg) 来将结果标准化，以便操作符是单一的。

将实部和虚部一起作为`input`的相同形状的一个张量返回。

该函数的反函数是 [`ifft()`](#torch.ifft "torch.ifft") 。

注意

对于CUDA张量，LRU高速缓存用于cuFFT计划，以加速在具有相同配置的相同几何的张量上重复运行FFT方法。

更改`torch.backends.cuda.cufft_plan_cache.max_size`(CUDA 10及更高版本上的默认值为4096，旧版本的CUDA上为1023）控制此缓存的容量。一些cuFFT计划可能会分配GPU内存。您可以使用`torch.backends.cuda.cufft_plan_cache.size`查询当前缓存中的计划数量，使用`torch.backends.cuda.cufft_plan_cache.clear()`清除缓存。

警告

对于CPU张量，此方法目前仅适用于MKL。使用`torch.backends.mkl.is_available()`检查是否安装了MKL。

参数：

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 至少`signal_ndim` `+ 1`维度的输入张量
*   **signal_ndim**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 每个信号中的维数。 `signal_ndim`只能是1,2或3
*   **归一化** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 控制是否返回归一化结果。默认值：`False`

| 返回： | 包含复数到复数傅立叶变换结果的张量 |
| --- | --- |
| 返回类型： | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

例：

```py
>>> # unbatched 2D FFT
>>> x = torch.randn(4, 3, 2)
>>> torch.fft(x, 2)
tensor([[[-0.0876,  1.7835],
 [-2.0399, -2.9754],
 [ 4.4773, -5.0119]],

 [[-1.5716,  2.7631],
 [-3.8846,  5.2652],
 [ 0.2046, -0.7088]],

 [[ 1.9938, -0.5901],
 [ 6.5637,  6.4556],
 [ 2.9865,  4.9318]],

 [[ 7.0193,  1.1742],
 [-1.3717, -2.1084],
 [ 2.0289,  2.9357]]])
>>> # batched 1D FFT
>>> torch.fft(x, 1)
tensor([[[ 1.8385,  1.2827],
 [-0.1831,  1.6593],
 [ 2.4243,  0.5367]],

 [[-0.9176, -1.5543],
 [-3.9943, -2.9860],
 [ 1.2838, -2.9420]],

 [[-0.8854, -0.6860],
 [ 2.4450,  0.0808],
 [ 1.3076, -0.5768]],

 [[-0.1231,  2.7411],
 [-0.3075, -1.7295],
 [-0.5384, -2.0299]]])
>>> # arbitrary number of batch dimensions, 2D FFT
>>> x = torch.randn(3, 3, 5, 5, 2)
>>> y = torch.fft(x, 2)
>>> y.shape
torch.Size([3, 3, 5, 5, 2])

```

```py
torch.ifft(input, signal_ndim, normalized=False) → Tensor
```

复数到复数的逆离散傅立叶变换

该方法计算复数到复数的离散傅里叶逆变换。忽略批量维度，它计算以下表达式：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/014337971856a1c52bd1bc756aa262b6.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/014337971856a1c52bd1bc756aa262b6.jpg)

where [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg) = `signal_ndim` is number of dimensions for the signal, and [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/4236d8cccece7d17f3a004865adbf94d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/4236d8cccece7d17f3a004865adbf94d.jpg) is the size of signal dimension [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/31df9c730e19ca29b59dce64b99d98c1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/31df9c730e19ca29b59dce64b99d98c1.jpg).

参数规范与 [`fft()`](#torch.fft "torch.fft") 几乎相同。但是，如果`normalized`设置为`True`，则返回结果乘以 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/363cf16ec847125aff9d3c88189beea7.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/363cf16ec847125aff9d3c88189beea7.jpg) ，成为单一运算符。因此，要反转 [`fft()`](#torch.fft "torch.fft") ，`normalized`参数应设置为 [`fft()`](#torch.fft "torch.fft") 相同。

Returns the real and the imaginary parts together as one tensor of the same shape of `input`.

该函数的反函数是 [`fft()`](#torch.fft "torch.fft") 。

Note

For CUDA tensors, an LRU cache is used for cuFFT plans to speed up repeatedly running FFT methods on tensors of same geometry with same same configuration.

Changing `torch.backends.cuda.cufft_plan_cache.max_size` (default is 4096 on CUDA 10 and newer, and 1023 on older CUDA versions) controls the capacity of this cache. Some cuFFT plans may allocate GPU memory. You can use `torch.backends.cuda.cufft_plan_cache.size` to query the number of plans currently in cache, and `torch.backends.cuda.cufft_plan_cache.clear()` to clear the cache.

Warning

For CPU tensors, this method is currently only available with MKL. Use `torch.backends.mkl.is_available()` to check if MKL is installed.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 至少`signal_ndim` `+ 1`维度的输入张量
*   **signal_ndim**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 每个信号中的维数。 `signal_ndim`只能是1,2或3
*   **归一化** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 控制是否返回归一化结果。默认值：`False`

| Returns: | 包含复数到复数逆傅立叶变换结果的张量 |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> x = torch.randn(3, 3, 2)
>>> x
tensor([[[ 1.2766,  1.3680],
 [-0.8337,  2.0251],
 [ 0.9465, -1.4390]],

 [[-0.1890,  1.6010],
 [ 1.1034, -1.9230],
 [-0.9482,  1.0775]],

 [[-0.7708, -0.8176],
 [-0.1843, -0.2287],
 [-1.9034, -0.2196]]])
>>> y = torch.fft(x, 2)
>>> torch.ifft(y, 2)  # recover x
tensor([[[ 1.2766,  1.3680],
 [-0.8337,  2.0251],
 [ 0.9465, -1.4390]],

 [[-0.1890,  1.6010],
 [ 1.1034, -1.9230],
 [-0.9482,  1.0775]],

 [[-0.7708, -0.8176],
 [-0.1843, -0.2287],
 [-1.9034, -0.2196]]])

```

```py
torch.rfft(input, signal_ndim, normalized=False, onesided=True) → Tensor
```

实对复离散傅立叶变换

该方法计算实数到复数的离散傅立叶变换。它在数学上等同于 [`fft()`](#torch.fft "torch.fft") ，仅在输入和输出的格式上有所不同。

该方法支持1D，2D和3D实对复变换，由`signal_ndim`表示。 `input`必须是具有至少`signal_ndim`尺寸的张量，可选择任意数量的前导批量。如果`normalized`设置为`True`，则通过将其除以 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/eee43fa49e4959712077ced4d4c25da3.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/eee43fa49e4959712077ced4d4c25da3.jpg) 来将结果标准化，以便操作符是单一的，其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/4236d8cccece7d17f3a004865adbf94d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/4236d8cccece7d17f3a004865adbf94d.jpg) 是信号的大小维 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/31df9c730e19ca29b59dce64b99d98c1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/31df9c730e19ca29b59dce64b99d98c1.jpg) 。

实对复傅里叶变换结果遵循共轭对称：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/f5ea82a4989c8b43517e53dc795d5516.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/f5ea82a4989c8b43517e53dc795d5516.jpg)

计算指数算术的模数是相应维数的大小， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b57c343978b88489065e0d2443349ae0.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b57c343978b88489065e0d2443349ae0.jpg) 是共轭算子， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg) = `signal_ndim`。 `onesided`标志控制是否避免输出结果中的冗余。如果设置为`True`(默认），输出将不是形状 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/d0cb8b79768b92645994f059e281aaca.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/d0cb8b79768b92645994f059e281aaca.jpg) 的完整复杂结果，其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg) 是`input`的形状，而是最后一个尺寸将是大小 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/baa9baa3b9c79a896640a3f7c20deb1c.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/baa9baa3b9c79a896640a3f7c20deb1c.jpg) 的一半。

该函数的反函数是 [`irfft()`](#torch.irfft "torch.irfft") 。

Note

For CUDA tensors, an LRU cache is used for cuFFT plans to speed up repeatedly running FFT methods on tensors of same geometry with same same configuration.

Changing `torch.backends.cuda.cufft_plan_cache.max_size` (default is 4096 on CUDA 10 and newer, and 1023 on older CUDA versions) controls the capacity of this cache. Some cuFFT plans may allocate GPU memory. You can use `torch.backends.cuda.cufft_plan_cache.size` to query the number of plans currently in cache, and `torch.backends.cuda.cufft_plan_cache.clear()` to clear the cache.

Warning

For CPU tensors, this method is currently only available with MKL. Use `torch.backends.mkl.is_available()` to check if MKL is installed.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 至少`signal_ndim`维度的输入张量
*   **signal_ndim**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 每个信号中的维数。 `signal_ndim`只能是1,2或3
*   **归一化** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 控制是否返回归一化结果。默认值：`False`
*   **单独** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 控制是否返回一半结果以避免冗余。默认值：`True`

| Returns: | 包含实数到复数傅立叶变换结果的张量 |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> x = torch.randn(5, 5)
>>> torch.rfft(x, 2).shape
torch.Size([5, 3, 2])
>>> torch.rfft(x, 2, onesided=False).shape
torch.Size([5, 5, 2])

```

```py
torch.irfft(input, signal_ndim, normalized=False, onesided=True, signal_sizes=None) → Tensor
```

复数到实数的逆离散傅立叶变换

该方法计算复数到实数的逆离散傅里叶变换。它在数学上等同于 [`ifft()`](#torch.ifft "torch.ifft") ，仅在输入和输出的格式上有所不同。

参数规范与 [`ifft()`](#torch.ifft "torch.ifft") 几乎相同。类似于 [`ifft()`](#torch.ifft "torch.ifft") ，如果`normalized`设置为`True`，则通过将其与 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/eee43fa49e4959712077ced4d4c25da3.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/eee43fa49e4959712077ced4d4c25da3.jpg) 相乘来使结果归一化，以便运算符是单一的，其中 [] ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/4236d8cccece7d17f3a004865adbf94d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/4236d8cccece7d17f3a004865adbf94d.jpg) 是信号维 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/31df9c730e19ca29b59dce64b99d98c1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/31df9c730e19ca29b59dce64b99d98c1.jpg) 的大小。

由于共轭对称性，`input`不需要包含完整的复频率值。大约一半的值就足够了， [`rfft()`](#torch.rfft "torch.rfft") `rfft(signal, onesided=True)`给出`input`的情况就足够了。在这种情况下，将此方法的`onesided`参数设置为`True`。此外，原始信号形状信息有时会丢失，可选地将`signal_sizes`设置为原始信号的大小(如果处于批处理模式，则没有批量维度）以正确的形状恢复它。

因此，要反转 [`rfft()`](#torch.rfft "torch.rfft") ，`normalized`和`onesided`参数应设置为 [`irfft()`](#torch.irfft "torch.irfft") 相同，并且最好给出`signal_sizes`以避免大小不匹配。有关尺寸不匹配的情况，请参阅下面的示例。

有关共轭对称性的详细信息，请参见 [`rfft()`](#torch.rfft "torch.rfft") 。

该函数的反函数是 [`rfft()`](#torch.rfft "torch.rfft") 。

Warning

一般而言，此函数的输入应包含共轭对称后的值。请注意，即使`onesided`为`True`，仍然需要对某些部分进行对称。当不满足此要求时， [`irfft()`](#torch.irfft "torch.irfft") 的行为未定义。由于 [`torch.autograd.gradcheck()`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/autograd.html#torch.autograd.gradcheck "torch.autograd.gradcheck") 估计具有点扰动的数值雅可比行列式， [`irfft()`](#torch.irfft "torch.irfft") 几乎肯定会失败。

Note

For CUDA tensors, an LRU cache is used for cuFFT plans to speed up repeatedly running FFT methods on tensors of same geometry with same same configuration.

Changing `torch.backends.cuda.cufft_plan_cache.max_size` (default is 4096 on CUDA 10 and newer, and 1023 on older CUDA versions) controls the capacity of this cache. Some cuFFT plans may allocate GPU memory. You can use `torch.backends.cuda.cufft_plan_cache.size` to query the number of plans currently in cache, and `torch.backends.cuda.cufft_plan_cache.clear()` to clear the cache.

Warning

For CPU tensors, this method is currently only available with MKL. Use `torch.backends.mkl.is_available()` to check if MKL is installed.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 至少`signal_ndim` `+ 1`维度的输入张量
*   **signal_ndim**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 每个信号中的维数。 `signal_ndim`只能是1,2或3
*   **归一化** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 控制是否返回归一化结果。默认值：`False`
*   **单独** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 控制`input`是否为半数以避免冗余，例如， [`rfft()`](#torch.rfft "torch.rfft") 。默认值：`True`
*   **signal_sizes** (列表或`torch.Size`，可选） - 原始信号的大小(无批量维度）。默认值：`None`

| Returns: | 包含复数到实数逆傅立叶变换结果的张量 |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> x = torch.randn(4, 4)
>>> torch.rfft(x, 2, onesided=True).shape
torch.Size([4, 3, 2])
>>>
>>> # notice that with onesided=True, output size does not determine the original signal size
>>> x = torch.randn(4, 5)

>>> torch.rfft(x, 2, onesided=True).shape
torch.Size([4, 3, 2])
>>>
>>> # now we use the original shape to recover x
>>> x
tensor([[-0.8992,  0.6117, -1.6091, -0.4155, -0.8346],
 [-2.1596, -0.0853,  0.7232,  0.1941, -0.0789],
 [-2.0329,  1.1031,  0.6869, -0.5042,  0.9895],
 [-0.1884,  0.2858, -1.5831,  0.9917, -0.8356]])
>>> y = torch.rfft(x, 2, onesided=True)
>>> torch.irfft(y, 2, onesided=True, signal_sizes=x.shape)  # recover x
tensor([[-0.8992,  0.6117, -1.6091, -0.4155, -0.8346],
 [-2.1596, -0.0853,  0.7232,  0.1941, -0.0789],
 [-2.0329,  1.1031,  0.6869, -0.5042,  0.9895],
 [-0.1884,  0.2858, -1.5831,  0.9917, -0.8356]])

```

```py
torch.stft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=True)
```

短时傅立叶变换(STFT）。

忽略可选批处理维度，此方法计算以下表达式：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b0732a4ca07b0a4bbb1c8e42755dd635.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b0732a4ca07b0a4bbb1c8e42755dd635.jpg)

其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg) 是滑动窗口的索引， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/fbd51655b696eb58cdc3e2a85d8138d3.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/fbd51655b696eb58cdc3e2a85d8138d3.jpg) 是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/309b97f749e9d19a615ef66fdc686f9f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/309b97f749e9d19a615ef66fdc686f9f.jpg) 的频率。当`onesided`是默认值`True`时，

*   `input`必须是1-D时间序列或2-D批时间序列。
*   如果`hop_length`为`None`(默认值），则视为等于`floor(n_fft / 4)`。
*   如果`win_length`为`None`(默认值），则视为等于`n_fft`。
*   `window`可以是尺寸`win_length`的1-D张量，例如来自 [`torch.hann_window()`](#torch.hann_window "torch.hann_window") 。如果`window`是`None`(默认值），则视为在窗口中的任何地方都有 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a3ea24a1f2a3549d3e5b0cacf3ecb7c7.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a3ea24a1f2a3549d3e5b0cacf3ecb7c7.jpg) 。如果 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/6331c460bba556da5e58d69a99aa834e.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/6331c460bba556da5e58d69a99aa834e.jpg) ，`window`将在施加之前在长度`n_fft`的两侧填充。
*   如果`center`为`True`(默认值），则`input`将在两侧填充，以便 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/654b00d1036ba7f7d93e02f57fc00a75.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/654b00d1036ba7f7d93e02f57fc00a75.jpg) 帧在 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/fca268610b657c34819b28d98ed12232.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/fca268610b657c34819b28d98ed12232.jpg) 时间居中。否则， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/654b00d1036ba7f7d93e02f57fc00a75.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/654b00d1036ba7f7d93e02f57fc00a75.jpg) - 帧在时间 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/fca268610b657c34819b28d98ed12232.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/fca268610b657c34819b28d98ed12232.jpg) 开始。
*   `pad_mode`确定`center`为`True`时`input`上使用的填充方法。有关所有可用选项，请参阅 [`torch.nn.functional.pad()`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/nn.html#torch.nn.functional.pad "torch.nn.functional.pad") 。默认为`"reflect"`。
*   如果`onesided`是`True`(默认值），则仅返回 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/960f7132c51380d407edbfffa1d01db2.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/960f7132c51380d407edbfffa1d01db2.jpg) 中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/fbd51655b696eb58cdc3e2a85d8138d3.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/fbd51655b696eb58cdc3e2a85d8138d3.jpg) 的值，因为实数到复数的傅里叶变换满足共轭对称性，即， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5fce0cf9e58e0a0a71eb6931ade5e784.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5fce0cf9e58e0a0a71eb6931ade5e784.jpg) 。
*   如果`normalized`是`True`(默认为`False`），则该函数返回标准化的STFT结果，即乘以 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/207c21435d21c4bdec529a2f3b922bff.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/207c21435d21c4bdec529a2f3b922bff.jpg) 。

将实部和虚部一起作为一个尺寸 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a272f1ebb738ea28a82ccb05e4284b0d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a272f1ebb738ea28a82ccb05e4284b0d.jpg) 返回，其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg) 是`input`， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)的可选批量大小](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)是应用STFT的频率的数量， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5a047a5ca04e45726dba21b8302977da.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5a047a5ca04e45726dba21b8302977da.jpg) 是使用的帧的总数，并且最后维度中的每对表示作为实部和虚部的复数。

Warning

此功能在0.4.1版本上更改了签名。使用先前的签名调用可能会导致错误或返回错误的结果。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **n_fft**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 傅立叶变换的大小
*   **hop_length**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _可选_） - 相邻滑动窗口帧之间的距离。默认值：`None`(视为等于`floor(n_fft / 4)`）
*   **win_length**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _任选_） - 窗口框架和STFT过滤器的大小。默认值：`None`(视为等于`n_fft`）
*   **窗口** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _可选_） - 可选窗函数。默认值：`None`(被视为所有 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a3ea24a1f2a3549d3e5b0cacf3ecb7c7.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a3ea24a1f2a3549d3e5b0cacf3ecb7c7.jpg) s的窗口）
*   **中心** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 是否在两侧垫`input`使 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/654b00d1036ba7f7d93e02f57fc00a75.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/654b00d1036ba7f7d93e02f57fc00a75.jpg) 第一帧以时间 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/fca268610b657c34819b28d98ed12232.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/fca268610b657c34819b28d98ed12232.jpg) 为中心。默认值：`True`
*   **pad_mode**  (_string_ _，_ _可选_） - 控制`center`为`True`时使用的填充方法。默认值：`"reflect"`
*   **归一化** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 控制是否返回归一化STFT结果默认值：`False`
*   **单独** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 控制是否返回一半结果以避免冗余默认：`True`

| Returns: | 包含具有上述形状的STFT结果的张量 |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

```py
torch.bartlett_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

巴特利特的窗口功能。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/25485c0c544da57274d0f702ecfbec35.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/25485c0c544da57274d0f702ecfbec35.jpg)

其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) 是完整的窗口大小。

输入`window_length`是控制返回窗口大小的正整数。 `periodic`标志确定返回的窗口是否从对称窗口中删除最后一个重复值，并准备用作具有 [`torch.stft()`](#torch.stft "torch.stft") 等功能的周期窗口。因此，如果`periodic`为真，则上式中的 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) 实际上是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8ec5251e790c02993c5bb875e109ed2c.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8ec5251e790c02993c5bb875e109ed2c.jpg) 。此外，我们总是`torch.bartlett_window(L, periodic=True)`等于`torch.bartlett_window(L + 1, periodic=False)[:-1])`。

Note

如果`window_length` [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b32b16dc37b80bf97e00ad0589be346b.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b32b16dc37b80bf97e00ad0589be346b.jpg) ，则返回的窗口包含单个值1。

Parameters:

*   **window_length**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 返回窗口的大小
*   **周期性** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果为True，则返回一个窗口作为周期函数。如果为False，则返回对称窗口。
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。默认值：if `None`，使用全局默认值(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。仅支持浮点类型。
*   **布局** ([`torch.layout`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.layout "torch.torch.layout") ，可选） - 返回窗口张量的理想布局。仅支持`torch.strided`(密集布局）。
*   **设备** ([`torch.device`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.device "torch.torch.device") ，可选） - 返回张量的所需设备。默认值：如果`None`，则使用当前设备作为默认张量类型(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。 `device`将是CPU张量类型的CPU和CUDA张量类型的当前CUDA设备。
*   **requires_grad**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果autograd应该记录对返回张量的操作。默认值：`False`。

| Returns: | 含有窗口的1-D张量 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8d7e55488941e3a1a5ac791b70ccda5f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8d7e55488941e3a1a5ac791b70ccda5f.jpg) |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

```py
torch.blackman_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

布莱克曼窗口功能。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5677fb096f4cdf5ffeb86c4d3646067a.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5677fb096f4cdf5ffeb86c4d3646067a.jpg)

where [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) is the full window size.

输入`window_length`是控制返回窗口大小的正整数。 `periodic`标志确定返回的窗口是否从对称窗口中删除最后一个重复值，并准备用作具有 [`torch.stft()`](#torch.stft "torch.stft") 等功能的周期窗口。因此，如果`periodic`为真，则上式中的 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) 实际上是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8ec5251e790c02993c5bb875e109ed2c.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8ec5251e790c02993c5bb875e109ed2c.jpg) 。此外，我们总是`torch.blackman_window(L, periodic=True)`等于`torch.blackman_window(L + 1, periodic=False)[:-1])`。

Note

If `window_length` [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b32b16dc37b80bf97e00ad0589be346b.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b32b16dc37b80bf97e00ad0589be346b.jpg), the returned window contains a single value 1.

Parameters:

*   **window_length**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 返回窗口的大小
*   **周期性** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果为True，则返回一个窗口作为周期函数。如果为False，则返回对称窗口。
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。默认值：if `None`，使用全局默认值(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。仅支持浮点类型。
*   **布局** ([`torch.layout`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.layout "torch.torch.layout") ，可选） - 返回窗口张量的理想布局。仅支持`torch.strided`(密集布局）。
*   **设备** ([`torch.device`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.device "torch.torch.device") ，可选） - 返回张量的所需设备。默认值：如果`None`，则使用当前设备作为默认张量类型(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。 `device`将是CPU张量类型的CPU和CUDA张量类型的当前CUDA设备。
*   **requires_grad**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果autograd应该记录对返回张量的操作。默认值：`False`。

| Returns: | A 1-D tensor of size [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8d7e55488941e3a1a5ac791b70ccda5f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8d7e55488941e3a1a5ac791b70ccda5f.jpg) containing the window |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

```py
torch.hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

汉明窗功能。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/6ef50ea7103d4d74ef7919bfc2edf193.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/6ef50ea7103d4d74ef7919bfc2edf193.jpg)

where [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) is the full window size.

输入`window_length`是控制返回窗口大小的正整数。 `periodic`标志确定返回的窗口是否从对称窗口中删除最后一个重复值，并准备用作具有 [`torch.stft()`](#torch.stft "torch.stft") 等功能的周期窗口。因此，如果`periodic`为真，则上式中的 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) 实际上是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8ec5251e790c02993c5bb875e109ed2c.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8ec5251e790c02993c5bb875e109ed2c.jpg) 。此外，我们总是`torch.hamming_window(L, periodic=True)`等于`torch.hamming_window(L + 1, periodic=False)[:-1])`。

Note

If `window_length` [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b32b16dc37b80bf97e00ad0589be346b.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b32b16dc37b80bf97e00ad0589be346b.jpg), the returned window contains a single value 1.

Note

这是 [`torch.hann_window()`](#torch.hann_window "torch.hann_window") 的通用版本。

Parameters:

*   **window_length**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 返回窗口的大小
*   **周期性** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果为True，则返回一个窗口作为周期函数。如果为False，则返回对称窗口。
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。默认值：if `None`，使用全局默认值(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。仅支持浮点类型。
*   **布局** ([`torch.layout`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.layout "torch.torch.layout") ，可选） - 返回窗口张量的理想布局。仅支持`torch.strided`(密集布局）。
*   **设备** ([`torch.device`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.device "torch.torch.device") ，可选） - 返回张量的所需设备。默认值：如果`None`，则使用当前设备作为默认张量类型(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。 `device`将是CPU张量类型的CPU和CUDA张量类型的当前CUDA设备。
*   **requires_grad**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果autograd应该记录对返回张量的操作。默认值：`False`。

| Returns: | A 1-D tensor of size [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8d7e55488941e3a1a5ac791b70ccda5f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8d7e55488941e3a1a5ac791b70ccda5f.jpg) containing the window |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

```py
torch.hann_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

汉恩窗功能。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/3574793e8ae7241ce6a84e44a219b914.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/3574793e8ae7241ce6a84e44a219b914.jpg)

where [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) is the full window size.

输入`window_length`是控制返回窗口大小的正整数。 `periodic`标志确定返回的窗口是否从对称窗口中删除最后一个重复值，并准备用作具有 [`torch.stft()`](#torch.stft "torch.stft") 等功能的周期窗口。因此，如果`periodic`为真，则上式中的 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) 实际上是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8ec5251e790c02993c5bb875e109ed2c.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8ec5251e790c02993c5bb875e109ed2c.jpg) 。此外，我们总是`torch.hann_window(L, periodic=True)`等于`torch.hann_window(L + 1, periodic=False)[:-1])`。

Note

If `window_length` [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b32b16dc37b80bf97e00ad0589be346b.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b32b16dc37b80bf97e00ad0589be346b.jpg), the returned window contains a single value 1.

Parameters:

*   **window_length**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 返回窗口的大小
*   **周期性** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果为True，则返回一个窗口作为周期函数。如果为False，则返回对称窗口。
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。默认值：if `None`，使用全局默认值(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。仅支持浮点类型。
*   **布局** ([`torch.layout`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.layout "torch.torch.layout") ，可选） - 返回窗口张量的理想布局。仅支持`torch.strided`(密集布局）。
*   **设备** ([`torch.device`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.device "torch.torch.device") ，可选） - 返回张量的所需设备。默认值：如果`None`，则使用当前设备作为默认张量类型(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。 `device`将是CPU张量类型的CPU和CUDA张量类型的当前CUDA设备。
*   **requires_grad**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果autograd应该记录对返回张量的操作。默认值：`False`。

| Returns: | A 1-D tensor of size [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8d7e55488941e3a1a5ac791b70ccda5f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8d7e55488941e3a1a5ac791b70ccda5f.jpg) containing the window |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

