### Spectral Ops

```py
torch.fft(input, signal_ndim, normalized=False) → Tensor
```

Complex-to-complex Discrete Fourier Transform

This method computes the complex-to-complex discrete Fourier transform. Ignoring the batch dimensions, it computes the following expression:

![](img/5562db7e8cbd1caa591b84aef7d65178.jpg)

where ![](img/9566974d45a96737f7e0ecf302d877b8.jpg) = `signal_ndim` is number of dimensions for the signal, and ![](img/4236d8cccece7d17f3a004865adbf94d.jpg) is the size of signal dimension ![](img/31df9c730e19ca29b59dce64b99d98c1.jpg).

This method supports 1D, 2D and 3D complex-to-complex transforms, indicated by `signal_ndim`. `input` must be a tensor with last dimension of size 2, representing the real and imaginary components of complex numbers, and should have at least `signal_ndim + 1` dimensions with optionally arbitrary number of leading batch dimensions. If `normalized` is set to `True`, this normalizes the result by dividing it with ![](img/eee43fa49e4959712077ced4d4c25da3.jpg) so that the operator is unitary.

Returns the real and the imaginary parts together as one tensor of the same shape of `input`.

The inverse of this function is [`ifft()`](#torch.ifft "torch.ifft").

Note

For CUDA tensors, an LRU cache is used for cuFFT plans to speed up repeatedly running FFT methods on tensors of same geometry with same same configuration.

Changing `torch.backends.cuda.cufft_plan_cache.max_size` (default is 4096 on CUDA 10 and newer, and 1023 on older CUDA versions) controls the capacity of this cache. Some cuFFT plans may allocate GPU memory. You can use `torch.backends.cuda.cufft_plan_cache.size` to query the number of plans currently in cache, and `torch.backends.cuda.cufft_plan_cache.clear()` to clear the cache.

Warning

For CPU tensors, this method is currently only available with MKL. Use `torch.backends.mkl.is_available()` to check if MKL is installed.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor of at least `signal_ndim` `+ 1` dimensions
*   **signal_ndim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the number of dimensions in each signal. `signal_ndim` can only be 1, 2 or 3
*   **normalized** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether to return normalized results. Default: `False`


| Returns: | A tensor containing the complex-to-complex Fourier transform result |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

Example:

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

Complex-to-complex Inverse Discrete Fourier Transform

This method computes the complex-to-complex inverse discrete Fourier transform. Ignoring the batch dimensions, it computes the following expression:

![](img/014337971856a1c52bd1bc756aa262b6.jpg)

where ![](img/9566974d45a96737f7e0ecf302d877b8.jpg) = `signal_ndim` is number of dimensions for the signal, and ![](img/4236d8cccece7d17f3a004865adbf94d.jpg) is the size of signal dimension ![](img/31df9c730e19ca29b59dce64b99d98c1.jpg).

The argument specifications are almost identical with [`fft()`](#torch.fft "torch.fft"). However, if `normalized` is set to `True`, this instead returns the results multiplied by ![](img/363cf16ec847125aff9d3c88189beea7.jpg), to become a unitary operator. Therefore, to invert a [`fft()`](#torch.fft "torch.fft"), the `normalized` argument should be set identically for [`fft()`](#torch.fft "torch.fft").

Returns the real and the imaginary parts together as one tensor of the same shape of `input`.

The inverse of this function is [`fft()`](#torch.fft "torch.fft").

Note

For CUDA tensors, an LRU cache is used for cuFFT plans to speed up repeatedly running FFT methods on tensors of same geometry with same same configuration.

Changing `torch.backends.cuda.cufft_plan_cache.max_size` (default is 4096 on CUDA 10 and newer, and 1023 on older CUDA versions) controls the capacity of this cache. Some cuFFT plans may allocate GPU memory. You can use `torch.backends.cuda.cufft_plan_cache.size` to query the number of plans currently in cache, and `torch.backends.cuda.cufft_plan_cache.clear()` to clear the cache.

Warning

For CPU tensors, this method is currently only available with MKL. Use `torch.backends.mkl.is_available()` to check if MKL is installed.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor of at least `signal_ndim` `+ 1` dimensions
*   **signal_ndim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the number of dimensions in each signal. `signal_ndim` can only be 1, 2 or 3
*   **normalized** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether to return normalized results. Default: `False`


| Returns: | A tensor containing the complex-to-complex inverse Fourier transform result |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

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

Real-to-complex Discrete Fourier Transform

This method computes the real-to-complex discrete Fourier transform. It is mathematically equivalent with [`fft()`](#torch.fft "torch.fft") with differences only in formats of the input and output.

This method supports 1D, 2D and 3D real-to-complex transforms, indicated by `signal_ndim`. `input` must be a tensor with at least `signal_ndim` dimensions with optionally arbitrary number of leading batch dimensions. If `normalized` is set to `True`, this normalizes the result by dividing it with ![](img/eee43fa49e4959712077ced4d4c25da3.jpg) so that the operator is unitary, where ![](img/4236d8cccece7d17f3a004865adbf94d.jpg) is the size of signal dimension ![](img/31df9c730e19ca29b59dce64b99d98c1.jpg).

The real-to-complex Fourier transform results follow conjugate symmetry:

![](img/f5ea82a4989c8b43517e53dc795d5516.jpg)

where the index arithmetic is computed modulus the size of the corresponding dimension, ![](img/b57c343978b88489065e0d2443349ae0.jpg) is the conjugate operator, and ![](img/9566974d45a96737f7e0ecf302d877b8.jpg) = `signal_ndim`. `onesided` flag controls whether to avoid redundancy in the output results. If set to `True` (default), the output will not be full complex result of shape ![](img/d0cb8b79768b92645994f059e281aaca.jpg), where ![](img/28ec51e742166ea3400be6e7343bbfa5.jpg) is the shape of `input`, but instead the last dimension will be halfed as of size ![](img/baa9baa3b9c79a896640a3f7c20deb1c.jpg).

The inverse of this function is [`irfft()`](#torch.irfft "torch.irfft").

Note

For CUDA tensors, an LRU cache is used for cuFFT plans to speed up repeatedly running FFT methods on tensors of same geometry with same same configuration.

Changing `torch.backends.cuda.cufft_plan_cache.max_size` (default is 4096 on CUDA 10 and newer, and 1023 on older CUDA versions) controls the capacity of this cache. Some cuFFT plans may allocate GPU memory. You can use `torch.backends.cuda.cufft_plan_cache.size` to query the number of plans currently in cache, and `torch.backends.cuda.cufft_plan_cache.clear()` to clear the cache.

Warning

For CPU tensors, this method is currently only available with MKL. Use `torch.backends.mkl.is_available()` to check if MKL is installed.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor of at least `signal_ndim` dimensions
*   **signal_ndim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the number of dimensions in each signal. `signal_ndim` can only be 1, 2 or 3
*   **normalized** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether to return normalized results. Default: `False`
*   **onesided** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether to return half of results to avoid redundancy. Default: `True`


| Returns: | A tensor containing the real-to-complex Fourier transform result |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

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

Complex-to-real Inverse Discrete Fourier Transform

This method computes the complex-to-real inverse discrete Fourier transform. It is mathematically equivalent with [`ifft()`](#torch.ifft "torch.ifft") with differences only in formats of the input and output.

The argument specifications are almost identical with [`ifft()`](#torch.ifft "torch.ifft"). Similar to [`ifft()`](#torch.ifft "torch.ifft"), if `normalized` is set to `True`, this normalizes the result by multiplying it with ![](img/eee43fa49e4959712077ced4d4c25da3.jpg) so that the operator is unitary, where ![](img/4236d8cccece7d17f3a004865adbf94d.jpg) is the size of signal dimension ![](img/31df9c730e19ca29b59dce64b99d98c1.jpg).

Due to the conjugate symmetry, `input` do not need to contain the full complex frequency values. Roughly half of the values will be sufficient, as is the case when `input` is given by [`rfft()`](#torch.rfft "torch.rfft") with `rfft(signal, onesided=True)`. In such case, set the `onesided` argument of this method to `True`. Moreover, the original signal shape information can sometimes be lost, optionally set `signal_sizes` to be the size of the original signal (without the batch dimensions if in batched mode) to recover it with correct shape.

Therefore, to invert an [`rfft()`](#torch.rfft "torch.rfft"), the `normalized` and `onesided` arguments should be set identically for [`irfft()`](#torch.irfft "torch.irfft"), and preferrably a `signal_sizes` is given to avoid size mismatch. See the example below for a case of size mismatch.

See [`rfft()`](#torch.rfft "torch.rfft") for details on conjugate symmetry.

The inverse of this function is [`rfft()`](#torch.rfft "torch.rfft").

Warning

Generally speaking, the input of this function should contain values following conjugate symmetry. Note that even if `onesided` is `True`, often symmetry on some part is still needed. When this requirement is not satisfied, the behavior of [`irfft()`](#torch.irfft "torch.irfft") is undefined. Since [`torch.autograd.gradcheck()`](autograd.html#torch.autograd.gradcheck "torch.autograd.gradcheck") estimates numerical Jacobian with point perturbations, [`irfft()`](#torch.irfft "torch.irfft") will almost certainly fail the check.

Note

For CUDA tensors, an LRU cache is used for cuFFT plans to speed up repeatedly running FFT methods on tensors of same geometry with same same configuration.

Changing `torch.backends.cuda.cufft_plan_cache.max_size` (default is 4096 on CUDA 10 and newer, and 1023 on older CUDA versions) controls the capacity of this cache. Some cuFFT plans may allocate GPU memory. You can use `torch.backends.cuda.cufft_plan_cache.size` to query the number of plans currently in cache, and `torch.backends.cuda.cufft_plan_cache.clear()` to clear the cache.

Warning

For CPU tensors, this method is currently only available with MKL. Use `torch.backends.mkl.is_available()` to check if MKL is installed.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor of at least `signal_ndim` `+ 1` dimensions
*   **signal_ndim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the number of dimensions in each signal. `signal_ndim` can only be 1, 2 or 3
*   **normalized** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether to return normalized results. Default: `False`
*   **onesided** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether `input` was halfed to avoid redundancy, e.g., by [`rfft()`](#torch.rfft "torch.rfft"). Default: `True`
*   **signal_sizes** (list or `torch.Size`, optional) – the size of the original signal (without batch dimension). Default: `None`


| Returns: | A tensor containing the complex-to-real inverse Fourier transform result |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

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

Short-time Fourier transform (STFT).

Ignoring the optional batch dimension, this method computes the following expression:

![](img/b0732a4ca07b0a4bbb1c8e42755dd635.jpg)

where ![](img/20ddd8181c2e0d0fb893637e8572d475.jpg) is the index of the sliding window, and ![](img/fbd51655b696eb58cdc3e2a85d8138d3.jpg) is the frequency that ![](img/309b97f749e9d19a615ef66fdc686f9f.jpg). When `onesided` is the default value `True`,

*   `input` must be either a 1-D time sequence or a 2-D batch of time sequences.
*   If `hop_length` is `None` (default), it is treated as equal to `floor(n_fft / 4)`.
*   If `win_length` is `None` (default), it is treated as equal to `n_fft`.
*   `window` can be a 1-D tensor of size `win_length`, e.g., from [`torch.hann_window()`](#torch.hann_window "torch.hann_window"). If `window` is `None` (default), it is treated as if having ![](img/a3ea24a1f2a3549d3e5b0cacf3ecb7c7.jpg) everywhere in the window. If ![](img/6331c460bba556da5e58d69a99aa834e.jpg), `window` will be padded on both sides to length `n_fft` before being applied.
*   If `center` is `True` (default), `input` will be padded on both sides so that the ![](img/654b00d1036ba7f7d93e02f57fc00a75.jpg)-th frame is centered at time ![](img/fca268610b657c34819b28d98ed12232.jpg). Otherwise, the ![](img/654b00d1036ba7f7d93e02f57fc00a75.jpg)-th frame begins at time ![](img/fca268610b657c34819b28d98ed12232.jpg).
*   `pad_mode` determines the padding method used on `input` when `center` is `True`. See [`torch.nn.functional.pad()`](nn.html#torch.nn.functional.pad "torch.nn.functional.pad") for all available options. Default is `"reflect"`.
*   If `onesided` is `True` (default), only values for ![](img/fbd51655b696eb58cdc3e2a85d8138d3.jpg) in ![](img/960f7132c51380d407edbfffa1d01db2.jpg) are returned because the real-to-complex Fourier transform satisfies the conjugate symmetry, i.e., ![](img/5fce0cf9e58e0a0a71eb6931ade5e784.jpg).
*   If `normalized` is `True` (default is `False`), the function returns the normalized STFT results, i.e., multiplied by ![](img/207c21435d21c4bdec529a2f3b922bff.jpg).

Returns the real and the imaginary parts together as one tensor of size ![](img/a272f1ebb738ea28a82ccb05e4284b0d.jpg), where ![](img/28ec51e742166ea3400be6e7343bbfa5.jpg) is the optional batch size of `input`, ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) is the number of frequencies where STFT is applied, ![](img/5a047a5ca04e45726dba21b8302977da.jpg) is the total number of frames used, and each pair in the last dimension represents a complex number as the real part and the imaginary part.

Warning

This function changed signature at version 0.4.1\. Calling with the previous signature may cause error or return incorrect result.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **n_fft** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – size of Fourier transform
*   **hop_length** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – the distance between neighboring sliding window frames. Default: `None` (treated as equal to `floor(n_fft / 4)`)
*   **win_length** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – the size of window frame and STFT filter. Default: `None` (treated as equal to `n_fft`)
*   **window** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the optional window function. Default: `None` (treated as window of all ![](img/a3ea24a1f2a3549d3e5b0cacf3ecb7c7.jpg) s)
*   **center** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – whether to pad `input` on both sides so that the ![](img/654b00d1036ba7f7d93e02f57fc00a75.jpg)-th frame is centered at time ![](img/fca268610b657c34819b28d98ed12232.jpg). Default: `True`
*   **pad_mode** (_string__,_ _optional_) – controls the padding method used when `center` is `True`. Default: `"reflect"`
*   **normalized** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether to return the normalized STFT results Default: `False`
*   **onesided** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether to return half of results to avoid redundancy Default: `True`


| Returns: | A tensor containing the STFT result with shape described above |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

```py
torch.bartlett_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

Bartlett window function.

![](img/25485c0c544da57274d0f702ecfbec35.jpg)

where ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) is the full window size.

The input `window_length` is a positive integer controlling the returned window size. `periodic` flag determines whether the returned window trims off the last duplicate value from the symmetric window and is ready to be used as a periodic window with functions like [`torch.stft()`](#torch.stft "torch.stft"). Therefore, if `periodic` is true, the ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) in above formula is in fact ![](img/8ec5251e790c02993c5bb875e109ed2c.jpg). Also, we always have `torch.bartlett_window(L, periodic=True)` equal to `torch.bartlett_window(L + 1, periodic=False)[:-1])`.

Note

If `window_length` ![](img/b32b16dc37b80bf97e00ad0589be346b.jpg), the returned window contains a single value 1.

Parameters: 

*   **window_length** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the size of returned window
*   **periodic** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If True, returns a window to be used as periodic function. If False, return a symmetric window.
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). Only floating point types are supported.
*   **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned window tensor. Only `torch.strided` (dense layout) is supported.
*   **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). `device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
*   **requires_grad** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If autograd should record operations on the returned tensor. Default: `False`.


| Returns: | A 1-D tensor of size ![](img/8d7e55488941e3a1a5ac791b70ccda5f.jpg) containing the window |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

```py
torch.blackman_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

Blackman window function.

![](img/5677fb096f4cdf5ffeb86c4d3646067a.jpg)

where ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) is the full window size.

The input `window_length` is a positive integer controlling the returned window size. `periodic` flag determines whether the returned window trims off the last duplicate value from the symmetric window and is ready to be used as a periodic window with functions like [`torch.stft()`](#torch.stft "torch.stft"). Therefore, if `periodic` is true, the ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) in above formula is in fact ![](img/8ec5251e790c02993c5bb875e109ed2c.jpg). Also, we always have `torch.blackman_window(L, periodic=True)` equal to `torch.blackman_window(L + 1, periodic=False)[:-1])`.

Note

If `window_length` ![](img/b32b16dc37b80bf97e00ad0589be346b.jpg), the returned window contains a single value 1.

Parameters: 

*   **window_length** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the size of returned window
*   **periodic** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If True, returns a window to be used as periodic function. If False, return a symmetric window.
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). Only floating point types are supported.
*   **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned window tensor. Only `torch.strided` (dense layout) is supported.
*   **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). `device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
*   **requires_grad** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If autograd should record operations on the returned tensor. Default: `False`.


| Returns: | A 1-D tensor of size ![](img/8d7e55488941e3a1a5ac791b70ccda5f.jpg) containing the window |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

```py
torch.hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

Hamming window function.

![](img/6ef50ea7103d4d74ef7919bfc2edf193.jpg)

where ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) is the full window size.

The input `window_length` is a positive integer controlling the returned window size. `periodic` flag determines whether the returned window trims off the last duplicate value from the symmetric window and is ready to be used as a periodic window with functions like [`torch.stft()`](#torch.stft "torch.stft"). Therefore, if `periodic` is true, the ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) in above formula is in fact ![](img/8ec5251e790c02993c5bb875e109ed2c.jpg). Also, we always have `torch.hamming_window(L, periodic=True)` equal to `torch.hamming_window(L + 1, periodic=False)[:-1])`.

Note

If `window_length` ![](img/b32b16dc37b80bf97e00ad0589be346b.jpg), the returned window contains a single value 1.

Note

This is a generalized version of [`torch.hann_window()`](#torch.hann_window "torch.hann_window").

Parameters: 

*   **window_length** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the size of returned window
*   **periodic** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If True, returns a window to be used as periodic function. If False, return a symmetric window.
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). Only floating point types are supported.
*   **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned window tensor. Only `torch.strided` (dense layout) is supported.
*   **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). `device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
*   **requires_grad** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If autograd should record operations on the returned tensor. Default: `False`.


| Returns: | A 1-D tensor of size ![](img/8d7e55488941e3a1a5ac791b70ccda5f.jpg) containing the window |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

```py
torch.hann_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

Hann window function.

![](img/3574793e8ae7241ce6a84e44a219b914.jpg)

where ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) is the full window size.

The input `window_length` is a positive integer controlling the returned window size. `periodic` flag determines whether the returned window trims off the last duplicate value from the symmetric window and is ready to be used as a periodic window with functions like [`torch.stft()`](#torch.stft "torch.stft"). Therefore, if `periodic` is true, the ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) in above formula is in fact ![](img/8ec5251e790c02993c5bb875e109ed2c.jpg). Also, we always have `torch.hann_window(L, periodic=True)` equal to `torch.hann_window(L + 1, periodic=False)[:-1])`.

Note

If `window_length` ![](img/b32b16dc37b80bf97e00ad0589be346b.jpg), the returned window contains a single value 1.

Parameters: 

*   **window_length** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the size of returned window
*   **periodic** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If True, returns a window to be used as periodic function. If False, return a symmetric window.
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). Only floating point types are supported.
*   **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned window tensor. Only `torch.strided` (dense layout) is supported.
*   **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). `device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
*   **requires_grad** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If autograd should record operations on the returned tensor. Default: `False`.


| Returns: | A 1-D tensor of size ![](img/8d7e55488941e3a1a5ac791b70ccda5f.jpg) containing the window |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |
