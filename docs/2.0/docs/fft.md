# torch.fft [¶](#torch-fft "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/fft>
>
> 原始地址：<https://pytorch.org/docs/stable/fft.html>


 离散傅立叶变换和相关函数。


## 快速傅立叶变换 [¶](#fast-fourier-transforms "此标题的固定链接")


|  |  |
| --- | --- |
| [`fft`](生成/torch.fft.fft.html#torch.fft.fft "torch.fft.fft") | 计算 `input` 的一维离散傅立叶变换。 |
| [`ifft`](生成/torch.fft.ifft.html#torch.fft.ifft "torch.fft.ifft") | 计算 `input` 的一维离散傅立叶逆变换。 |
| [`fft2`](生成/torch.fft.fft2.html#torch.fft.fft2 "torch.fft.fft2") | 计算 `input` 的二维离散傅里叶变换。 |
| [`ifft2`](生成/torch.fft.ifft2.html#torch.fft.ifft2 "torch.fft.ifft2") | 计算 `input` 的二维离散傅里叶逆变换。 |
| [`fftn`](生成/torch.fft.fftn.html#torch.fft.fftn "torch.fft.fftn") | 计算 `input` 的 N 维离散傅里叶变换。 |
| [`ifftn`](生成/torch.fft.ifftn.html#torch.fft.ifftn "torch.fft.ifftn") | 计算 `input` 的 N 维离散傅里叶逆变换。 |
| [`rfft`](生成/torch.fft.rfft.html#torch.fft.rfft "torch.fft.rfft") | 计算实值 `input` 的一维傅里叶变换。 |
| [`irfft`](生成/torch.fft.irfft.html#torch.fft.irfft "torch.fft.irfft") | 计算 [`rfft()`]( generated/torch.fft.rfft.html#torch.fft.rfft "torch.fft.rfft") 的逆。 |
| [`rfft2`](生成/torch.fft.rfft2.html#torch.fft.rfft2 "torch.fft.rfft2") | 计算真实 `input` 的二维离散傅立叶变换。 |
| [`irfft2`](生成/torch.fft.irfft2.html#torch.fft.irfft2 "torch.fft.irfft2") | 计算 [`rfft2()`]( generated/torch.fft.rfft2.html#torch.fft.rfft2 "torch.fft.rfft2") 的逆。 |
| [`rfftn`](生成/torch.fft.rfftn.html#torch.fft.rfftn "torch.fft.rfftn") | 计算真实 `input` 的 N 维离散傅里叶变换。 |
| [`irfftn`](生成/torch.fft.irfftn.html#torch.fft.irfftn "torch.fft.irfftn") | 计算 [`rfftn()`]( generated/torch.fft.rfftn.html#torch.fft.rfftn "torch.fft.rfftn") 的逆。 |
| [`hfft`](生成/torch.fft.hfft.html#torch.fft.hfft "torch.fft.hfft") | 计算埃尔米特对称 `input` 信号的一维离散傅立叶变换。 |
| [`ihfft`](生成/torch.fft.ihfft.html#torch.fft.ihfft "torch.fft.ihfft") | 计算 [`hfft()`]( generated/torch.fft.hfft.html#torch.fft.hfft "torch.fft.hfft") 的逆。 |
| [`hfft2`](生成/torch.fft.hfft2.html#torch.fft.hfft2 "torch.fft.hfft2") | 计算埃尔米特对称 `input` 信号的二维离散傅里叶变换。 |
| [`ihfft2`](生成/torch.fft.ihfft2.html#torch.fft.ihfft2 "torch.fft.ihfft2") |  计算真实 `input` 的二维离散傅里叶逆变换。 |
| [`hfftn`](生成/torch.fft.hfftn.html#torch.fft.hfftn "torch.fft.hfftn") |  计算 Hermitian 对称 `input` 信号的 n 维离散傅立叶变换。 |
| [`ihfftn`](生成/torch.fft.ihfftn.html#torch.fft.ihfftn "torch.fft.ihfftn") |  计算真实 `input` 的 N 维离散傅里叶逆变换。 |


## 辅助函数 [¶](#helper-functions "此标题的永久链接")


|  |  |
| --- | --- |
| [`fftfreq`](生成/torch.fft.fftfreq.html#torch.fft.fftfreq "torch.fft.fftfreq") | 计算大小为“n”的信号的离散傅立叶变换采样频率。 |
| [`rfftfreq`](生成/torch.fft.rfftfreq.html#torch.fft.rfftfreq "torch.fft.rfftfreq") | 使用大小为 n 的信号计算 [`rfft()`]( generated/torch.fft.rfft.html#torch.fft.rfft "torch.fft.rfft") 的采样频率。 |
| [`fftshift`](生成/torch.fft.fftshift.html#torch.fft.fftshift“torch.fft.fftshift”)|对 [`fftn()`]( generated/torch.fft.fftn.html#torch.fft.fftn "torch.fft.fftn") 提供的 n 维 FFT 数据重新排序，以首先具有负频率项。 |
| [`ifftshift`](生成/torch.fft.ifftshift.html#torch.fft.ifftshift "torch.fft.ifftshift") |  [`fftshift()`]( generated/torch.fft.fftshift.html#torch.fft.fftshift "torch.fft.fftshift") 的逆。 |