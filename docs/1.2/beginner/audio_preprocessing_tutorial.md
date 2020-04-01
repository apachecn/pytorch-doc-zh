# torchaudio教程

> 译者：[片刻](https://github.com/jiangzhonglian)
> 
> 校验：[片刻](https://github.com/jiangzhonglian)

PyTorch是一个开源深度学习平台，提供了从研究原型到具有GPU支持的生产部署的无缝路径。

解决机器学习问题的巨大努力在于数据准备。torchaudio利用PyTorch的GPU支持，并提供许多工具来简化数据加载并使其更具可读性。在本教程中，我们将看到如何从简单的数据集中加载和预处理数据。

对于本教程，请确保`matplotlib`已安装该软件包, 以方便查看。

    import torch
    import torchaudio
    import matplotlib.pyplot as plt


## 打开数据集

torchaudio支持以wav和mp3格式加载声音文件。我们将波形称为原始音频信号。

    filename = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
    waveform, sample_rate = torchaudio.load(filename)

    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))
    
    plt.figure()
    plt.plot(waveform.t().numpy())

![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_001.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_001.png)

日期：

    Shape of waveform: torch.Size([2, 276858])
    Sample rate of waveform: 44100
    

转换
torchaudio支持越来越多的 [转换](https://pytorch.org/audio/transforms.html)

  * **Resample** ：将波形重采样为其他采样率。
  * **Spectrogram** ：根据波形创建频谱图。
  * **MelScale** ：使用转换矩阵将普通STFT转换为Mel频率STFT。
  * **AmplitudeToDB** ：这将频谱图从功率/振幅标度转换为分贝标度。
  * **MFCC** ：从波形创建梅尔频率倒谱系数。
  * **MelSpectrogram** ：使用PyTorch中的STFT功能从波形创建MEL频谱图。
  * **MuLawEncoding** ：基于mu-law压扩对波形进行编码。
  * **MuLawDecoding** ：解码mu-law编码的波形。

由于所有变换都是nn.Modules或jit.ScriptModules，因此它们可以随时用作神经网络的一部分。

首先，我们可以以对数刻度查看频谱图的对数。

    specgram = torchaudio.transforms.Spectrogram()(waveform)
    
    print("Shape of spectrogram: {}".format(specgram.size()))
    
    plt.figure()
    plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')
    

![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_002.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_002.png)

Out:

    Shape of spectrogram: torch.Size([2, 201, 1385])

或者我们可以以对数刻度查看梅尔光谱图。

    specgram = torchaudio.transforms.MelSpectrogram()(waveform)
    
    print("Shape of spectrogram: {}".format(specgram.size()))
    
    plt.figure()
    p = plt.imshow(specgram.log2()[0,:,:].detach().numpy(), cmap='gray')
    

![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_003.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_003.png)

Out:
    
    Shape of spectrogram: torch.Size([2, 128, 1385])
    

我们可以一次对一个通道重新采样波形。

    new_sample_rate = sample_rate/10
    
    # Since Resample applies to a single channel, we resample first channel here
    channel = 0
    transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))
    
    print("Shape of transformed waveform: {}".format(transformed.size()))
    
    plt.figure()
    plt.plot(transformed[0,:].numpy())
    

![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_004.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_004.png)

Out:
    
    Shape of transformed waveform: torch.Size([1, 27686])
    
作为变换的另一个示例，我们可以基于Mu-Law编码对信号进行编码。但是要这样做，我们需要信号在-1和1之间。由于张量只是常规的PyTorch张量，因此我们可以在其上应用标准运算符。

    # Let's check if the tensor is in the interval [-1,1]
    print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean()))
    

Out:

    Min of waveform: -0.572845458984375
    Max of waveform: 0.575958251953125
    Mean of waveform: 9.293758921558037e-05
    
由于波形已经在-1和1之间，因此我们不需要对其进行归一化。

    def normalize(tensor):
        # Subtract the mean, and scale to the interval [-1,1]
        tensor_minusmean = tensor - tensor.mean()
        return tensor_minusmean/tensor_minusmean.abs().max()
    
    # Let's normalize to the full interval [-1,1]
    # waveform = normalize(waveform)
    

让我们对波形进行编码。

    transformed = torchaudio.transforms.MuLawEncoding()(waveform)
    
    print("Shape of transformed waveform: {}".format(transformed.size()))
    
    plt.figure()
    plt.plot(transformed[0,:].numpy())


![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_005.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_005.png)

Out:

    Shape of transformed waveform: torch.Size([2, 276858])
    

现在解码。

    reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)
    
    print("Shape of recovered waveform: {}".format(reconstructed.size()))
    
    plt.figure()
    plt.plot(reconstructed[0,:].numpy())
    

![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_006.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_006.png)

Out:

    Shape of recovered waveform: torch.Size([2, 276858])
    

我们最终可以将原始波形与其重构版本进行比较。

    # Compute median relative difference
    err = ((waveform-reconstructed).abs() / waveform.abs()).median()
    
    print("Median relative difference between original and MuLaw reconstucted signals: {:.2%}".format(err))
    

Out:
    
    Median relative difference between original and MuLaw reconstucted signals: 1.28%
    

## 从Kaldi迁移到Torchaudio
用户可能熟悉 语音识别工具包[Kaldi](http://github.com/kaldi-asr/kaldi)。torchaudio在中提供与之的兼容性 `torchaudio.kaldi_io`。实际上，它可以通过以下方式从kaldi scp或ark文件或流中读取：

  * read_vec_int_ark
  * read_vec_flt_scp
  * read_vec_flt_arkfile /流
  * read_mat_scp
  * read_mat_ark

torchaudio为GPU提供支持 spectrogram 并 fbank受益于Kaldi兼容的转换，请参见[此处](compliance.kaldi.html)以获取更多信息。

    n_fft = 400.0
    frame_length = n_fft / sample_rate * 1000.0
    frame_shift = frame_length / 2.0
    
    params = {
        "channel": 0,
        "dither": 0.0,
        "window_type": "hanning",
        "frame_length": frame_length,
        "frame_shift": frame_shift,
        "remove_dc_offset": False,
        "round_to_power_of_two": False,
        "sample_frequency": sample_rate,
    }
    
    specgram = torchaudio.compliance.kaldi.spectrogram(waveform, **params)
    
    print("Shape of spectrogram: {}".format(specgram.size()))
    
    plt.figure()
    plt.imshow(specgram.t().numpy(), cmap='gray')
    

![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_007.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_007.png)

Out:

    Shape of spectrogram: torch.Size([1383, 201])

我们还支持根据波形计算滤波器组特征，与Kaldi的实现相匹配。
    
    fbank = torchaudio.compliance.kaldi.fbank(waveform, **params)
    
    print("Shape of fbank: {}".format(fbank.size()))
    
    plt.figure()
    plt.imshow(fbank.t().numpy(), cmap='gray')
    

![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_008.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_008.png)

Out:

    Shape of fbank: torch.Size([1383, 23])

## 结论

我们使用示例原始音频信号或波形来说明如何使用torchaudio打开音频文件，以及如何预处理和转换此类波形。鉴于torchaudio是基于PyTorch构建的，则这些技术可在利用GPU的同时用作更高级音频应用(例如语音识别）的构建块。

**脚本的总运行时间：** (0分钟2.343秒）

[`Download Python source code:
audio_preprocessing_tutorial.py`](https://pytorch.org/tutorials/_downloads/5ffe15ce830e55b3a9e9c294d04ab41c/audio_preprocessing_tutorial.py)

[`Download Jupyter notebook:
audio_preprocessing_tutorial.ipynb`](https://pytorch.org/tutorials/_downloads/7303ce3181f4dbc9a50bc1ed5bb3218f/audio_preprocessing_tutorial.ipynb)
