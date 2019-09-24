# torchaudio教程

PyTorch是一个开源的深度学习平台，提供给生产部署从研究原型的无缝路径与GPU的支持。

在解决机器学习问题显著的努力进入数据准备。
torchaudio利用PyTorch的GPU支持，并提供了许多工具，使数据加载容易，更具可读性。在本教程中，我们将看到如何从一个简单的数据集加载和数据预处理。

在本教程中，请确保`matplotlib`安装包，方便的可视化。

    import torch
    import torchaudio
    import matplotlib.pyplot as plt


## 打开一个数据集

torchaudio支持加载在WAV和MP3格式的声音文件。我们称波形的最终原始音频信号。

    
    
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
    

## 变换

torchaudio支持[变换](https://pytorch.org/audio/transforms.html)越来越多了。

  * **重新取样** ：重新取样波形到不同的采样率。
  * **谱图** ：建立从波形的频谱。
  * **MelScale** ：这接通正常STFT成梅尔频率STFT，使用转换矩阵。
  * **AmplitudeToDB** ：可打开的频谱从功率/幅度刻度分贝标度。
  * **MFCC** ：创建从波形的梅尔频率倒谱系数。
  * **MelSpectrogram** ：从使用PyTorch的STFT函数的波形创建MEL频谱图。
  * **MuLawEncoding** ：基于μ律压扩编码波形。
  * **MuLawDecoding** ：解码μ律编码波形。

由于所有的变换是nn.Modules或jit.ScriptModules，它们可以用作在任意点的神经网络的一部分。

首先，我们可以看看日志对数标度频谱图。

    
    
    specgram = torchaudio.transforms.Spectrogram()(waveform)
    
    print("Shape of spectrogram: {}".format(specgram.size()))
    
    plt.figure()
    plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')
    

![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_002.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_002.png)

Out:

    Shape of spectrogram: torch.Size([2, 201, 1385])
    

或者，我们可以看看梅尔谱图对数尺度。

    specgram = torchaudio.transforms.MelSpectrogram()(waveform)
    
    print("Shape of spectrogram: {}".format(specgram.size()))
    
    plt.figure()
    p = plt.imshow(specgram.log2()[0,:,:].detach().numpy(), cmap='gray')
    

![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_003.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_003.png)

Out:
    
    Shape of spectrogram: torch.Size([2, 128, 1385])
    

我们可以重新取样的波形，一次一个通道。

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
    

作为变革的另一个例子，我们可以编码基于Mu律enconding信号。但要做到这一点，我们需要的信号为-1到1之间。由于张量仅仅是一个普通PyTorch张量，我们可以把它应用标准的运营商。

    # Let's check if the tensor is in the interval [-1,1]
    print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean()))
    

Out:

    Min of waveform: -0.572845458984375
    Max of waveform: 0.575958251953125
    Mean of waveform: 9.293758921558037e-05
    

由于波形已经是-1到1之间，我们不需要正常化它。

    def normalize(tensor):
        # Subtract the mean, and scale to the interval [-1,1]
        tensor_minusmean = tensor - tensor.mean()
        return tensor_minusmean/tensor_minusmean.abs().max()
    
    # Let's normalize to the full interval [-1,1]
    # waveform = normalize(waveform)
    

让我们看看用编码波形。

    transformed = torchaudio.transforms.MuLawEncoding()(waveform)
    
    print("Shape of transformed waveform: {}".format(transformed.size()))
    
    plt.figure()
    plt.plot(transformed[0,:].numpy())


![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_005.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_005.png)

Out:

    Shape of transformed waveform: torch.Size([2, 276858])
    

而现在进行解码。

    reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)
    
    print("Shape of recovered waveform: {}".format(reconstructed.size()))
    
    plt.figure()
    plt.plot(reconstructed[0,:].numpy())
    

![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_006.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_006.png)

Out:

    Shape of recovered waveform: torch.Size([2, 276858])
    

我们终于可以比较其重建版本的原始波形。

    # Compute median relative difference
    err = ((waveform-reconstructed).abs() / waveform.abs()).median()
    
    print("Median relative difference between original and MuLaw reconstucted signals: {:.2%}".format(err))
    

Out:
    
    Median relative difference between original and MuLaw reconstucted signals: 1.28%
    

## 从移植到Kaldi torchaudio

用户可能熟悉[ Kaldi ](http://github.com/kaldi-asr/kaldi)，用于语音识别的工具包。
torchaudio提供兼容性与它在`torchaudio.kaldi_io`。它可以从kaldi SCP，或方舟文件确实读取或流：

  * read_vec_int_ark
  * read_vec_flt_scp
  * read_vec_flt_arkfile /流
  * read_mat_scp
  * read_mat_ark

torchaudio提供Kaldi兼容变换为`谱图 `和`fbank`与GPU支持的益处，参见[这里[HTG9用于更多信息。](compliance.kaldi.html)

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

我们也支持从波形计算滤波器功能，匹配Kaldi的实现。
    
    fbank = torchaudio.compliance.kaldi.fbank(waveform, **params)
    
    print("Shape of fbank: {}".format(fbank.size()))
    
    plt.figure()
    plt.imshow(fbank.t().numpy(), cmap='gray')
    

![https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_008.png](https://pytorch.org/tutorials/_images/sphx_glr_audio_preprocessing_tutorial_008.png)

Out:

    Shape of fbank: torch.Size([1383, 23])

## 结论

我们使用的示例原始音频信号，或波形，以说明如何使用torchaudio打开音频文件，以及如何进行预处理和变换这样的波形。鉴于torchaudio是建立在PyTorch，这些技术可以被用来作为更先进的音频应用，如语音识别积木，同时充分利用GPU的。

**脚本的总运行时间：** （0分钟2.343秒）

[`Download Python source code:
audio_preprocessing_tutorial.py`](https://pytorch.org/tutorials/_downloads/5ffe15ce830e55b3a9e9c294d04ab41c/audio_preprocessing_tutorial.py)

[`Download Jupyter notebook:
audio_preprocessing_tutorial.ipynb`](https://pytorch.org/tutorials/_downloads/7303ce3181f4dbc9a50bc1ed5bb3218f/audio_preprocessing_tutorial.ipynb)
