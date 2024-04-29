# 音频数据增强 [¶](#audio-data-augmentation "此标题的固定链接")

> 译者：[龙琰](https://github.com/bellongyan)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/audio_data_augmentation_tutorial>
>
> 原始地址：<https://pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html>

**作者**: [Moto Hira](moto@meta.com)

`torchaudio` 提供了多种方法来增强音频数据。

在本教程中，我们研究了一种应用效果，滤波器，RIR（房间声学冲激响应）和编解码器的方法。

最后，我们通过电话从干净的语音中合成嘈杂的语音。

```python
import torch
import torchaudio
import torchaudio.functional as F

print(torch.__version__)
print(torchaudio.__version__)

import matplotlib.pyplot as plt
```

输出：

```python
2.3.0
2.3.0
```

## 准备工作

首先，我们导入模块并下载本教程中使用的音频资料。

```python
from IPython.display import Audio

from torchaudio.utils import download_asset

SAMPLE_WAV = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.wav")
SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
SAMPLE_SPEECH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")
SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
```

输出：

```python
  0%|          | 0.00/427k [00:00<?, ?B/s]
100%|##########| 427k/427k [00:00<00:00, 169MB/s]

  0%|          | 0.00/31.3k [00:00<?, ?B/s]
100%|##########| 31.3k/31.3k [00:00<00:00, 29.8MB/s]

  0%|          | 0.00/78.2k [00:00<?, ?B/s]
100%|##########| 78.2k/78.2k [00:00<00:00, 54.5MB/s]
```

## 应用效果和过滤

[`torchaudio.io.AudioEffector`](https://pytorch.org/audio/stable/generated/torchaudio.io.AudioEffector.html#torchaudio.io.AudioEffector) 允许直接将过滤器和编解码器应用于 Tensor 对象，与 `ffmpeg` 命令类似

`AudioEffector Usages <./effector_tutorial.html>` 解释了如何使用该类，详细内容请参考教程。

```python
# Load the data
waveform1, sample_rate = torchaudio.load(SAMPLE_WAV, channels_first=False)

# Define effects
effect = ",".join(
    [
        "lowpass=frequency=300:poles=1",  # apply single-pole lowpass filter
        "atempo=0.8",  # reduce the speed
        "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|delays=400:decays=0.3"
        # Applying echo gives some dramatic feeling
    ],
)


# Apply effects
def apply_effect(waveform, sample_rate, effect):
    effector = torchaudio.io.AudioEffector(effect=effect)
    return effector.apply(waveform, sample_rate)


waveform2 = apply_effect(waveform1, sample_rate, effect)

print(waveform1.shape, sample_rate)
print(waveform2.shape, sample_rate)
```

输出：

```python
torch.Size([109368, 2]) 44100
torch.Size([144642, 2]) 44100
```

请注意，应用效果后，帧数和通道数与原始帧数和通道数不同。 我们来听一下音频。

```python
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
```

```python
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, _ = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
```

### 原始音频

```python
plot_waveform(waveform1.T, sample_rate, title="Original", xlim=(-0.1, 3.2))
plot_specgram(waveform1.T, sample_rate, title="Original", xlim=(0, 3.04))
Audio(waveform1.T, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_001.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_001.png" alt="Original" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_002.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_002.png" alt="Original" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_01.wav">
    Your browser does not support the audio element.
</audio>

### 应用效果后的音频

```python
plot_waveform(waveform2.T, sample_rate, title="Effects Applied", xlim=(-0.1, 3.2))
plot_specgram(waveform2.T, sample_rate, title="Effects Applied", xlim=(0, 3.04))
Audio(waveform2.T, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_003.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_003.png" alt="Effects Applied" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_004.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_004.png" alt="Effects Applied" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_02.wav">
    Your browser does not support the audio element.
</audio>
## 模拟房间混响

[卷积混响（Convolution
reverb）](https://en.wikipedia.org/wiki/Convolution_reverb)是一种用于制造干净音频的技术，就像在不同的环境中产生的声音一样。

例如，使用房间声学冲激响应（RIR），我们可以使干净的语音听起来就像在会议室里说话一样。

对于这个过程，我们需要 RIR 数据。以下数据来自 VOiCES 数据集，但你也可以自己录制——只需打开麦克风并拍手。

```python
rir_raw, sample_rate = torchaudio.load(SAMPLE_RIR)
plot_waveform(rir_raw, sample_rate, title="Room Impulse Response (raw)")
plot_specgram(rir_raw, sample_rate, title="Room Impulse Response (raw)")
Audio(rir_raw, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_005.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_005.png" alt="Room Impulse Response (raw)" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_006.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_006.png" alt="Room Impulse Response (raw)" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_03.wav">
    Your browser does not support the audio element.
</audio>

首先，我们需要清理 RIR，提取主脉冲并将其归一化。

```python
rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
rir = rir / torch.linalg.vector_norm(rir, ord=2)

plot_waveform(rir, sample_rate, title="Room Impulse Response")
```

<img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_007.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_007.png" alt="Room Impulse Response" class="sphx-glr-single-img" width=661 height=331>

然后，使用 [`torchaudio.function.fftconvolve()`](https://pytorch.org/audio/stable/generated/torchaudio.functional.fftconvolve.html#torchaudio.functional.fftconvolve)，我们将语音信号与 RIR 进行卷积。

```python
speech, _ = torchaudio.load(SAMPLE_SPEECH)
augmented = F.fftconvolve(speech, rir)
```

### 原始音频

```python
plot_waveform(speech, sample_rate, title="Original")
plot_specgram(speech, sample_rate, title="Original")
Audio(speech, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_008.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_008.png" alt="Original" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_009.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_009.png" alt="Original" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_04.wav">
    Your browser does not support the audio element.
</audio>

### 经过 RIR 处理后的音频

```python
plot_waveform(augmented, sample_rate, title="RIR Applied")
plot_specgram(augmented, sample_rate, title="RIR Applied")
Audio(augmented, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_010.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_010.png" alt="RIR Applied" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_011.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_011.png" alt="RIR Applied" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_05.wav">
    Your browser does not support the audio element.
</audio>

## 添加背景噪声

为了在音频数据中引入背景噪声，我们可以根据期望的信噪比(SNR)[[wikipedia](https://en.wikipedia.org/wiki/Signal-to-noise_ratio)]在表示音频数据的张量中添加一个噪声张量，SNR 决定了音频数据相对于输出噪声的强度。

$$ \\mathrm{SNR} = \\frac{P*{signal}}{P*{noise}} $$

$$ \\mathrm{SNR*{dB}} = 10 \\log *{{10}} \\mathrm {SNR} $$

为了按 SNR 添加噪声到音频数据，我们使用 [`torchaudio.functional.add_noise`](https://pytorch.org/audio/stable/generated/torchaudio.functional.add_noise.html#torchaudio.functional.add_noise)。

```python
speech, _ = torchaudio.load(SAMPLE_SPEECH)
noise, _ = torchaudio.load(SAMPLE_NOISE)
noise = noise[:, : speech.shape[1]]

snr_dbs = torch.tensor([20, 10, 3])
noisy_speeches = F.add_noise(speech, noise, snr_dbs)
```

### 背景噪声

```python
plot_waveform(noise, sample_rate, title="Background noise")
plot_specgram(noise, sample_rate, title="Background noise")
Audio(noise, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_012.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_012.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_013.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_013.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_06.wav">
    Your browser does not support the audio element.
</audio>

### 信噪比 20 分贝（SNR 20 dB）

```python
snr_db, noisy_speech = snr_dbs[0], noisy_speeches[0:1]
plot_waveform(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
plot_specgram(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
Audio(noisy_speech, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_014.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_014.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_015.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_015.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_07.wav">
    Your browser does not support the audio element.
</audio>

### 信噪比 10 分贝（SNR 10 dB）

```python
snr_db, noisy_speech = snr_dbs[1], noisy_speeches[1:2]
plot_waveform(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
plot_specgram(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
Audio(noisy_speech, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_016.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_016.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_017.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_017.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_08.wav">
    Your browser does not support the audio element.
</audio>

### 信噪比 3 分贝（SNR 3 dB）

```python
snr_db, noisy_speech = snr_dbs[2], noisy_speeches[2:3]
plot_waveform(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
plot_specgram(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
Audio(noisy_speech, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_018.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_018.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_019.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_019.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_09.wav">
    Your browser does not support the audio element.
</audio>

## 将编解码器应用于 Tensor 对象

[`torchaudio.io.AudioEffector`](https://pytorch.org/audio/stable/generated/torchaudio.io.AudioEffector.html#torchaudio.io.AudioEffector) 也可以将编解码器应用于张量对象。

```python
waveform, sample_rate = torchaudio.load(SAMPLE_SPEECH, channels_first=False)


def apply_codec(waveform, sample_rate, format, encoder=None):
    encoder = torchaudio.io.AudioEffector(format=format, encoder=encoder)
    return encoder.apply(waveform, sample_rate)
```

### 原始音频

```python
plot_waveform(waveform.T, sample_rate, title="Original")
plot_specgram(waveform.T, sample_rate, title="Original")
Audio(waveform.T, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_020.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_020.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_021.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_021.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_10.wav">
    Your browser does not support the audio element.
</audio>

### 8 bit mu-law

```python
mulaw = apply_codec(waveform, sample_rate, "wav", encoder="pcm_mulaw")
plot_waveform(mulaw.T, sample_rate, title="8 bit mu-law")
plot_specgram(mulaw.T, sample_rate, title="8 bit mu-law")
Audio(mulaw.T, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_022.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_022.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_023.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_023.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_11.wav">
    Your browser does not support the audio element.
</audio>

### G.722

```python
g722 = apply_codec(waveform, sample_rate, "g722")
plot_waveform(g722.T, sample_rate, title="G.722")
plot_specgram(g722.T, sample_rate, title="G.722")
Audio(g722.T, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_024.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_024.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_025.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_025.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_12.wav">
    Your browser does not support the audio element.
</audio>

### Vorbis

```python
vorbis = apply_codec(waveform, sample_rate, "ogg", encoder="vorbis")
plot_waveform(vorbis.T, sample_rate, title="Vorbis")
plot_specgram(vorbis.T, sample_rate, title="Vorbis")
Audio(vorbis.T, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_026.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_026.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_027.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_027.png" alt="Background noise" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_13.wav">
    Your browser does not support the audio element.
</audio>

## 模拟电话录音

结合前面的技术，我们可以模拟声音，听起来就像一个人在有回声的房间里打电话，而背景中有人在说话。

```python
sample_rate = 16000
original_speech, sample_rate = torchaudio.load(SAMPLE_SPEECH)

plot_specgram(original_speech, sample_rate, title="Original")

# Apply RIR
rir_applied = F.fftconvolve(speech, rir)

plot_specgram(rir_applied, sample_rate, title="RIR Applied")

# Add background noise
# Because the noise is recorded in the actual environment, we consider that
# the noise contains the acoustic feature of the environment. Therefore, we add
# the noise after RIR application.
noise, _ = torchaudio.load(SAMPLE_NOISE)
noise = noise[:, : rir_applied.shape[1]]

snr_db = torch.tensor([8])
bg_added = F.add_noise(rir_applied, noise, snr_db)

plot_specgram(bg_added, sample_rate, title="BG noise added")

# Apply filtering and change sample rate
effect = ",".join(
    [
        "lowpass=frequency=4000:poles=1",
        "compand=attacks=0.02:decays=0.05:points=-60/-60|-30/-10|-20/-8|-5/-8|-2/-8:gain=-8:volume=-7:delay=0.05",
    ]
)

filtered = apply_effect(bg_added.T, sample_rate, effect)
sample_rate2 = 8000

plot_specgram(filtered.T, sample_rate2, title="Filtered")

# Apply telephony codec
codec_applied = apply_codec(filtered, sample_rate2, "g722")
plot_specgram(codec_applied.T, sample_rate2, title="G.722 Codec Applied")
```

<ul class="sphx-glr-horizontal">
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_028.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_028.png" alt="Original" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_029.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_029.png" alt="RIR Applied" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_030.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_030.png" alt="BG noise added" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_031.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_031.png" alt="Filtered" class="sphx-glr-multi-img" width=268 height=134></li>
<li><img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_032.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_data_augmentation_tutorial_032.png" alt="G.722 Codec Applied" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

### 原始音频

```python
Audio(original_speech, rate=sample_rate)
```

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_14.wav">
    Your browser does not support the audio element.
</audio>

### 经过 RIR 处理后的音频

```python
Audio(rir_applied, rate=sample_rate)
```

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_15.wav">
    Your browser does not support the audio element.
</audio>

### 添加背景噪声

```python
Audio(bg_added, rate=sample_rate)
```

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_16.wav">
    Your browser does not support the audio element.
</audio>

### 经过滤波器处理后

```python
Audio(filtered.T, rate=sample_rate2)
```

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_17.wav">
    Your browser does not support the audio element.
</audio>

### 应用编解码器

```python
Audio(codec_applied.T, rate=sample_rate2)
```

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/augmentation_audio_18.wav">
    Your browser does not support the audio element.
</audio>

脚本的总运行时间:(0 分钟 14.677 秒)
