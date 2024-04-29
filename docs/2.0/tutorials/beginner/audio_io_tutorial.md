# 音频 I/O [¶](#audio-i-o "此标题的永久链接")

> 译者：[龙琰](https://github.com/bellongyan)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/audio_io_tutorial>
>
> 原始地址：<https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html>

**作者**: [Moto Hira](moto@meta.com)

本教程展示了如何使用 TorchAudio 的基本 I/O API 来检查音频数据，
将它们加载到 PyTorch 张量中并保存 PyTorch 张量。

> **Warning**
>
> 在最近的版本中，音频 I/O 有多个计划/做出的更改。有关这些更改的详细信息，请参阅[Dispatcher 的介绍](https://pytorch.org/audio/stable/torchaudio.html#dispatcher-migration)。

```python
import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)
```

输出：

```python
2.3.0
2.3.0
```

## 准备工作

首先，我们导入模块并下载我们在本教程中使用的音频资源。

> **Note**
>
> 在 Google Colab 中运行本教程时，请使用以下命令安装所需的包：
>
> ```python
> !pip install boto3
> ```

```python
import io
import os
import tarfile
import tempfile

import boto3
import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset

SAMPLE_GSM = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.gsm")
SAMPLE_WAV = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SAMPLE_WAV_8000 = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")


def _hide_seek(obj):
    class _wrapper:
        def __init__(self, obj):
            self.obj = obj

        def read(self, n):
            return self.obj.read(n)

    return _wrapper(obj)
```

输出：

```python
  0%|          | 0.00/7.99k [00:00<?, ?B/s]
100%|##########| 7.99k/7.99k [00:00<00:00, 15.8MB/s]

  0%|          | 0.00/53.2k [00:00<?, ?B/s]
100%|##########| 53.2k/53.2k [00:00<00:00, 43.3MB/s]
```

## 查询音频元数据

函数 `torchaudio.info()` 获取音频元数据。 您可以提供类似路径的对象或类似文件的对象。

```python
metadata = torchaudio.info(SAMPLE_WAV)
print(metadata)
```

输出：

```python
AudioMetaData(sample_rate=16000, num_frames=54400, num_channels=1, bits_per_sample=16, encoding=PCM_S)
```

其中

- `sample_rate` 是音频的采样率
- `num_channels` 是通道的数量
- `num_frames` 是每个通道的帧数
- `bits_per_sample` 是位深度
- `encoding` 是样本编码格式

`encoding` 可以采用下列值之一：

- `"PCM_S"`: 有符号整数线性 PCM
- `"PCM_U"`: 无符号整数线性 PCM
- `"PCM_F"`: 浮点线性 PCM
- `"FLAC"`: Flac, [Free Lossless Audio
  Codec](https://xiph.org/flac/)
- `"ULAW"`: Mu-law,
  [[wikipedia](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)]
- `"ALAW"`: A-law
  [[wikipedia](https://en.wikipedia.org/wiki/A-law_algorithm)]
- `"MP3"` : MP3, MPEG-1 Audio Layer III
- `"VORBIS"`: OGG Vorbis [[xiph.org](https://xiph.org/vorbis/)]
- `"AMR_NB"`: 自适应多速率
  [[wikipedia](https://en.wikipedia.org/wiki/Adaptive_Multi-Rate_audio_codec)]
- `"AMR_WB"`: 自适应多速率宽带
  [[wikipedia](https://en.wikipedia.org/wiki/Adaptive_Multi-Rate_Wideband)]
- `"OPUS"`: Opus [[opus-codec.org](https://opus-codec.org/)]
- `"GSM"`: GSM-FR
  [[wikipedia](https://en.wikipedia.org/wiki/Full_Rate)]
- `"HTK"`: 单通道 16 位 PCM
- `"UNKNOWN"` 以上都不是

**注意**

- 对于具有压缩和/或可变比特率的格式(如 MP3)，`bits_per_sample`可以为`0`。

- 对于 GSM-FR 格式，`num_frames`可以为`0`。

```python
metadata = torchaudio.info(SAMPLE_GSM)
print(metadata)
```

输出：

```python
AudioMetaData(sample_rate=8000, num_frames=39680, num_channels=1, bits_per_sample=0, encoding=GSM)
```

## 查询 file-like object

[`torchaudio.info`](https://pytorch.org/audio/stable/generated/torchaudio.info.html#torchaudio.info) 适用于 file-like object。

```python
url = "https://download.pytorch.org/torchaudio/tutorial-assets/steam-train-whistle-daniel_simon.wav"
with requests.get(url, stream=True) as response:
    metadata = torchaudio.info(_hide_seek(response.raw))
print(metadata)
```

输出：

```python
AudioMetaData(sample_rate=44100, num_frames=109368, num_channels=2, bits_per_sample=16, encoding=PCM_S)
```

> **NOTE**
>
> 当传递一个 file-like object 时，`info`不会读取所有底层数据;相反，它从一开始只读取数据的一部分。因此，对于给定的音频格式，它可能无法检索正确的元数据，包括格式本身。在这种情况下，可以通过`format`参数指定音频的格式。

## 加载音频数据

要加载音频数据，可以使用[`torchaudio.load()`](https://pytorch.org/audio/stable/generated/torchaudio.load.html#torchaudio.load)。

这个函数接受一个 path-like object 或 file-like object 作为输入。

返回值是一个由波形(`Tensor`)和采样率(`int`)组成的元组。

默认情况下，生成的张量对象的`dtype=torch.float32`类型，其取值范围为`[-1.0,1.0]`。

有关支持的格式列表，请参阅[`torchaudio文档`](https://pytorch.org/audio)。

```python
waveform, sample_rate = torchaudio.load(SAMPLE_WAV)
```

```python
def plot_waveform(waveform, sample_rate):
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
    figure.suptitle("waveform")
```

```python
plot_waveform(waveform, sample_rate)
```

<img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_io_tutorial_001.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_io_tutorial_001.png" alt="waveform" class="sphx-glr-single-img" width=661 height=331>

```python
def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
```

```python
plot_specgram(waveform, sample_rate)
```

<img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_io_tutorial_002.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_io_tutorial_002.png" alt="Spectrogram" class="sphx-glr-single-img" width=661 height=331>

```python
Audio(waveform.numpy()[0], rate=sample_rate)
```

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/io_01.wav">
    Your browser does not support the audio element.
</audio>

## 从 file-like object 加载

I/O 函数支持类文件对象。这允许从本地文件系统内外的位置获取和解码音频数据。下面的例子说明了这一点。

```python
# Load audio data as HTTP request
url = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
with requests.get(url, stream=True) as response:
    waveform, sample_rate = torchaudio.load(_hide_seek(response.raw))
plot_specgram(waveform, sample_rate, title="HTTP datasource")
```

<img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_io_tutorial_003.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_io_tutorial_003.png" alt="Spectrogram" class="sphx-glr-single-img" width=661 height=331>

```python
# Load audio from tar file
tar_path = download_asset("tutorial-assets/VOiCES_devkit.tar.gz")
tar_item = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
with tarfile.open(tar_path, mode="r") as tarfile_:
    fileobj = tarfile_.extractfile(tar_item)
    waveform, sample_rate = torchaudio.load(fileobj)
plot_specgram(waveform, sample_rate, title="TAR file")
```

<img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_io_tutorial_004.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_io_tutorial_004.png" alt="Spectrogram" class="sphx-glr-single-img" width=661 height=331>

输出：

```python
  0%|          | 0.00/110k [00:00<?, ?B/s]
100%|##########| 110k/110k [00:00<00:00, 62.6MB/s]
```

```python
# Load audio from S3
bucket = "pytorch-tutorial-assets"
key = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
response = client.get_object(Bucket=bucket, Key=key)
waveform, sample_rate = torchaudio.load(_hide_seek(response["Body"]))
plot_specgram(waveform, sample_rate, title="From S3")
```

<img src="https://pytorch.org/audio/stable/_images/sphx_glr_audio_io_tutorial_005.png" srcset="https://pytorch.org/audio/stable/_images/sphx_glr_audio_io_tutorial_005.png" alt="Spectrogram" class="sphx-glr-single-img" width=661 height=331>

## 切片技巧

提供 `num_frames` 和 `frame_offset` 参数将解码限制为输入的相应段。

使用普通张量切片也可以实现相同的结果，(即`waveform[:, frame_offset:frame_offset+num_frames]`)。但提供 `num_frames` 和 `frame_offset` 参数更高效。

这是因为函数一旦完成对请求帧的解码，就会结束数据采集和解码。当音频数据通过网络传输时，这是有利的，因为一旦获取了必要的数据量，数据传输就会停止。

下面的例子说明了这一点。

```python
# Illustration of two different decoding methods.
# The first one will fetch all the data and decode them, while
# the second one will stop fetching data once it completes decoding.
# The resulting waveforms are identical.

frame_offset, num_frames = 16000, 16000  # Fetch and decode the 1 - 2 seconds

url = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
print("Fetching all the data...")
with requests.get(url, stream=True) as response:
    waveform1, sample_rate1 = torchaudio.load(_hide_seek(response.raw))
    waveform1 = waveform1[:, frame_offset : frame_offset + num_frames]
    print(f" - Fetched {response.raw.tell()} bytes")

print("Fetching until the requested frames are available...")
with requests.get(url, stream=True) as response:
    waveform2, sample_rate2 = torchaudio.load(
        _hide_seek(response.raw), frame_offset=frame_offset, num_frames=num_frames
    )
    print(f" - Fetched {response.raw.tell()} bytes")

print("Checking the resulting waveform ... ", end="")
assert (waveform1 == waveform2).all()
print("matched!")
```

输出：

```python
Fetching all the data...
 - Fetched 108844 bytes
Fetching until the requested frames are available...
 - Fetched 108844 bytes
Checking the resulting waveform ... matched!
```

## 将音频保存到文件

要将音频数据保存为常见应用程序可解释的格式，您可以使用 [`torchaudio.save()`](https://pytorch.org/audio/stable/generated/torchaudio.save.html#torchaudio.save)。

该函数接受类似路径的对象或类似文件的对象。

当传递类似文件的对象时，您还需要提供参数`格式`，以便函数知道应该使用哪种格式。 对于类似路径的对象，该函数将从扩展名推断格式。 如果要保存到没有扩展名的文件，则需要提供参数`格式`。

保存 WAV 格式的数据时，`float32` Tensor 的默认编码是 32 位浮点 PCM。 您可以提供参数 `encoding` 和 `bits_per_sample` 来更改此行为。 例如，要将数据保存在 16 位有符号整数 PCM 中，您可以执行以下操作。

> **Note**
>
> 以较低位深度的编码保存数据会减少生成的文件大小，但也会降低精度。

```python
waveform, sample_rate = torchaudio.load(SAMPLE_WAV)
```

```python
def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")
    print()
```

不带任何编码选项保存。该函数将选择所提供数据适合的编码

```python
with tempfile.TemporaryDirectory() as tempdir:
    path = f"{tempdir}/save_example_default.wav"
    torchaudio.save(path, waveform, sample_rate)
    inspect_file(path)
```

输出：

```python
----------
Source: /tmp/tmp7e22i972/save_example_default.wav
----------
 - File size: 108878 bytes
 - AudioMetaData(sample_rate=16000, num_frames=54400, num_channels=1, bits_per_sample=16, encoding=PCM_S)
```

保存为 16 位有符号整数线性 PCM，生成的文件占用一半存储空间但会损失精度

```python
with tempfile.TemporaryDirectory() as tempdir:
    path = f"{tempdir}/save_example_PCM_S16.wav"
    torchaudio.save(path, waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)
    inspect_file(path)
```

输出：

```python
----------
Source: /tmp/tmpo_4z9s1q/save_example_PCM_S16.wav
----------
 - File size: 108878 bytes
 - AudioMetaData(sample_rate=16000, num_frames=54400, num_channels=1, bits_per_sample=16, encoding=PCM_S)
```

[`torchaudio.save()`](https://pytorch.org/audio/stable/generated/torchaudio.save.html#torchaudio.save) 也可以处理其他格式。

举几个例子:

```python
formats = [
    "flac",
    # "vorbis",
    # "sph",
    # "amb",
    # "amr-nb",
    # "gsm",
]
```

```python
waveform, sample_rate = torchaudio.load(SAMPLE_WAV_8000)
with tempfile.TemporaryDirectory() as tempdir:
    for format in formats:
        path = f"{tempdir}/save_example.{format}"
        torchaudio.save(path, waveform, sample_rate, format=format)
        inspect_file(path)
```

输出：

```python
----------
Source: /tmp/tmp3t5yx5qq/save_example.flac
----------
 - File size: 45262 bytes
 - AudioMetaData(sample_rate=8000, num_frames=27200, num_channels=1, bits_per_sample=16, encoding=FLAC)
```

## 保存为 file-like object

与其他 I/O 功能类似，您可以将音频保存为 file-like object。 保存为 file-like object 需要参数`format`。

```python
waveform, sample_rate = torchaudio.load(SAMPLE_WAV)

# Saving to bytes buffer
buffer_ = io.BytesIO()
torchaudio.save(buffer_, waveform, sample_rate, format="wav")

buffer_.seek(0)
print(buffer_.read(16))
```

输出：

```python
b'RIFFF\xa9\x01\x00WAVEfmt '
```

脚本总运行时间：（0 分 2.000 秒）
