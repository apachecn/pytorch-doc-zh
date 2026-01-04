


# 音频数据集 [¶](#audio-datasets "此标题的固定链接")


> 译者：[龙琰](https://github.com/bellongyan)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/audio_datasets_tutorial>
>
> 原始地址：<https://docs.pytorch.org/audio/stable/tutorials/audio_datasets_tutorial.html>

**Author**: [Moto Hira](moto@meta.com)

`torchaudio` 提供了对公共可访问数据集的轻松访问。有关可用数据集的列表，请参阅官方文档。

```python
import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)
```

输出：
```shell
2.10.0.dev20251013+cu126
2.8.0a0+1d65bbe
```

```python
import os

import IPython

import matplotlib.pyplot as plt


_SAMPLE_DIR = "_assets"
YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no")
os.makedirs(YESNO_DATASET_PATH, exist_ok=True)


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    figure, ax = plt.subplots()
    ax.specgram(waveform[0], Fs=sample_rate)
    figure.suptitle(title)
    figure.tight_layout()
```

在这里，我们展示了如何使用 `torchaudio.dataset.YESNO`数据集。

```python
dataset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH, download=True)
```

输出：
```shelll
2.8%
5.6%
8.4%
11.1%
13.9%
16.7%
19.5%
22.3%
25.1%
27.9%
30.7%
33.4%
36.2%
39.0%
41.8%
44.6%
47.4%
50.2%
52.9%
55.7%
58.5%
61.3%
64.1%
66.9%
69.7%
72.5%
75.2%
78.0%
80.8%
83.6%
86.4%
89.2%
92.0%
94.7%
97.5%
100.0%
```


```python
i = 1
waveform, sample_rate, label = dataset[i]
plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
IPython.display.Audio(waveform, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://docs.pytorch.org/audio/2.9.0/_images/sphx_glr_audio_datasets_tutorial_001.png" srcset="https://docs.pytorch.org/audio/2.9.0/_images/sphx_glr_audio_datasets_tutorial_001.png" alt="Original" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/audio_datasets_01.wav.wav">
    Your browser does not support the audio element.
</audio>


```python
i = 3
waveform, sample_rate, label = dataset[i]
plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
IPython.display.Audio(waveform, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://docs.pytorch.org/audio/2.9.0/_images/sphx_glr_audio_datasets_tutorial_002.png" srcset="https://docs.pytorch.org/audio/2.9.0/_images/sphx_glr_audio_datasets_tutorial_002.png" alt="Original" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/audio_datasets_02.wav.wav">
    Your browser does not support the audio element.
</audio>

```python
i = 5
waveform, sample_rate, label = dataset[i]
plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
IPython.display.Audio(waveform, rate=sample_rate)
```

<ul class="sphx-glr-horizontal">
<li><img src="https://docs.pytorch.org/audio/2.9.0/_images/sphx_glr_audio_datasets_tutorial_003.png" srcset="https://docs.pytorch.org/audio/2.9.0/_images/sphx_glr_audio_datasets_tutorial_003.png" alt="Original" class="sphx-glr-multi-img" width=268 height=134></li>
</ul>

<audio controls="controls" src="https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/2.0/tutorials/beginner/audio/audio_datasets_03.wav.wav">
    Your browser does not support the audio element.
</audio>

**脚本的总运行时间**：（0分钟3.295秒）