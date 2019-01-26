

# Generating Names with a Character-Level RNN
# 使用字符级别特征的RNN网络生成姓氏

> 译者：[hhxx2015](https://github.com/hhxx2015)


**Author**: [Sean Robertson](https://github.com/spro/practical-pytorch)
**作者**: [Sean Robertson](https://github.com/spro/practical-pytorch)

In the [last tutorial](char_rnn_classification_tutorial.html) we used a RNN to classify names into their language of origin. 
再上一个 [例子](char_rnn_classification_tutorial.html) 中我们使用RNN网络对名字所属的语言进行分类。
This time we’ll turn around and generate names from languages.
这一次我们会反过来根据语言生成姓氏。

```py
> python sample.py Russian RUS
Rovakov
Uantov
Shavakov

> python sample.py German GER
Gerren
Ereng
Rosher

> python sample.py Spanish SPA
Salla
Parer
Allan

> python sample.py Chinese CHI
Chan
Hang
Iun

```

We are still hand-crafting a small RNN with a few linear layers. 
我们仍使用几层linear层简单实现的RNN。
The big difference is instead of predicting a category after reading in all the letters of a name, we input a category and output one letter at a time. 
最大的区别在于，不是在读取一个姓氏的所有字母后预测类别，而是输入一个类别之后在每一时刻输出一个字母。
Recurrently predicting characters to form language (this could also be done with words or other higher order constructs) is often referred to as a “language model”.
循环预测字符以形成语言也通常被称为“语言模型”。（也可以将字符换成单词或更高级的结构来完成）


**阅读建议:**

我默认你已经安装好了PyTorch，熟悉Python语言，理解“张量”的概念：

*   [https://pytorch.org/](https://pytorch.org/) PyTorch安装指南
*   [Deep Learning with PyTorch: A 60 Minute Blitz](../beginner/deep_learning_60min_blitz.html) PyTorch入门
*   [Learning PyTorch with Examples](../beginner/pytorch_with_examples.html) 一些PyTorch的例子
*   [PyTorch for Former Torch Users](../beginner/former_torchies_tutorial.html) Lua Torch 用户参考

事先学习并了解RNN的工作原理对理解这个例子十分有帮助:

*   [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) shows a bunch of real life examples
*   [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) is about LSTMs specifically but also informative about RNNs in general

## 准备数据


Download the data from [here](https://download.pytorch.org/tutorial/data.zip) and extract it to the current directory.
点击这里[下载数据](https://download.pytorch.org/tutorial/data.zip) 并将其解压到当前文件夹。

See the last tutorial for more detail of this process. 
有关此过程的更多详细信息，请参阅上一个教程。
In short, there are a bunch of plain text files `data/names/[Language].txt` with a name per line. 
简而言之，有一些纯文本文件`data/names/[Language].txt`，它们的每行都有一个姓氏。
We split lines into an array, convert Unicode to ASCII, and end up with a dictionary `{language: [names ...]}`.
我们按行将文本按行切分得到一个数组，将Unicode编码转化为ASCII编码，最终得到`{language: [names ...]}`格式存储的字典变量。

```py
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))

```

Out:

```py
# categories: 18 ['Italian', 'German', 'Portuguese', 'Chinese', 'Greek', 'Polish', 'French', 'English', 'Spanish', 'Arabic', 'Czech', 'Russian', 'Irish', 'Dutch', 'Scottish', 'Vietnamese', 'Korean', 'Japanese']
O'Neal

```

## Creating the Network
## 构造神经网络
This network extends [the last tutorial’s RNN](#Creating-the-Network) with an extra argument for the category tensor, which is concatenated along with the others. 
这个神经网络比 [最后一个RNN教程](#Creating-the-Network)中的网络增加了额外参数类别张量，该参数与其他输入连接在一起。

The category tensor is a one-hot vector just like the letter input.
类别可以像字母一样组成one-hot向量构成张量输入。
We will interpret the output as the probability of the next letter. 
我们将输出当成下一字母的可能性。
When sampling, the most likely output letter is used as the next input letter.
采样过程中，当前输出可能性最高的字母作为下一时刻输入字母。
I added a second linear layer `o2o` (after combining hidden and output) to give it more muscle to work with. 
在组合隐藏状态和输出之后我增加了第二个linear层`o2o`，使模型的性能更好。

There’s also a dropout layer, which [randomly zeros parts of its input](https://arxiv.org/abs/1207.0580) with a given probability (here 0.1) and is usually used to fuzz inputs to prevent overfitting. 
当然还有一个dropout层，参考这篇论文[随机将输入部分替换为0](https://arxiv.org/abs/1207.0580)给出的参数（dropout=0.1）来模糊处理输入防止过拟合。

Here we’re using it towards the end of the network to purposely add some chaos and increase sampling variety.
我们将它添加到网络的末端，故意添加一些混乱使采样种类增加。

![](img/28a4f1426695fb55f1f6bc86278f6547.jpg)

```py
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

```

## Training
## 训练

### Preparing for Training
### 训练准备

First of all, helper functions to get random pairs of (category, line):
首先，构造一个可以随机获取成对训练数据(category, line)的函数。

```py
import random

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

```

For each timestep (that is, for each letter in a training word) the inputs of the network will be `(category, current letter, hidden state)` and the outputs will be `(next letter, next hidden state)`. 
对于每个时间步长（即，对于要训练单词中的每个字母），网络的输入将是“（类别，当前字母，隐藏状态）”，输出将是“（下一个字母，下一个隐藏状态）”。

So for each training set, we’ll need the category, a set of input letters, and a set of output/target letters.
因此，对于每个训练集，我们将需要类别、一组输入字母和一组输出/目标字母。

Since we are predicting the next letter from the current letter for each timestep, the letter pairs are groups of consecutive letters from the line - e.g. for `"ABCD&lt;EOS&gt;"` we would create (“A”, “B”), (“B”, “C”), (“C”, “D”), (“D”, “EOS”).
每一个时间序列，我们使用当前字母预测下一个字母，所以字母对来自于成行的单词。例如 对于 `"ABCD&lt;EOS&gt;"`，我们将创建（“A”，“B”），（“B”，“C”），（“C”，“D”），（“D”，“EOS”））。

![](img/3fae03d85aed3a2237fd4b2f7fb7b480.jpg)

The category tensor is a [one-hot tensor](https://en.wikipedia.org/wiki/One-hot) of size `&lt;1 x n_categories&gt;`. 
类别张量是一个`&lt;1 x n_categories&gt;`尺寸的[one-hot 向量](https://en.wikipedia.org/wiki/One-hot)

When training we feed it to the network at every timestep - this is a design choice, it could have been included as part of initial hidden state or some other strategy.
当训练时，我们按照时间序列将其提供给神经网络。

```py
# One-hot vector for category9
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

```

For convenience during training we’ll make a `randomTrainingExample` function that fetches a random (category, line) pair and turns them into the required (category, input, target) tensors.
为了方便训练，我们将创建一个`randomTrainingExample`函数，该函数随机获取（类别，行）的对并将它们转换为所需要的（类别，输入，目标）格式张量。

```py
# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

```

### Training the Network
### 训练神经网络

In contrast to classification, where only the last output is used, we are making a prediction at every step, so we are calculating loss at every step.
只使用了最后一个时刻输出的分类任务相比，我们

The magic of autograd allows you to simply sum these losses at each step and call backward at the end.

```py
criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

```

To keep track of how long training takes I am adding a `timeSince(timestamp)` function which returns a human readable string:

```py
import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

```

Training is business as usual - call train a bunch of times and wait a few minutes, printing the current time and loss every `print_every` examples, and keeping store of an average loss per `plot_every` examples in `all_losses` for plotting later.

```py
rnn = RNN(n_letters, 128, n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d  %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

```

Out:

```py
0m 21s (5000 5%) 2.5152
0m 43s (10000 10%) 2.7758
1m 4s (15000 15%) 2.2884
1m 25s (20000 20%) 3.2404
1m 47s (25000 25%) 2.7298
2m 8s (30000 30%) 3.4301
2m 29s (35000 35%) 2.2306
2m 51s (40000 40%) 2.5628
3m 12s (45000 45%) 1.7700
3m 34s (50000 50%) 2.4657
3m 55s (55000 55%) 2.1909
4m 16s (60000 60%) 2.1004
4m 38s (65000 65%) 2.3524
4m 59s (70000 70%) 2.3339
5m 21s (75000 75%) 2.3936
5m 42s (80000 80%) 2.1886
6m 3s (85000 85%) 2.0739
6m 25s (90000 90%) 2.5451
6m 46s (95000 95%) 1.5104
7m 7s (100000 100%) 2.4600

```

### Plotting the Losses
### 损失数据作图

Plotting the historical loss from all_losses shows the network learning:
从`all_losses`得到历史损失记录，反映了神经网络的学习情况：

```py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

```

![https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_generation_tutorial_001.png](img/5ad82e2b23a82287af2caa2fe4b316b3.jpg)

## Sampling the Network

To sample we give the network a letter and ask what the next one is, feed that in as the next letter, and repeat until the EOS token.

*   Create tensors for input category, starting letter, and empty hidden state
*   Create a string `output_name` with the starting letter
*   Up to a maximum output length,
    *   Feed the current letter to the network
    *   Get the next letter from highest output, and next hidden state
    *   If the letter is EOS, stop here
    *   If a regular letter, add to `output_name` and continue
*   Return the final name


Rather than having to give it a starting letter, another strategy would have been to include a “start of string” token in training and have the network choose its own starting letter.

```py
max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHI')

```

Out:

```py
Rovanik
Uakilovev
Shaveri
Garter
Eren
Romer
Santa
Parera
Artera
Chan
Ha
Iua

```

## Exercises
## 练习

*   Try with a different dataset of category -&gt; line, for example:
*   尝试其它的成行排列格式的数据集，比如:
    *   Fictional series -&gt; Character name
    *   Part of speech -&gt; Word
    *   Country -&gt; City
*   Use a “start of sentence” token so that sampling can be done without choosing a start letter
*   Get better results with a bigger and/or better shaped network
*   通过更大和更复杂的网络获得更好的结果
    *   Try the nn.LSTM and nn.GRU layers
    *   尝试 `nn.LSTM` 和 `nn.GRU` 层
    *   Combine multiple of these RNNs as a higher level network
    *   组合这些 RNN构造更复杂的神经网络



**Total running time of the script:** ( 7 minutes 7.943 seconds)

[`下载 Python源码: char_rnn_generation_tutorial.py`](../_downloads/8167177b6dd8ddf05bb9fe58744ac406/char_rnn_generation_tutorial.py)[`下载 Jupyter notebook: char_rnn_generation_tutorial.ipynb`](../_downloads/a35c00bb5afae3962e1e7869c66872fa/char_rnn_generation_tutorial.ipynb)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.readthedocs.io)

