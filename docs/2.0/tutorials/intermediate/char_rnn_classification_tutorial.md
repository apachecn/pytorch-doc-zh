


 没有10



 单击
 [此处](#sphx-glr-download-intermediate-char-rnn-classification-tutorial-py)
 下载完整的示例代码








 NLP 从头开始​​：使用字符级 RNN 对名称进行分类
 [¶](#nlp-from-scratch-classifying-names-with-a-character-level-rnn "此标题的永久链接")
==================================================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/char_rnn_classification_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html>




**作者** 
 :
 [肖恩·罗伯逊](https://github.com/spro)




 我们将构建和训练一个基本的字符级循环神经网络 (RNN) 来对单词进行分类。本教程以及其他两个
自然语言处理 (NLP)“ 从头开始​​” 教程
 [NLP 从头开始​​：使用字符级 RNN 生成名称](char_rnn_ Generation_tutorial.html)\ n 和
 [NLP 从头开始​​：使用序列到序列网络和注意力的翻译](seq2seq_translation_tutorial.html)
 ，展示如何
预处理数据以建模 NLP。特别是，这些教程没有
使用 torchtext
 
 的许多便利功能，因此您可以
了解 NLP 模型的预处理如何在较低级别上工作。




 字符级 RNN 将单词读取为一系列字符 -
在每一步输出预测和 “hidden 状态”，将其
之前的隐藏状态输入到每个下一步。我们将最终的预测
作为输出，即该单词属于哪个类。




 具体而言，我们’ 将训练来自 18 种语言
的数千个姓氏，并根据
拼写预测名称来自哪种语言：






```
$ python predict.py Hinton
(-0.47) Scottish
(-1.52) English
(-3.57) Irish

$ python predict.py Schmidhuber
(-0.19) German
(-2.48) Czech
(-2.68) Dutch

```





 推荐准备
 [¶](#recommended-preparation "此标题的固定链接")
----------------------------------------------------------------------------------------------------



 在开始本教程之前，建议您安装 PyTorch，
并对 Python 编程语言和张量有基本的了解：



* <https://pytorch.org/>
 安装说明
* [使用 PyTorch 进行深度学习：60 分钟闪电战](../beginner/deep_learning_60min_blitz.html)
 一般开始使用 PyTorch\并学习张量的基础知识
* [通过示例学习 PyTorch](../beginner/pytorch_with_examples.html)
 进行广泛而深入的概述
* [针对前 Torch 用户的 PyTorch](../beginner/former_torchies_tutorial.html)
 如果您是前 Lua Torch 用户



 了解 RNN 及其工作原理也很有用：



* [循环神经网络的不合理有效性](https://karpathy.github.io/2015/05/21/rnn-effectness/) 
 展示了一堆现实生活中的例子
* [理解 LSTM
网络](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) 
 专门介绍 LSTM，但也提供有关 RNN 的一般信息






 准备数据
 [¶](#preparing-the-data "永久链接到此标题")
--------------------------------------------------------------------------------------- -




 注意




 从
 [此处](https://download.pytorch.org/tutorial/data.zip) 下载数据并将其解压到当前目录。





 `data/names`
 目录中包含 18 个名为
 `[Language].txt`
 的文本文件。每个文件包含一堆名称，每行一个名称，大部分是罗马化的（但我们仍然需要从 Unicode 转换为 ASCII）。




 我们’ll 最终得到每种语言的名称列表的字典，
 `{language:
 

 [names
 

...]}`
 。通用变量 “category” 和 “line”
（在我们的例子中用于语言和名称）用于以后的扩展性。






```
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

```






```
['data/names/Arabic.txt', 'data/names/Chinese.txt', 'data/names/Czech.txt', 'data/names/Dutch.txt', 'data/names/English.txt', 'data/names/French.txt', 'data/names/German.txt', 'data/names/Greek.txt', 'data/names/Irish.txt', 'data/names/Italian.txt', 'data/names/Japanese.txt', 'data/names/Korean.txt', 'data/names/Polish.txt', 'data/names/Portuguese.txt', 'data/names/Russian.txt', 'data/names/Scottish.txt', 'data/names/Spanish.txt', 'data/names/Vietnamese.txt']
Slusarski

```




 现在我们有了
 `category_lines`
 ，一个将每个类别
（语言）映射到行（名称）列表的字典。我们还跟踪
“所有_类别”
（只是语言列表）和
“n_类别”
，以供
以后参考。






```
print(category_lines['Italian'][:5])

```






```
['Abandonato', 'Abatangelo', 'Abatantuono', 'Abate', 'Abategiovanni']

```




### 
 将名称转换为张量
 [¶](#turning-names-into-tensors "永久链接到此标题")



 现在我们已经组织好了所有名称，我们需要将它们转换为
张量才能使用它们。




 为了表示单个字母，我们使用大小为 
 `<1
 

 x
 

 n_letters>` 的 “one-hot 向量” 
 。除了当前字母的 1
at 索引之外，one-hot 向量都用 0 填充，例如
 `"b"
 

 =
 

 <0
 

 1 
 

 0
 

 0
 

 0
 

...>`
 。




 为了创建一个单词，我们将一堆单词连接成一个 2D 矩阵
 `<line_length
 

 x
 

 1
 

 x
 

 n\ \_字母>`
.




 额外的 1 维是因为 PyTorch 假定所有内容都在
批次中 - 我们’ 这里仅使用批次大小 1。






```
import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())

```






```
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0.]])
torch.Size([5, 1, 57])

```







 创建网络
 [¶](#creating-the-network "永久链接到此标题")
---------------------------------------------------------------------------------



 在 autograd 之前，在 Torch 中创建循环神经网络需要
在多个时间步长上克隆层的参数。层拥有隐藏状态和梯度，现在完全由图本身处理。这意味着您可以以非常 “pure” 的方式实现 RNN，




 这个 RNN 模块（大部分复制自 [the PyTorch for Torch users
tutorial](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#example-2-recurrent-net) 
 ） 
 只是 2 个线性层，它们对输入和隐藏状态进行操作，

 `LogSoftmax`
 层位于输出之后。






```
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

```


为了运行这个网络的一个步骤，我们需要传递一个输入（在我们的例子中，是当前字母的张量）和一个先前的隐藏状态（我们首先将其初始化为零）。我们’将得到输出（每种语言的概率）和下一个隐藏状态（我们为下一步保留
）。






```
input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)

```




 为了提高效率，我们不想’ 每一步都创建一个新的张量，因此我们将使用
 `lineToTensor`
 而不是
 `letterToTensor`
 并且使用切片。这可以通过
预计算批量张量来进一步优化。






```
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)

```






```
tensor([[-2.9083, -2.9270, -2.9167, -2.9590, -2.9108, -2.8332, -2.8906, -2.8325,
         -2.8521, -2.9279, -2.8452, -2.8754, -2.8565, -2.9733, -2.9201, -2.8233,
         -2.9298, -2.8624]], grad_fn=<LogSoftmaxBackward0>)

```




 正如你所看到的，输出是一个
 `<1
 

 x
 

 n_categories>`
 张量，其中
每个项目都是该类别的可能性（较高更有可能）。






 训练
 [¶](#training "固定链接到此标题")
-----------------------------------------------------



### 
 准备训练
 [¶](#preparing-for-training "永久链接到此标题")



 在开始训练之前，我们应该创建一些辅助函数。首先是解释网络的输出，我们知道它是每个类别的可能性。我们可以使用
 `Tensor.topk`
 来获取最大值的索引
:






```
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))

```






```
('Scottish', 15)

```




 我们还需要一种快速获取训练示例的方法（名称及其
语言）：






```
import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

```






```
category = Chinese / line = Hou
category = Scottish / line = Mckay
category = Arabic / line = Cham
category = Russian / line = V'Yurkov
category = Irish / line = O'Keeffe
category = French / line = Belrose
category = Spanish / line = Silva
category = Japanese / line = Fuchida
category = Greek / line = Tsahalis
category = Korean / line = Chang

```





### 
 训练网络
 [¶](#training-the-network "永久链接到此标题")



 现在训练这个网络所需要做的就是向它展示一堆示例，
让它做出猜测，并告诉它’ 是否错误。




 对于损失函数
 `nn.NLLLoss`
 是合适的，因为 RNN 的最后
层是
 `nn.LogSoftmax`
 。






```
criterion = nn.NLLLoss()

```




 每个训练循环将：



* 创建输入和目标张量
* 创建归零的初始隐藏状态
* 读取 and 中的每个字母



	+ 保留下一个字母的隐藏状态
* 将最终输出与目标进行比较
* 反向传播
* 返回输出和损失





```
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

```




 现在我们只需要用一堆例子来运行它。由于
 `train`
 函数返回输出和损失，我们可以打印其
猜测，并跟踪损失以进行绘图。由于有 1000 个
示例，我们仅打印每个
 `print_every`
 个示例，并
取损失的平均值。






```
import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print ``iter`` number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

```






```
5000 5% (0m 4s) 2.6379 Horigome / Japanese ✓
10000 10% (0m 9s) 2.0172 Miazga / Japanese ✗ (Polish)
15000 15% (0m 14s) 0.2680 Yukhvidov / Russian ✓
20000 20% (0m 18s) 1.8239 Mclaughlin / Irish ✗ (Scottish)
25000 25% (0m 23s) 0.6978 Banh / Vietnamese ✓
30000 30% (0m 28s) 1.7433 Machado / Japanese ✗ (Portuguese)
35000 35% (0m 32s) 0.0340 Fotopoulos / Greek ✓
40000 40% (0m 37s) 1.4637 Quirke / Irish ✓
45000 45% (0m 41s) 1.9018 Reier / French ✗ (German)
50000 50% (0m 46s) 0.9174 Hou / Chinese ✓
55000 55% (0m 51s) 1.0506 Duan / Vietnamese ✗ (Chinese)
60000 60% (0m 56s) 0.9617 Giang / Vietnamese ✓
65000 65% (1m 0s) 2.4557 Cober / German ✗ (Czech)
70000 70% (1m 5s) 0.8502 Mateus / Portuguese ✓
75000 75% (1m 10s) 0.2750 Hamilton / Scottish ✓
80000 80% (1m 14s) 0.7515 Maessen / Dutch ✓
85000 85% (1m 19s) 0.0912 Gan / Chinese ✓
90000 90% (1m 23s) 0.1190 Bellomi / Italian ✓
95000 95% (1m 28s) 0.0137 Vozgov / Russian ✓
100000 100% (1m 33s) 0.7810 Tong / Vietnamese ✓

```





### 
 绘制结果
 [¶](#plotting-the-results "此标题的固定链接")



 绘制
 `all_losses` 的历史损失
 显示网络
学习情况：






```
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

```



![char rnn 分类教程](https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_classification_tutorial_001.png)



```
[<matplotlib.lines.Line2D object at 0x7fe720dabdc0>]

```







 评估结果
 [¶](#evaluating-the-results "固定链接到此标题")
-------------------------------------------------------------------------------------



 为了查看网络在不同类别上的表现，我们将
创建一个混淆矩阵，指示每种实际语言（行）
网络猜测哪种语言（列）。为了计算混淆矩阵，使用
 `evaluate()`
 在网络中运行一堆样本，这与
 `train()`
 减去反向传播相同。






```
# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

```



![char rnn 分类教程](https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_classification_tutorial_002.png)



```
/var/lib/jenkins/workspace/intermediate_source/char_rnn_classification_tutorial.py:445: UserWarning:

set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.

/var/lib/jenkins/workspace/intermediate_source/char_rnn_classification_tutorial.py:446: UserWarning:

set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.

```




 您可以从主轴上找出亮点，以显示它猜错的
语言，例如韩语用中文，意大利语用西班牙语
。它似乎对希腊语表现很好，但对英语表现不佳（可能是因为与其他语言重叠）。




### 
 根据用户输入运行
 [¶](#running-on-user-input "永久链接到此标题")





```
def predict(input_line, n_predictions=3):
    print('> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')

```






```
> Dovesky
(-0.57) Czech
(-0.97) Russian
(-3.43) English

> Jackson
(-1.02) Scottish
(-1.49) Russian
(-1.96) English

> Satoshi
(-0.42) Japanese
(-1.70) Polish
(-2.74) Italian

```




 脚本的最终版本
 [在 Practical PyTorch
repo 中](https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification) 
 拆分上述内容代码到几个文件中：



* `data.py`
（加载文件）
* `model.py`
（定义 RNN）
* `train.py`
（运行训练）
* `predict.py`\ n（使用命令行参数运行
 `predict()`
）
* `server.py`
（使用
 `bottle.py`
 将预测作为 JSON API 提供）



 运行
 `train.py`
 来训练并保存网络。




 使用名称运行
 `predict.py`
 以查看预测:






```
$ python predict.py Hazaki
(-0.42) Japanese
(-1.39) Polish
(-3.51) Czech

```




 运行
 `server.py`
 并访问
 <http://localhost:5533/Yourname>
 以获取预测的 JSON
输出。







 练习
 [¶](#exercises "永久链接到此标题")
---------------------------------------------------------------------


* 尝试使用不同的行 -> 类别数据集，例如：



	+ 任何单词 -> 语言
	+ 名字 -> 性别
	+ 角色名称 -> 作者
	+ 页面标题 -> 博客或 subreddit
* 通过更大和/或更好的形状获得更好的结果网络



	+ 添加更多线性层
	+ 尝试
	 `nn.LSTM`
	 和
	 `nn.GRU`
	 层
	+ 将这些 RNN 的多个组合为更高层次的网络



**脚本的总运行时间：** 
（1 分 38.442 秒）






[`下载
 

 Python
 

 源
 

 代码:
 

 char_rnn_classification_tutorial.py`](../_downloads /37c8905519d3fd3f437b783a48d06eac/char_rnn_classification_tutorial.py)






[`下载
 

 Jupyter
 

 笔记本:
 

 char_rnn_classification_tutorial.ipynb`](../_downloads/13b143c2380f4768d9432d808ad50799/char_rnn_classification_tutorial. ipynb）






[Sphinx-Gallery 生成的图库](https://sphinx-gallery.github.io)









