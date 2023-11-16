
 NLP 从头开始​​：使用字符级 RNN 生成名称
 [¶](#nlp-from-scratch-generate-names-with-a-character-level-rnn "此标题的永久链接")
================================================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/char_rnn_generation_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html>




**作者** 
 :
 [肖恩·罗伯逊](https://github.com/spro)




 这是我们关于 “NLP From Scratch” 的三个教程中的第二个。
在
 [第一个教程](/intermediate/char_rnn_classification_tutorial) 
 我们使用 RNN 进行分类名称转换成其原始语言。这次
我们’将转身并根据语言生成名称。






```
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




 我们仍在手工制作一个带有几个线性层的小型 RNN。最大的区别是，我们不是在读取名称的所有字母后预测类别，而是输入一个类别并一次输出一个字母。反复预测字符以形成语言（这也可以通过单词或
完成）其他高阶结构）通常被称为“语言模型”。




**推荐阅读：**




 我假设您至少已经安装了 PyTorch，了解 Python，并且
了解张量：



* <https://pytorch.org/>
 安装说明
* [使用 PyTorch 进行深度学习：60 分钟闪电战](../beginner/deep_learning_60min_blitz.html)
 一般开始使用 PyTorch\ n* [通过示例学习 PyTorch](../beginner/pytorch_with_examples.html)
 进行广泛而深入的概述
* [针对前 Torch 用户的 PyTorch](../beginner/former_torchies_tutorial.html)
 如果您是前 Lua Torch 用户



 了解 RNN 及其工作原理也很有用：



* [循环神经网络的不合理有效性](https://karpathy.github.io/2015/05/21/rnn-effectness/) 
 展示了一堆现实生活中的例子
* [理解 LSTM
网络](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) 
 专门介绍 LSTM，但也提供有关 RNN 的一般信息




 我还建议之前的教程，
 [从头开始的 NLP：使用字符级 RNN 进行名称分类](char_rnn_classification_tutorial.html)





 准备数据
 [¶](#preparing-the-data "永久链接到此标题")
--------------------------------------------------------------------------------------- -




 注意




 从
 [此处](https://download.pytorch.org/tutorial/data.zip) 下载数据并将其解压到当前目录。





 有关此过程的更多详细信息，请参阅上一个教程。简而言之，有一堆纯文本文件
 `data/names/[Language].txt`
 每行都有一个
名称。我们将行分割成一个数组，将 Unicode 转换为 ASCII，
最后得到一个字典
 `{语言:
 

 [names
 

...]}`
 。






```
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
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]

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






```
# categories: 18 ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 'Greek', 'Irish', 'Italian', 'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese']
O'Neal

```






 创建网络
 [¶](#creating-the-network "永久链接到此标题")
--------------------------------------------------------------------------------



 该网络扩展了
 [上一个教程’s RNN](#Creating-the-Network) 
 并为类别张量添加了一个额外参数，该参数与其他张量一起连接
。类别张量是一个单热向量，
就像字母输入一样。




 我们将把输出解释为下一个字母的概率。采样时，
最可能的输出字母将用作下一个输入
字母。




 我添加了第二个线性层
 `o2o`
 （在组合隐藏和
输出之后）以赋予它更多的功能。还有’s还有一个dropout
层，
[随机将其输入的部分归零](https://arxiv.org/abs/1207.0580)
以给定的概率
(此处为0.1)通常用于模糊输入以防止过度拟合。
这里我们’在网络末端使用它来故意添加一些
混沌并增加采样多样性。




![](https://i.imgur.com/jzVrf7f.png)




```
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






 训练
 [¶](#training "此标题的固定链接")
---------------------------------------------------------------------



### 
 准备训练
 [¶](#preparing-for-training "永久链接到此标题")



 首先，辅助函数获取随机对（类别、行）：






```
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




 对于每个时间步长（即训练单词中的每个字母），网络的输入将是
 `(类别，
 

 当前
 

 字母，
 

 隐藏
 

 状态)`
 并且输出将为
 `(下一个
 

 字母，
 

 下一个
 

 隐藏
 

 状态)`
.因此，对于每个训练集，我们’ll
需要类别、一组输入字母和一组输出/目标
字母。




 由于我们正在为每个时间步预测当前字母的下一个字母，因此字母对是来自该行的连续字母组 - 例如对于
 `"ABCD<EOS>"`
 我们将创建 (\xe2\x80\x9cA\xe2\x80\x9d, \xe2\x80\x9cB\xe2\x80\x9d), (\xe2\x80\ x9cB\xe2\x80\x9d,\xe2\x80\x9cC\xe2\x80\x9d),
(\xe2\x80\x9cC\xe2\x80\x9d,\xe2\x80\x9cD\xe2\x80\x9d ), (\xe2\x80\x9cD\xe2\x80\x9d, \xe2\x80\x9cEOS\xe2\x80\x9d)。




![](https://i.imgur.com/JH58tXY.png)


 类别张量是一个
 [one-hot
张量](https://en.wikipedia.org/wiki/One-hot)
，大小
 `<1
 

 x
 
 
 n_categories>`
 。训练时，我们在每个
时间步将其提供给网络 - 这是一种设计选择，它可能已被
作为初始隐藏状态或其他策略的一部分。






```
# One-hot vector for category
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

# ``LongTensor`` of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

```




 为了训练过程中的方便，我们’ 将创建一个
 `randomTrainingExample`
 函数，用于获取随机（类别、行）对并将它们
转换为
所需的（类别、输入、目标）张量.






```
# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

```





### 
 训练网络
 [¶](#training-the-network "永久链接到此标题")



 与仅使用最后一个输出的分类相反，我们在每一步都进行预测，因此我们在每一步都计算损失。




 autograd 的魔力使您可以简单地在每一步对这些损失求和
并在最后调用 back。






```
criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = torch.Tensor([0]) # you can also just simply use ``loss = 0``

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)

```




 为了跟踪训练需要多长时间，我添加了一个 
 `timeSince(timestamp)`
 函数，该函数返回人类可读的字符串：






```
import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

```




 训练照常进行 - 多次调用训练并等待
几分钟，打印当前时间和每个
 `print_every`
 个示例的损失，并存储每个示例的平均损失
 `plot_every`
 示例

 
 `all_losses`
 用于稍后绘制。






```
rnn = RNN(n_letters, 128, n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every ``plot_every`` ``iters``

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

```






```
0m 38s (5000 5%) 3.1506
1m 16s (10000 10%) 2.5070
1m 56s (15000 15%) 3.3047
2m 34s (20000 20%) 2.4247
3m 14s (25000 25%) 2.6406
3m 53s (30000 30%) 2.0266
4m 32s (35000 35%) 2.6520
5m 10s (40000 40%) 2.4261
5m 49s (45000 45%) 2.2302
6m 28s (50000 50%) 1.6496
7m 7s (55000 55%) 2.7101
7m 46s (60000 60%) 2.5396
8m 25s (65000 65%) 2.5978
9m 4s (70000 70%) 1.6029
9m 43s (75000 75%) 0.9634
10m 22s (80000 80%) 3.0950
11m 1s (85000 85%) 2.0512
11m 40s (90000 90%) 2.5302
12m 19s (95000 95%) 3.2365
12m 58s (100000 100%) 1.7113

```





### 
 绘制损失
 [¶](#plotting-the-losses "永久链接到此标题")



 绘制所有 _losses 的历史损失显示网络
学习情况：






```
import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)

```



![char rnn 生成教程](https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_ Generation_tutorial_001.png)



```
[<matplotlib.lines.Line2D object at 0x7efbc9fe3070>]

```







 对网络进行采样
 [¶](#sampling-the-network "固定链接到此标题")
---------------------------------------------------------------------------------



 为了进行采样，我们给网络一个字母并询问下一个字母是什么，
将其作为下一个字母输入，然后重复，直到 EOS 代币。



* 为输入类别、起始字母和空隐藏状态创建张量
* 创建一个字符串
 `output_name`
 以起始字母
* 达到最大输出长度，



	+ 将当前字母输入网络
	+ 从最高输出中获取下一个字母，以及下一个隐藏状态
	+ 如果字母是 EOS，则在此停止
	+ 如果是常规字母，则添加到
 	 `output_name`
	 并继续
* 返回最终名称




 注意




 另一种策略是在训练中包含字符串 xe2\x80\x9d 标记的 \xe2\x80\x9cstart ，而不是必须给它一个起始字母，并让网络选择自己的起始字母。
 n






```
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






```
Rovaki
Uarinovev
Shinan
Gerter
Eeren
Roune
Santera
Paneraz
Allan
Chin
Han
Ion

```






 练习
 [¶](#exercises "永久链接到此标题")
---------------------------------------------------------------------


* 尝试使用类别 -> 行的不同数据集，例如：



	+ 虚构系列 -> 角色名称
	+ 词性 -> 单词
	+ 国家/地区 -> 城市
* 使用 “ 句子开头 ” 标记，以便采样可以无需
选择起始字母
* 通过更大和/或形状更好的网络获得更好的结果



	+ 尝试
	 `nn.LSTM`
	 和
	 `nn.GRU`
	 层
	+ 将多个 RNN 组合为更高级别的网络



**脚本总运行时间:** 
 ( 12 分 58.984 秒)
