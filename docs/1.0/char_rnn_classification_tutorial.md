

# Classifying Names with a Character-Level RNN
# 使用字符级别特征的RNN网络进行单词分类

**Author**: [Sean Robertson](https://github.com/spro/practical-pytorch)
**作者**: [Sean Robertson](https://github.com/spro/practical-pytorch)

We will be building and training a basic character-level RNN to classify words. A character-level RNN reads words as a series of characters - outputting a prediction and “hidden state” at each step, feeding its previous hidden state into each next step. We take the final prediction to be the output, i.e. which class the word belongs to.
我们将构建和训练字符级RNN来对单词进行分类。 字符级RNN将单词作为一系列字符读取，在每一步输出预测和“隐藏状态”，将其先前的隐藏状态输入至下一时刻。 我们将最终时刻输出作为预测结果，即表示该词属于哪个类。
Specifically, we’ll train on a few thousand surnames from 18 languages of origin, and predict which language a name is from based on the spelling:
具体来说，我们将在18种语言构成的几千个姓氏数据集上训练模型，根据一个单词的拼写预测它是哪种语言的姓氏：

```py
$ python predict.py Hinton
(-0.47) Scottish
(-1.52) English
(-3.57) Irish

$ python predict.py Schmidhuber
(-0.19) German
(-2.48) Czech
(-2.68) Dutch

```

**Recommended Reading:**
**阅读建议:**

I assume you have at least installed PyTorch, know Python, and understand Tensors:
我默认你已经安装好了PyTorch，熟悉Python语言，理解“张量”的概念：

*   [https://pytorch.org/](https://pytorch.org/) For installation instructions
*   [https://pytorch.org/](https://pytorch.org/) PyTorch安装指南
*   [Deep Learning with PyTorch: A 60 Minute Blitz](../beginner/deep_learning_60min_blitz.html) to get started with PyTorch in general
*   [Deep Learning with PyTorch: A 60 Minute Blitz](../beginner/deep_learning_60min_blitz.html) to get started with PyTorch in general
*   [Learning PyTorch with Examples](../beginner/pytorch_with_examples.html) for a wide and deep overview
*   [PyTorch for Former Torch Users](../beginner/former_torchies_tutorial.html) if you are former Lua Torch user

It would also be useful to know about RNNs and how they work:
事先学习并了解RNN的工作原理对理解这个例子十分有帮助:

*   [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) shows a bunch of real life examples
*   [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) is about LSTMs specifically but also informative about RNNs in general

## Preparing the Data
## 准备数据


Download the data from [here](https://download.pytorch.org/tutorial/data.zip) and extract it to the current directory.
点击这里[下载数据](https://download.pytorch.org/tutorial/data.zip) 并将其解压到当前文件夹。

Included in the `data/names` directory are 18 text files named as “[Language].txt”. Each file contains a bunch of names, one name per line, mostly romanized (but we still need to convert from Unicode to ASCII).

We’ll end up with a dictionary of lists of names per language, `{language: [names ...]}`. The generic variables “category” and “line” (for language and name in our case) are used for later extensibility.

```py
from __future__ import unicode_literals, print_function, division
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
# 按行读取文件
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

```

Out:
输出:

```py
['data/names/Italian.txt', 'data/names/German.txt', 'data/names/Portuguese.txt', 'data/names/Chinese.txt', 'data/names/Greek.txt', 'data/names/Polish.txt', 'data/names/French.txt', 'data/names/English.txt', 'data/names/Spanish.txt', 'data/names/Arabic.txt', 'data/names/Czech.txt', 'data/names/Russian.txt', 'data/names/Irish.txt', 'data/names/Dutch.txt', 'data/names/Scottish.txt', 'data/names/Vietnamese.txt', 'data/names/Korean.txt', 'data/names/Japanese.txt']
Slusarski

```

Now we have `category_lines`, a dictionary mapping each category (language) to a list of lines (names). We also kept track of `all_categories` (just a list of languages) and `n_categories` for later reference.

```py
print(category_lines['Italian'][:5])

```

Out:
输出:

```py
['Abandonato', 'Abatangelo', 'Abatantuono', 'Abate', 'Abategiovanni']

```

### Turning Names into Tensors

Now that we have all the names organized, we need to turn them into Tensors to make any use of them.

To represent a single letter, we use a “one-hot vector” of size `&lt;1 x n_letters&gt;`. A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. `"b" = &lt;0 1 0 0 0 ...&gt;`.

To make a word we join a bunch of those into a 2D matrix `&lt;line_length x 1 x n_letters&gt;`.

That extra 1 dimension is because PyTorch assumes everything is in batches - we’re just using a batch size of 1 here.

```py
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

Out:

```py
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0.]])
torch.Size([5, 1, 57])

```

## Creating the Network

Before autograd, creating a recurrent neural network in Torch involved cloning the parameters of a layer over several timesteps. The layers held hidden state and gradients which are now entirely handled by the graph itself. This means you can implement a RNN in a very “pure” way, as regular feed-forward layers.

This RNN module (mostly copied from [the PyTorch for Torch users tutorial](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#example-2-recurrent-net)) is just 2 linear layers which operate on an input and hidden state, with a LogSoftmax layer after the output.

![](img/592fae78143370fffc1d0c7957706384.jpg)

```py
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

```

To run a step of this network we need to pass an input (in our case, the Tensor for the current letter) and a previous hidden state (which we initialize as zeros at first). We’ll get back the output (probability of each language) and a next hidden state (which we keep for the next step).

```py
input = letterToTensor('A')
hidden =torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)

```

For the sake of efficiency we don’t want to be creating a new Tensor for every step, so we will use `lineToTensor` instead of `letterToTensor` and use slices. This could be further optimized by pre-computing batches of Tensors.

```py
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)

```

Out:

```py
tensor([[-2.8857, -2.9005, -2.8386, -2.9397, -2.8594, -2.8785, -2.9361, -2.8270,
         -2.9602, -2.8583, -2.9244, -2.9112, -2.8545, -2.8715, -2.8328, -2.8233,
         -2.9685, -2.9780]], grad_fn=<LogSoftmaxBackward>)

```

As you can see the output is a `&lt;1 x n_categories&gt;` Tensor, where every item is the likelihood of that category (higher is more likely).

## Training

### Preparing for Training

Before going into training we should make a few helper functions. The first is to interpret the output of the network, which we know to be a likelihood of each category. We can use `Tensor.topk` to get the index of the greatest value:

```py
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))

```

Out:

```py
('Vietnamese', 15)

```

We will also want a quick way to get a training example (a name and its language):

```py
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

Out:

```py
category = Russian / line = Minkin
category = French / line = Masson
category = German / line = Hasek
category = Dutch / line = Kloeten
category = Scottish / line = Allan
category = Italian / line = Agostini
category = Japanese / line = Fumihiko
category = Polish / line = Gajos
category = Scottish / line = Duncan
category = Arabic / line = Gerges

```

### Training the Network

Now all it takes to train this network is show it a bunch of examples, have it make guesses, and tell it if it’s wrong.

For the loss function `nn.NLLLoss` is appropriate, since the last layer of the RNN is `nn.LogSoftmax`.

```py
criterion = nn.NLLLoss()

```

Each loop of training will:

*   Create input and target tensors
*   Create a zeroed initial hidden state
*   Read each letter in and
    *   Keep hidden state for next letter
*   Compare final output to target
*   Back-propagate
*   Return the output and loss

```py
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
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

```

Now we just have to run that with a bunch of examples. Since the `train` function returns both the output and loss we can print its guesses and also keep track of loss for plotting. Since there are 1000s of examples we print only every `print_every` examples, and take an average of the loss.

```py
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

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d  %d%% (%s) %.4f  %s / %s  %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

```

Out:

```py
5000 5% (0m 11s) 2.0318 Jaeger / German ✓
10000 10% (0m 18s) 2.1296 Sokolofsky / Russian ✗ (Polish)
15000 15% (0m 26s) 1.2620 Jo / Korean ✓
20000 20% (0m 34s) 1.9295 Livson / Scottish ✗ (Russian)
25000 25% (0m 41s) 1.2325 Fortier / French ✓
30000 30% (0m 49s) 2.5714 Purdes / Dutch ✗ (Czech)
35000 35% (0m 56s) 2.3312 Bayer / Arabic ✗ (German)
40000 40% (1m 4s) 2.3792 Mitchell / Dutch ✗ (Scottish)
45000 45% (1m 12s) 1.3536 Maes / Dutch ✓
50000 50% (1m 20s) 2.6095 Sai / Chinese ✗ (Vietnamese)
55000 55% (1m 28s) 0.5883 Cheung / Chinese ✓
60000 60% (1m 35s) 1.5788 William / Irish ✓
65000 65% (1m 43s) 2.5809 Mulder / Scottish ✗ (Dutch)
70000 70% (1m 51s) 1.3440 Bruce / German ✗ (Scottish)
75000 75% (1m 58s) 1.1839 Romero / Italian ✗ (Spanish)
80000 80% (2m 6s) 2.6453 Reyes / Portuguese ✗ (Spanish)
85000 85% (2m 14s) 0.0290 Mcmillan / Scottish ✓
90000 90% (2m 22s) 0.7337 Riagan / Irish ✓
95000 95% (2m 30s) 2.6208 Maneates / Dutch ✗ (Greek)
100000 100% (2m 37s) 0.5170 Szwarc / Polish ✓

```

### Plotting the Results

Plotting the historical loss from `all_losses` shows the network learning:

```py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

```

![https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_classification_tutorial_001.png](img/cc57a36a43d450df4bfc1d1d1b1ce274.jpg)

## Evaluating the Results

To see how well the network performs on different categories, we will create a confusion matrix, indicating for every actual language (rows) which language the network guesses (columns). To calculate the confusion matrix a bunch of samples are run through the network with `evaluate()`, which is the same as `train()` minus the backprop.

```py
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

![https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_classification_tutorial_002.png](img/029a9d26725997aae97e9e3f6f10067f.jpg)

You can pick out bright spots off the main axis that show which languages it guesses incorrectly, e.g. Chinese for Korean, and Spanish for Italian. It seems to do very well with Greek, and very poorly with English (perhaps because of overlap with other languages).

### Running on User Input

```py
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
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

Out:

```py
> Dovesky
(-0.74) Russian
(-0.77) Czech
(-3.31) English

> Jackson
(-0.80) Scottish
(-1.69) English
(-1.84) Russian

> Satoshi
(-1.16) Japanese
(-1.89) Arabic
(-1.90) Polish

```

The final versions of the scripts [in the Practical PyTorch repo](https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification) split the above code into a few files:

*   `data.py` (loads files)
*   `model.py` (defines the RNN)
*   `train.py` (runs training)
*   `predict.py` (runs `predict()` with command line arguments)
*   `server.py` (serve prediction as a JSON API with bottle.py)

Run `train.py` to train and save the network.

Run `predict.py` with a name to view predictions:

```py
$ python predict.py Hazaki
(-0.42) Japanese
(-1.39) Polish
(-3.51) Czech

```

Run `server.py` and visit [http://localhost:5533/Yourname](http://localhost:5533/Yourname) to get JSON output of predictions.

## Exercises

*   Try with a different dataset of line -&gt; category, for example:
    *   Any word -&gt; language
    *   First name -&gt; gender
    *   Character name -&gt; writer
    *   Page title -&gt; blog or subreddit
*   Get better results with a bigger and/or better shaped network
    *   Add more linear layers
    *   Try the `nn.LSTM` and `nn.GRU` layers
    *   Combine multiple of these RNNs as a higher level network

**Total running time of the script:** ( 2 minutes 45.719 seconds)

[`Download Python source code: char_rnn_classification_tutorial.py`](../_downloads/ccb15f8365bdae22a0a019e57216d7c6/char_rnn_classification_tutorial.py)[`Download Jupyter notebook: char_rnn_classification_tutorial.ipynb`](../_downloads/977c14818c75427641ccb85ad21ed6dc/char_rnn_classification_tutorial.ipynb)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.readthedocs.io)

