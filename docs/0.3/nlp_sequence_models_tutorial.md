# 序列模型和 LSTM 网络(长短记忆网络）

> 译者：[@JingTao](https://github.com/jingwangfei)、[@friedhelm739](https://github.com/friedhelm739)

之前我们已经学过了许多的前馈网络. 所谓前馈网络, 就是网络中不会保存状态. 然而有时 这并不是我们想要的效果. 在自然语言处理 (NLP, Natural Language Processing) 中, 序列模型是一个核心的概念. 所谓序列模型, 即输入依赖于时间信息的模型. 一个典型 的序列模型是隐马尔科夫模型 (HMM, Hidden Markov Model). 另一个序列模型的例子 是条件随机场 (CRF, Conditional Random Field).

递归神经网络是指可以保存某种状态的神经网络. 比如说, 网络上个时刻的输出可以作为下个 时刻的输入, 这样信息就可以通过序列在网络中一直往后传递. 对于LSTM (Long-Short Term Memory) 来说, 序列中的每个元素都有一个相应的隐状态 ![h_t](img/tex-6c4ff69dbcc329835a33b80fe3a145c7.gif), 该隐状态 原则上可以包含序列当前结点之前的任一节点的信息. 我们可以使用隐藏状态来预测语言模型 中的单词, 词性标签以及其他各种各样的东西.

## Pytorch 中的 LSTM

开始例子之前,有几个点说明一下. Pytorch 中, LSTM 的所有的形式固定为3D 的 tensor. 每个维度有固定的语义含义, 不能乱掉. 其中第一维是序列本身, 第二维以 mini-batch 形式 来索引实例, 而第三维则索引输入的元素. 因为我们没有讨论过 mini-batch, 所以在这里我们 假设第二维的维度总是1\. 如果我们想在句子 “The cow jumped” 上运行一个序列模型, 模型 的输入类似这样:

![\begin{split}\begin{bmatrix} \overbrace{q_\text{The}}^\text{row vector} \\ q_\text{cow} \\ q_\text{jumped} \end{bmatrix}\end{split}](img/tex-48898c98ae1c479f5e5743a4ed94ff85.gif)

除了有一个额外的大小为1的第二维度.

此外, 你还可以向网络逐个输入序列, 在这种情况下, 第一个轴的大小也是1.

来看一个简单的例子.

```py
# 作者: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

```

```py
lstm = nn.LSTM(3, 3)  # 输入维度是3, 输出维度也是3
inputs = [autograd.Variable(torch.randn((1, 3)))
          for _ in range(5)]  # 构造一个长度为5的序列

# 初始化隐藏状态
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
          autograd.Variable(torch.randn((1, 1, 3))))
for i in inputs:
    # 将序列的元素逐个输入到LSTM
    # 经过每步操作,hidden 的值包含了隐藏状态的信息
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# 另外, 我们还可以一次对整个序列进行训练. LSTM 返回的第一个值表示所有时刻的隐状态值,
# 第二个值表示最近的隐状态值 (因此下面的 "out"的最后一个值和 "hidden" 的值是一样的).
# 之所以这样设计, 是为了通过 "out" 的值来获取所有的隐状态值, 而用 "hidden" 的值来
# 进行序列的反向传播运算, 具体方式就是将它作为参数传入后面的 LSTM 网络.

# 增加额外的第二个维度
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(
    torch.randn((1, 1, 3))))  # 清空输出隐状态
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)

```

## 例子: 用 LSTM 来进行词性标注

在这部分, 我们将会使用一个 LSTM 网络来进行词性标注. 在这里我们不会用到维特比算法, 前向后向算法或者任何类似的算法, 而是将这部分内容作为一个 (有挑战) 的练习留给读者, 希望读者在了解了这部分的内容后能够实现如何将维特比算法应用到 LSTM 网络中来.

整个模型的参数定义如下: 输入的句子定义为 ![w_1, \dots, w_M](img/tex-ee8a819016d6a550cab1f4c74f4ab6e5.gif), 其中动词定义 为 ![w_1, \dots, w_M](img/tex-ee8a819016d6a550cab1f4c74f4ab6e5.gif), 标签集合定义为 ![T](img/tex-b9ece18c950afbfa6b0fdbfa4ff731d3.gif), 单词 ![w_i](img/tex-aa38f107289d4d73d516190581397349.gif) 的实际 标签为 ![y_i](img/tex-8d62e469fb30ed435a668eb5c035b1f6.gif). 定义单词 ![w_i](img/tex-aa38f107289d4d73d516190581397349.gif) 的预测标签为 ![\hat{y}_i](img/tex-99600e95adcc97c986df86d4b612cf52.gif).

这是一个结构预测模型, 我们的输出是一个序列 ![\hat{y}_1, \dots, \hat{y}_M](img/tex-efc974dfec8352388b7f29d324a914fa.gif), 其中 ![\hat{y}_i \in T](img/tex-56b37e7a393cf3c9d2cf53b75da99b8e.gif).

在进行预测时, 需将句子每个词输入到一个 LSTM 网络中. 将时刻 ![i](img/tex-865c0c0b4ab0e063e5caa3387c1a8741.gif) 的隐状态标记 为 ![h_i](img/tex-57339b77b3af15427b7154f4daf8a223.gif). 同样地, 对每个标签赋一个独一无二的索引 (类似 word embeddings 部分 word_to_ix 的设置). 然后就得到了 ![\hat{y}_i](img/tex-99600e95adcc97c986df86d4b612cf52.gif) 的预测规则:

![\hat{y}_i = \text{argmax}_j \ (\log \text{Softmax}(Ah_i + b))_j](img/tex-edd60118caf4e12048613a1e39235fbe.gif)

即先对隐状态进行一个仿射变换, 然后计算一个对数 softmax, 最后得到的预测标签即为对数 softmax 中最大的值对应的标签. 注意, 这也意味着 ![A](img/tex-7fc56270e7a70fa81a5935b72eacbe29.gif) 空间的维度是 ![|T|](img/tex-d661bcf5071c228c5d4e72a7a34b0ff4.gif).

准备数据:

```py
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# 实际中通常使用更大的维度如32维, 64维.
# 这里我们使用小的维度, 为了方便查看训练过程中权重的变化.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

```

构造模型:

```py
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        #  LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # 线性层将隐状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 开始时刻, 没有隐状态
        # 关于维度设置的详情,请参考 Pytorch 文档
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

```

训练模型:

```py
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 查看下训练前得分的值
# 注意: 输出的 i,j 元素的值表示单词 i 的 j 标签的得分
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(300):  # 再次说明下, 实际情况下你不会训练300个周期, 此例中我们只是构造了一些假数据
    for sentence, tags in training_data:
        # Step 1\. 请记住 Pytorch 会累加梯度
        # 每次训练前需要清空梯度值
        model.zero_grad()

        # 此外还需要清空 LSTM 的隐状态
        # 将其从上个实例的历史中分离出来
        model.hidden = model.init_hidden()

        # Step 2\. 准备网络输入, 将其变为词索引的 Variables 类型数据
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3\. 前向传播
        tag_scores = model(sentence_in)

        # Step 4\. 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# 查看训练后得分的值
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# 句子是 "the dog ate the apple", i,j 表示对于单词 i, 标签 j 的得分.
# 我们采用得分最高的标签作为预测的标签. 从下面的输出我们可以看到, 预测得
# 到的结果是0 1 2 0 1\. 因为 索引是从0开始的, 因此第一个值0表示第一行的
# 最大值, 第二个值1表示第二行的最大值, 以此类推. 所以最后的结果是 DET
# NOUN VERB DET NOUN, 整个序列都是正确的!
print(tag_scores)

```

## 练习: 使用字符级特征来增强 LSTM 词性标注器

在上面的例子中, 每个词都有一个词嵌入, 作为序列模型的输入. 接下来让我们使用每个的单词的 字符级别的表达来增强词嵌入. 我们期望这个操作对结果能有显著提升, 因为像词缀这样的字符级 信息对于词性有很大的影响. 比如说, 像包含词缀 _-ly_ 的单词基本上都是被标注为副词.

具体操作如下. 用 ![c_w](img/tex-8038e14a2604b0c99b2a9481de35e3a5.gif) 来表示单词 ![w](img/tex-f1290186a5d0b1ceab27f4e77c0c5d68.gif) 的字符级表达, 同之前一样, 我们使 用 ![x_w](img/tex-5a7fa14dac5c9d494aa75b18e1ad9a51.gif) 来表示词嵌入. 序列模型的输入就变成了 ![x_w](img/tex-5a7fa14dac5c9d494aa75b18e1ad9a51.gif) 和 ![c_w](img/tex-8038e14a2604b0c99b2a9481de35e3a5.gif) 的拼接. 因此, 如果 ![x_w](img/tex-5a7fa14dac5c9d494aa75b18e1ad9a51.gif) 的维度是5, ![c_w](img/tex-8038e14a2604b0c99b2a9481de35e3a5.gif) 的维度是3, 那么我们的 LSTM 网络的输入维度大小就是8.

为了得到字符级别的表达, 将单词的每个字符输入一个 LSTM 网络, 而 ![c_w](img/tex-8038e14a2604b0c99b2a9481de35e3a5.gif) 则为这个 LSTM 网络最后的隐状态. 一些提示:

*   新模型中需要两个 LSTM, 一个跟之前一样, 用来输出词性标注的得分, 另外一个新增加的用来 获取每个单词的字符级别表达.
*   为了在字符级别上运行序列模型, 你需要用嵌入的字符来作为字符 LSTM 的输入.
