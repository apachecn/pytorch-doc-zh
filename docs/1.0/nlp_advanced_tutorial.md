# 高级：制定动态决策和Bi-LSTM CRF

> 作者：[PyTorch](https://github.com/pytorch)
>
> 译者：[ApacheCN](https://github.com/apachecn)
>
> 校对者：[enningxie](https://github.com/enningxie)

## 动态与静态深度学习工具包

Pytorch是一种 _动态_ 神经网络套件。另一个动态套件的例子是 [Dynet](https://github.com/clab/dynet) (我之所以提到这一点，因为与Pytorch和Dynet一起使用是相似的。如果你在Dynet中看到一个例子，它可能会帮助你在Pytorch中实现它）。相反的是 _静态_ 工具包，其中包括Theano，Keras，TensorFlow等。核心区别如下：

*   在静态工具包中，您可以定义一次计算图，对其进行编译，然后将实例流式传输给它。
*   在动态工具包中，为每个实例定义计算图。它永远不会被编译并且是即时执行的。

在没有很多经验的情况下，很难理解其中的差异。一个例子是假设我们想要构建一个深层组成解析器。假设我们的模型大致涉及以下步骤：

*   我们自下而上建造树
*   标记根节点(句子的单词）
*   从那里，使用神经网络和单词的嵌入来找到形成组成部分的组合。每当你形成一个新的成分时，使用某种技术来嵌入成分。在这种情况下，我们的网络架构将完全取决于输入句子。在“绿猫划伤墙”一句中，在模型中的某个点上，我们想要结合跨度 $$(i,j,r) = (1, 3, \text{NP})$$(即，NP组成部分跨越单词1到单词3，在这种情况下是“绿猫” )。

然而，另一句话可能是“某处，大肥猫划伤了墙”。在这句话中，我们希望在某个时刻形成组成 $$(2, 4, NP)$$。我们想要形成的成分将取决于实例。如果我们只编译计算图一次，就像在静态工具包中那样，编写这个逻辑将是非常困难或不可能的。但是，在动态工具包中，不仅有1个预定义的计算图。每个实例都可以有一个新的计算图，所以这个问题就消失了。

动态工具包还具有易于调试和代码更接近宿主语言的优点(我的意思是Pytorch和Dynet看起来更像是比Keras或Theano更实际的Python代码）。

## Bi-LSTM条件随机场讨论

对于本节，我们将看到用于命名实体识别的Bi-LSTM条件随机场的完整复杂示例。上面的LSTM标记符通常足以用于词性标注，但是像CRF这样的序列模型对于NER上的强大性能非常重要。假设熟悉CRF。虽然这个名字听起来很可怕，但所有模型都是CRF，但是LSTM提供了这些功能。这是一个高级模型，比本教程中的任何早期模型复杂得多。如果你想跳过它，那很好。要查看您是否准备好，请查看是否可以：

*   在步骤i中为标记k写出维特比变量的递归。
*   修改上述重复以计算转发变量。
*   再次修改上面的重复计算以计算日志空间中的转发变量(提示：log-sum-exp）

如果你可以做这三件事，你应该能够理解下面的代码。回想一下，CRF计算条件概率。设 $$y$$ 为标签序列，$$x$$ 为字的输入序列。然后我们计算

$$P(y|x)=\frac{\exp{(\text {Score}(x，y）})} {\sum_ {y'} \exp {(\text {Score}(x，y')})} $$

通过定义一些对数电位 $$\log\psi_i(x,y)$$ 来确定得分

$$\text {Score}(x，y)= \sum_i\log\psi_i(x，y)$$

为了使分区功能易于处理，电位必须仅查看局部特征。

在Bi-LSTM CRF中，我们定义了两种潜力：发射和过渡。索引 $$i$$ 处的单词的发射电位来自时间步长 $$i$$ 处的Bi-LSTM的隐藏状态。转换分数存储在 $$|T|x|T|$$ 矩阵 $$\textbf{P}$$ 中，其中 $$T$$ 是标记集。在我的实现中，$$\textbf{P}_{j,k}$$ 是从标签 $$  $$ 转换到标签 $$ j $$ 的分数。所以：

$$
\begin{align}
\text{Score}(x,y) &= \sum_i \log \psi_\text{EMIT}(y_i \rightarrow x_i) + \log \psi_\text{TRANS}(y_{i-1} \rightarrow y_i)\\
&= \sum_i h_i[y_i] + \textbf{P}_{y_i, y_{i-1}}\\
\end{align}
$$

在第二个表达式中，我们将标记视为分配了唯一的非负索引。

如果上面的讨论过于简短，你可以查看[这个](http://www.cs.columbia.edu/%7Emcollins/crf.pdf)从迈克尔柯林斯那里写的关于CRF的文章。

## 实施说明

下面的示例实现了日志空间中的前向算法来计算分区函数，以及用于解码的维特比算法。反向传播将自动为我们计算梯度。我们不需要手工做任何事情。

实施未优化。如果您了解发生了什么，您可能会很快发现在前向算法中迭代下一个标记可能是在一个大的操作中完成的。我想编码更具可读性。如果您想进行相关更改，可以将此标记器用于实际任务。

```
# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

```

帮助程序的功能是使代码更具可读性。

```
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

```

创建模型

```
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

```

进行训练

```
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1\. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2\. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3\. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4\. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!

```

日期：

```
(tensor(2.6907), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1])
(tensor(20.4906), [0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])

```

## 练习：区分标记的新损失函数

我们没有必要在进行解码时创建计算图，因为我们不会从维特比路径得分反向传播。因为无论如何我们都有它，尝试训练标记器，其中损失函数是维特比路径得分和金标准路径得分之间的差异。应该清楚的是，当预测的标签序列是正确的标签序列时，该功能是非负的和0。这基本上是 _结构感知器_。

由于已经实现了Viterbi和score_sentence，因此这种修改应该很短。这是取决于训练实例的计算图形_的形状的示例。虽然我没有尝试在静态工具包中实现它，但我想它可能但不那么直截了当。

拿起一些真实数据并进行比较！
