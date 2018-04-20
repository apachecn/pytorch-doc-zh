# -*- coding: utf-8 -*-
r"""
PyTorch深度学习
**************************

深度学习构建模块: Affine maps, non-linearities and objectives
==========================================================================

深度学习由以巧妙的方式组合non-linearities的linearities组成的.non-linearities的引入允许强大的模型. 在本节中,我们将使用这些核心组件,构建一个objective函数,并且看看模型是如何训练的.


Affine Maps
~~~~~~~~~~~

深度学习的核心工作之一是affine map, 这是一个函数 :math:`f(x)` 其中

.. math::  f(x) = Ax + b

对于矩阵 :math:`A` 和向量 :math:`x, b`. 这里学习的参数是 :math:`A` and :math:`b`. 通常,:math:`b` 被称为 *偏差* 项.


Pytorch和大多数其他深度学习框架与传统的线性代数有所不同. 它映射输入的行而不是列. 也就是说, 它映射输入的行而不是列. 也就是说,下面的输出的第 :math:`i` 行是 :math:`A` 的输入的第 :math:`i` 行加上偏置项的映射. 看下面的例子.

"""

# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


######################################################################

lin = nn.Linear(5, 3)  # maps from R^5 to R^3, parameters A, b
# data is 2x5.  A maps from 5 to 3... can we map "data" under A?
data = autograd.Variable(torch.randn(2, 5))
print(lin(data))  # yes


######################################################################
# Non-Linearities
# ~~~~~~~~~~~~~~~
#
# 首先,注意以下事实,这将解释为什么我们首先需要
# non-linearities.假设我们有两个 maps
# :math:`f(x) = Ax + b` and :math:`g(x) = Cx + d`. 什么是
# :math:`f(g(x))`?
#
# .. math::  f(g(x)) = A(Cx + d) + b = ACx + (Ad + b)
#
# :math:`AC` 是一个矩阵, :math:`Ad + b` 是一个向量, 所以我们看到组合affine map给你一个affine map
#
# 由此可以看出,如果你想让你的神经网络成为仿射组合的长链, 那么这不会给你的模型增加新的动力,而只是做一个affine map.
#
# 如果我们在affine层之间引入non-linearities,则不再是这种情况,我们可以构建更强大的模型.
#
# 有一些核心的non-linearities.
# :math:`\tanh(x), \sigma(x), \text{ReLU}(x)` 是最常见的.你可能想知道：“为什么这些功能？我可以想到很多其他的non-linearities.“ 其原因是他们的梯度是eassy计算的,并且计算梯度对学习是必不可少的.例如
#
# .. math::  \frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))
#
# 一个简单的提示：虽然你可能已经在AI class的介绍中学习到了一些神经网络,其中 :math:`\sigma(x) 是默认的non-linearity,但通常人们在实践中会回避它.这是因为随着参数绝对值的增长,梯度会很快*消失*.小梯度意味着很难学习.大多数人默认tanh或ReLU.
#

# In pytorch, most non-linearities are in torch.functional (we have it imported as F)
# Note that non-linearites typically don't have parameters like affine maps do.
# That is, they don't have weights that are updated during training.
data = autograd.Variable(torch.randn(2, 2))
print(data)
print(F.relu(data))


######################################################################
# Softmax and Probabilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 函数 :math:`\text{Softmax}(x)` 也只是一个 non-linearity,但它的特殊之处在于它通常是网络中最后一次完成的操作. 这是因为它接受了一个实数向量并返回一个概率分布. 其定义如下. 让 :math:`x` 是一个实数的向量(正面,负面,不管,没有限制). 然后,第i个 :math:`\text{Softmax}(x)` 的组成是
#
# .. math::  \frac{\exp(x_i)}{\sum_j \exp(x_j)}
#
# 应该清楚的是,输出是一个概率分布：每个元素都是非负的,并且所有组件的总和都是1.
#
# 你也可以把它看作只是将一个元素明确的指数运算符应用于输入,以使所有内容都为非负值,然后除以归一化常数.
#

# Softmax is also in torch.nn.functional
data = autograd.Variable(torch.randn(5))
print(data)
print(F.softmax(data, dim=0))
print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
print(F.log_softmax(data, dim=0))  # theres also log_softmax


######################################################################
# Objective Functions
# ~~~~~~~~~~~~~~~~~~~
#
# objective function是您的网络正在接受培训以最小化的功能(在这种情况下,它通常被称为*损失函数*或*成本函数*).首先选择一个训练实例,通过神经网络运行它,然后计算输出的损失.然后通过采用损失函数的导数来更新模型的参数.直观地说,如果你的模型对答案完全有信心,而且答案是错误的,你的损失就会很高.如果它的答案非常有信心,而且答案是正确的,那么损失就会很低.
#
# 将训练样例的损失函数最小化的想法是,您的网络希望能够很好地概括,并且在开发集,测试集或生产环境中看不见的示例有小的损失. 一个示例损失函数是*负对数似然损失*,这是多类分类的一个非常普遍的目标. 对于有监督的多类别分类,这意味着训练网络以最小化正确输出的负对数概率(或等同地,最大化正确输出的对数概率).
#


######################################################################
# Optimization and Training
# =========================
#
# 那么我们可以计算一个实例的损失函数？我们该怎么做？我们之前看到autograd.变量知道如何计算与计算梯度有关的事物.那么,因为我们的损失是一个autograd.Variable,我们可以计算所有用于计算它的参数的梯度！然后我们可以执行标准渐变更新.让 :math:`\theta` 是我们的参数,:math:`L(\theta)` 损失函数,以及：:math:`\eta`是一个积极的学习率.然后：
#
# .. math::  \theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta L(\theta)
#
# 有很多算法和积极的研究集合,试图做更多的不仅仅是这个香草梯度更新. 许多人试图根据列车时间发生的情况改变学习率. 除非你真的感兴趣,否则你不必担心这些算法具体做什么. Torch提供了许多torch.optim包,它们都是完全透明的. 使用最简单的梯度更新与更复杂的算法相同. 尝试不同的更新算法和更新算法的不同参数(如不同的初始学习速率)对于优化网络性能非常重要. 通常,只需用Adam或RMSProp等优化器替换vanilla SGD即可显着提升性能.
#


######################################################################
# Creating Network Components in Pytorch
# ======================================
#
# 在我们开始关注NLP之前,让我们做一个注释的例子,在Pytorch中使用affine maps和non-linearities构建网络. 我们还将看到如何使用Pytorch建立的负对数似然计算损失函数,并通过反向传播更新参数.
#
# 所有网络组件都应该从nn.Module继承并重写forward()方法. 这就是关于它,就样板而言. 继承自nn.Module为您的组件提供了功能. 例如,它使它跟踪其可训练参数,可以使用.cuda()或.cpu()函数等在CPU和GPU之间交换它,等等.
#
# 我们来编写一个带有注释的网络示例,该网络采用稀疏的词袋表示法,并输出概率分布在两个标签上：“英语”和“西班牙语”. 这个模型只是逻辑回归.
#


######################################################################
# Example: Logistic Regression Bag-of-Words classifier
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 我们的模型将映射一个稀疏的BOW表示来记录标签上的概率. 我们为词汇表中的每个单词分配一个索引. 例如,说我们的整个词汇是两个词“你好”和“世界”,分别指数为0和1. BoW向量为句子“hello hello hello hello”"
# 是
#
# .. math::  \left[ 4, 0 \right]
#
# 对于“hello world world hello”,它是
#
# .. math::  \left[ 2, 2 \right]
#
# 等等.一般来说,它是
#
# .. math::  \left[ \text{Count}(\text{hello}), \text{Count}(\text{world}) \right]
#
# 将这个BOW向量表示为 :math:`x`. 我们的网络输出是：
#
# .. math::  \log \text{Softmax}(Ax + b)
#
# 也就是说,我们通过affine map传递输入,然后记录softmax.
#

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# the model knows its parameters.  The first output below is A, the second is b.
# Whenever you assign a component to a class variable in the __init__ function
# of a module, which was done with the line
# self.linear = nn.Linear(...)
# Then through some Python magic from the Pytorch devs, your module
# (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
for param in model.parameters():
    print(param)

# To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
sample = data[0]
bow_vector = make_bow_vector(sample[0], word_to_ix)
log_probs = model(autograd.Variable(bow_vector))
print(log_probs)


######################################################################
# 以上哪个值对应于英语的日志概率,以及哪个值是西班牙语？ 我们从来没有定义过它,但如果我们想要训练这个东西,我们需要.
#

label_to_ix = {"SPANISH": 0, "ENGLISH": 1}


######################################################################
# 所以,让训练！ 要做到这一点,我们通过实例来获取日志概率,计算损失函数,计算损失函数的梯度,然后用梯度步骤更新参数. 火炬在nn软件包中提供了丢失功能. nn.NLLLoss()是我们想要的负对数似然损失. 它还定义了torch.optim中的优化函数. 在这里,我们只会使用SGD.
#
# 请注意,NLLLoss的*输入*是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率. 这就是为什么我们网络的最后一层是log softmax. 损失函数nn.CrossEntropyLoss()与NLLLoss()相同,只是它为您记录softmax.
#

# Run on test data before we train, just to see a before-and-after
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)

# Print the matrix column corresponding to "creo"
print(next(model.parameters())[:, word_to_ix["creo"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
for epoch in range(100):
    for instance, label in data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Variable as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)

# Index corresponding to Spanish goes up, English goes down!
print(next(model.parameters())[:, word_to_ix["creo"]])


######################################################################
# 我们得到了正确的答案！ 您可以看到,第一个示例中西班牙语的日志概率要高得多,而测试数据的第二个英语日志概率应该高得多.
#
# 现在你看看如何制作一个Pytorch组件,通过它传递一些数据并做梯度更新. 我们准备深入挖掘NLP所能提供的内容.
#
