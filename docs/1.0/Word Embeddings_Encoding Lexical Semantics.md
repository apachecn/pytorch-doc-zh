# 词嵌入：编码形式的词汇语义

词嵌入是一种由真实数字组成的稠密向量，每个向量都代表了单词表里的一个单词. 在自然语言处理中，总会遇到这样的情况：特征全是单词！ 但是，如何在电脑上表述一个单词呢？你在电脑上存储的单词的ascii码，但是它仅仅代表单词怎么拼写，没有说明单词的内在含义(你也许能够从词缀中了解它的词性，或者从大小写中得到一些属性，但仅此而已). 更重要的是，你能把这些ascii码字符组合成什么含义？? 当V代表词汇表、输入数据是|V|维的情况下，我们往往想从神经网络中得到数据密集的结果，但是结果只有很少的几个维度（例如，预测的数据只有几个标签时）。我们如何从大的数据维度空间中得到稍小一点的维度空间？

放弃使用ascii码字符的形式表示单词，换用one-hot encoding会怎么样了？好吧，W这个单词就能这样表示：

cf775cf1814914c00f5bf7ada7de4369.gif

其中，1就是表示w的独有位置，其他位置全是0。其他的词都类似，在另外不一样的位置有一个1代表它，其他位置也都是0。
这种表达除了占用巨大的空间外，还有个很大的缺陷。 它只是简单的把词看做一个单独个体，认为它们之间毫无联系。 我们真正想要的是能够表达单词之间一些相似的含义。为什么要这样做呢？来看下面的例子：

假如我们正在搭建一个语言模型，训练数据有下面一些句子：

*   The mathematician ran to the store.
*   The physicist ran to the store.
*   The mathematician solved the open problem.

现在又得到一个没见过的新句子:

*   The physicist solved the open problem.

我们的模型可能在这个句子上表现的还不错，但是，如果利用了下面两个事实，模型会表现更佳：

*   我们发现数学家和物理学家在句子里有相同的作用，所以在某种程度上，他们有语义的联系。
*   当看见物理学家在新句子中的作用时，我们发现数学家也有起着相同的作用。

然后我们就推测，物理学家在上面的句子里也类似于数学家吗？ 这就是我们所指的相似性理念： 指的是语义相似，而不是简单的拼写相似。 这就是一种通过连接我们发现的和没发现的一些内容相似点、用于解决语言数据稀疏性的技术。 这个例子依赖于一个基本的语言假设： 那些在相似语句中出现的单词，在语义上也是相互关联的。 这就叫做 [distributional hypothesis（分布式假设）](https://en.wikipedia.org/wiki/Distributional_semantics)。

## Getting Dense Word Embeddings（密集词嵌入）

我们如何解决这个问题呢？也就是，怎么编码单词中的语义相似性？ 也许我们会想到一些语义属性。 举个例子，我们发现数学家和物理学家都能跑， 所以也许可以给含有“能跑”语义属性的单词打高分，考虑一下其他的属性，想象一下你可能会在这些属性上给普通的单词打什么分。

如果每个属性都表示一个维度，那我们也许可以用一个向量表示一个单词，就像这样：

```py
\[ q_\text{mathematician} = \left[ \overbrace{2.3}^\text{can run}, \overbrace{9.4}^\text{likes coffee}, \overbrace{-5.5}^\text{majored in Physics}, \dots \right]\]
```

```py
\[ q_\text{physicist} = \left[ \overbrace{2.5}^\text{can run}, \overbrace{9.1}^\text{likes coffee}, \overbrace{6.4}^\text{majored in Physics}, \dots \right]\]
```

那么，我们就这可以通过下面的方法得到这些单词之间的相似性：

```py
\[\text{Similarity}(\text{physicist}, \text{mathematician}) = q_\text{physicist} \cdot q_\text{mathematician}\]
```

尽管通常情况下需要进行长度归一化：

```py
\[ \text{Similarity}(\text{physicist}, \text{mathematician}) = \frac{q_\text{physicist} \cdot q_\text{mathematician}} {\| q_\text{\physicist} \| \| q_\text{mathematician} \|} = \cos (\phi)\]
```

Φ是两个向量的夹角。 这就意味着，完全相似的单词相似度为1。完全不相似的单词相似度为-1。

你可以把本章开头介绍的one-hot稀疏向量看做是我们新定义向量的一种特殊形式，那里的单词相似度为0， 现在我们给每个单词一些独特的语义属性。 这些向量数据密集，也就是说它们数字通常都非零。

But these new vectors are a big pain: you could think of thousands of different semantic attributes that might be relevant to determining similarity, and how on earth would you set the values of the different attributes? Central to the idea of deep learning is that the neural network learns representations of the features, rather than requiring the programmer to design them herself. So why not just let the word embeddings be parameters in our model, and then be updated during training? This is exactly what we will do. We will have some _latent semantic attributes_ that the network can, in principle, learn. Note that the word embeddings will probably not be interpretable. That is, although with our hand-crafted vectors above we can see that mathematicians and physicists are similar in that they both like coffee, if we allow a neural network to learn the embeddings and see that both mathematicians and physicists have a large value in the second dimension, it is not clear what that means. They are similar in some latent semantic dimension, but this probably has no interpretation to us.

In summary, **word embeddings are a representation of the *semantics* of a word, efficiently encoding semantic information that might be relevant to the task at hand**. You can embed other things too: part of speech tags, parse trees, anything! The idea of feature embeddings is central to the field.

## Word Embeddings in Pytorch

Before we get to a worked example and an exercise, a few quick notes about how to use embeddings in Pytorch and in deep learning programming in general. Similar to how we defined a unique index for each word when making one-hot vectors, we also need to define an index for each word when using embeddings. These will be keys into a lookup table. That is, embeddings are stored as a `\(|V| \times D\)` matrix, where `\(D\)` is the dimensionality of the embeddings, such that the word assigned index `\(i\)` has its embedding stored in the `\(i\)`’th row of the matrix. In all of my code, the mapping from words to indices is a dictionary named word_to_ix.

The module that allows you to use embeddings is torch.nn.Embedding, which takes two arguments: the vocabulary size, and the dimensionality of the embeddings.

To index into this table, you must use torch.LongTensor (since the indices are integers, not floats).

```py
# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

```

```py
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

```

Out:

```py
tensor([[ 0.6614,  0.2669,  0.0617,  0.6213, -0.4519]],
       grad_fn=<EmbeddingBackward>)

```

## An Example: N-Gram Language Modeling

Recall that in an n-gram language model, given a sequence of words `\(w\)`, we want to compute

```py
\[P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )\]
```

Where `\(w_i\)` is the ith word of the sequence.

In this example, we will compute the loss function on some training examples and update the parameters with backpropagation.

```py
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:

        # Step 1\. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2\. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3\. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4\. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5\. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!

```

Out:

```py
[(['When', 'forty'], 'winters'), (['forty', 'winters'], 'shall'), (['winters', 'shall'], 'besiege')]
[518.6343855857849, 516.0739576816559, 513.5321269035339, 511.0085496902466, 508.5003893375397, 506.0077188014984, 503.52977323532104, 501.06553316116333, 498.6121823787689, 496.16915798187256]

```

## Exercise: Computing Word Embeddings: Continuous Bag-of-Words

The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep learning. It is a model that tries to predict words given the context of a few words before and a few words after the target word. This is distinct from language modeling, since CBOW is not sequential and does not have to be probabilistic. Typcially, CBOW is used to quickly train word embeddings, and these embeddings are used to initialize the embeddings of some more complicated model. Usually, this is referred to as _pretraining embeddings_. It almost always helps performance a couple of percent.

The CBOW model is as follows. Given a target word `\(w_i\)` and an `\(N\)` context window on each side, `\(w_{i-1}, \dots, w_{i-N}\)` and `\(w_{i+1}, \dots, w_{i+N}\)`, referring to all context words collectively as `\(C\)`, CBOW tries to minimize

```py
\[-\log p(w_i | C) = -\log \text{Softmax}(A(\sum_{w \in C} q_w) + b)\]
```

where `\(q_w\)` is the embedding for word `\(w\)`.

Implement this model in Pytorch by filling in the class below. Some tips:

*   Think about which parameters you need to define.
*   Make sure you know what shape each operation expects. Use .view() if you need to reshape.

```py
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])

class CBOW(nn.Module):

    def __init__(self):
        pass

    def forward(self, inputs):
        pass

# create your model and train.  here are some functions to help you make
# the data ready for use by your module

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

make_context_vector(data[0][0], word_to_ix)  # example

```

Out:

```py
[(['We', 'are', 'to', 'study'], 'about'), (['are', 'about', 'study', 'the'], 'to'), (['about', 'to', 'the', 'idea'], 'study'), (['to', 'study', 'idea', 'of'], 'the'), (['study', 'the', 'of', 'a'], 'idea')]

```

**Total running time of the script:** ( 0 minutes 0.568 seconds)

[`Download Python source code: word_embeddings_tutorial.py`](../../_downloads/8807094f6210189fde9923211274dc82/word_embeddings_tutorial.py)[`Download Jupyter notebook: word_embeddings_tutorial.ipynb`](../../_downloads/e6a250a908acf3362a7ae511adf55881/word_embeddings_tutorial.ipynb)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.readthedocs.io)