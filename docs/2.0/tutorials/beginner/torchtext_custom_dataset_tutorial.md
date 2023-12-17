# 使用 Torchtext 预处理自定义文本数据集 [¶](#preprocess-custom-text-dataset-using-torchtext "永久链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
> 人工校正：[xiaoxstz](https://github.com/xiaoxstz)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/torchtext_custom_dataset_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/torchtext_custom_dataset_tutorial.html>

**作者** : [Anupam Sharma](https://anp-scp.github.io/)

 本教程说明了 torchtext 在非内置数据集上的用法。在本教程中，我们将预处理一个数据集，该数据集可进一步用于训练用于机器翻译的序列到序列模型（类似于本教程中的内容： [使用神经网络进行序列到序列学习](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)) 但不使用 torchtext 的旧版本。

 在本教程中，我们将学习如何：

* 读取数据集
* 对句子进行分词
* 对句子应用转换
* 执行存储桶批处理

 假设我们需要准备一个数据集来训练一个可以执行英语到德语翻译的模型。我们将使用由[Tatoeba 项目](https://tatoeba.org/en) 提供的制表符分隔的德语-英语句子对，可以从[此链接](https://tatoeba.org/en) 下载该句子对。 /www.manythings.org/anki/deu-eng.zip) .

 其他语言的句子对可以在 [此链接](https://www.manythings.org/anki/) 中找到 。

## 设置 [¶](#setup "此标题的永久链接")

 首先，下载数据集，解压缩 zip，并记下文件的路径 deu.txt 。

 确保安装了以下软件包:

* [Torchdata 0.6.0](https://pytorch.org/data/beta/index.html)  ( [安装说明](https://github.com/pytorch/data)  )
* [Torchtext 0.15.0](https://pytorch.org/text/stable/index.html)  ( [安装说明](https://github.com/pytorch/text) )
* [Spacy](https://spacy.io/usage)

 在这里，我们使用 Spacy 来标记文本。简而言之，标记化意味着将句子转换为单词列表。 Spacy 是一个用于各种自然语言处理 (NLP) 任务的 Python 包。

 从 Spacy 下载英语和德语模型，如下所示：

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

```

 让我们首先导入所需的模块：

```python
import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
from torchtext.vocab import build_vocab_from_iterator
eng = spacy.load("en_core_web_sm") # Load the English model to tokenize English text
de = spacy.load("de_core_news_sm") # Load the German model to tokenize German text

```

 现在我们将加载数据集

```python
FILE_PATH = 'data/deu.txt'
data_pipe = dp.iter.IterableWrapper([FILE_PATH])
data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)

```

 在上面的代码块中，我们做了以下事情：

1. 在第 2 行，我们正在创建文件名的可迭代
2. 在第 3 行，我们将迭代传递给 FileOpener，然后以读取模式打开文件
3. 在第 4 行，我们调用一个函数来解析文件，该函数再次返回表示制表符分隔文件的每一行的可迭代元组

 DataPipes 可以被认为是一个类似于数据集对象的东西，我们可以在其上执行各种操作。检查 [本教程](https://pytorch.org/data/beta/dp_tutorial.html) 有关DataPipes 的更多详细信息。

 我们可以验证可迭代对象是否具有如下所示的一对句子：

```python
for sample in data_pipe:
    print(sample)
    break

```

输出：

```txt
('Go.', 'Geh.', 'CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #8597805 (Roujin)')

```

 请注意，我们还有归因详细信息以及一对句子。我们将编写一个小函数来删除归因详细信息：

```python
def removeAttribution(row):
 """
 Function to keep the first two elements in a tuple
 """
    return row[:2]
data_pipe = data_pipe.map(removeAttribution)

```

上面代码块中第 6 行的  map 函数可用于对  data_pipe 的每个元素应用某些函数。现在，我们可以验证 data_pipe  仅包含句子对。

```python
for sample in data_pipe:
    print(sample)
    break

```

输出：

```txt
('Go.', 'Geh.')

```

 现在，让我们定义几个函数来执行标记化：

```python
def engTokenize(text):
 """
 Tokenize an English text and return a list of tokens
 """
    return [token.text for token in eng.tokenizer(text)]

def deTokenize(text):
 """
 Tokenize a German text and return a list of tokens
 """
    return [token.text for token in de.tokenizer(text)]

```

 上述函数接受文本并返回单词列表，如下所示：

```python
print(engTokenize("Have a good day!!!"))
print(deTokenize("Haben Sie einen guten Tag!!!"))

```

输出：

```txt
['Have', 'a', 'good', 'day', '!', '!', '!']
['Haben', 'Sie', 'einen', 'guten', 'Tag', '!', '!', '!']

```

## 构建词汇 [¶](#building-the-vocabulary "永久链接到此标题")

 让我们考虑一个英语句子作为源，一个德语句子作为目标。

 词汇表可以被视为我们在数据集中拥有的唯一单词的集合。我们现在将为源和目标构建词汇表。

 让我们定义一个函数来从迭代器中的元组元素中获取标记。

```python
def getTokens(data_iter, place):
 """
 Function to yield tokens from an iterator. Since, our iterator contains
 tuple of sentences (source and target), `place` parameters defines for which
 index to return the tokens for. `place=0` for source and `place=1` for target
 """
    for english, german in data_iter:
        if place == 0:
            yield engTokenize(english)
        else:
            yield deTokenize(german)

```

 现在，我们将为源构建词汇表：

```python
source_vocab = build_vocab_from_iterator(
    getTokens(data_pipe,0),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
source_vocab.set_default_index(source_vocab['<unk>'])

```

 上面的代码从迭代器构建词汇表。在上面的代码块中：

* 在第 2 行，我们使用 place=0 调用 getTokens() 函数，因为我们需要源句子的词汇。
* 在第 3 行，我们设置 min_freq=2 。这意味着，该函数将跳过那些出现次数少于 2 次的单词。
* 在第 4 行，我们指定了一些特殊标记：
  * `<sos>` 句子的开始
  * `<eos>` 句子的结束
  * `<unk>` 表示未知单词。未知单词的一个示例是由于 min_freq=2 而跳过的单词。
  * `<pad>` 是填充令牌。在训练时，我们主要是批量训练模型。在一个批次中，可以有不同长度的句子。因此，我们用`<pad>` 标记填充较短的句子，以使批次中所有序列的长度相等。
* 在第 5 行，我们设置 特别_first=True 。这意味着 `<pad>` 将获得字典中的索引 0、`<sos>` 索引 1、`<eos>`索引 2 ; `<unk>` 将获得字典中的索引 3.
* At line 7, we set default index as index of `<unk>`. That means if some word is not in vocabulary, we will use `<unk>` instead of that unknown word.

 同样，我们将为目标句子构建词汇表：

```python
target_vocab = build_vocab_from_iterator(
    getTokens(data_pipe,1),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
target_vocab.set_default_index(target_vocab['<unk>'])

```

 请注意，上面的示例显示了如何向词汇表中添加特殊标记。特殊标记可能会根据要求而更改。

 现在，我们可以验证特殊标记是否放置在开头，然后是其他单词。在下面的代码中， source_vocab.get_itos()
 返回一个包含标记的列表基于词汇的索引。

```python
print(source_vocab.get_itos()[:9])

```

输出：

```txt
['<pad>', '<sos>', '<eos>', '<unk>', '.', 'I', 'Tom', 'to', 'you']

```

## 使用词汇对句子进行数值化 [¶](#numericalize-sentences-using-vocabulary "永久链接到此标题")

 构建词汇表后，我们需要将句子转换为相应的索引。让我们为此定义一些函数：

```python
def getTransform(vocab):
 """
 Create transforms based on given vocabulary. The returned transform is applied to sequence
 of tokens.
 """
    text_tranform = T.Sequential(
        ## converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        ## Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
        # 1 as seen in previous section
        T.AddToken(1, begin=True),
        ## Add <eos> at beginning of each sentence. 2 because the index for <eos> in vocabulary is
        # 2 as seen in previous section
        T.AddToken(2, begin=False)
    )
    return text_tranform

```

 现在，让我们看看如何使用上面的函数。该函数返回一个 Transforms 的对象，我们将在我们的句子中使用它。让我们随机取一个句子并检查转换的工作原理。

```python
temp_list = list(data_pipe)
some_sentence = temp_list[798][0]
print("Some sentence=", end="")
print(some_sentence)
transformed_sentence = getTransform(source_vocab)(engTokenize(some_sentence))
print("Transformed sentence=", end="")
print(transformed_sentence)
index_to_string = source_vocab.get_itos()
for index in transformed_sentence:
    print(index_to_string[index], end=" ")

```

输出：

```txt
Some sentence=I giggled.
Transformed sentence=[1, 5, 4894, 4, 2]
<sos> I giggled . <eos>

```

 在上面的代码中：

* 在第 2 行，我们从第 1 行的 data_pipe 创建的列表中获取源句子
* 在第 5 行，我们获得基于源词汇表的转换并应用它到一个标记化的句子。请注意，转换采用单词列表而不是句子。
* 在第 8 行，我们获取索引到字符串的映射，然后使用它获取转换后的句子

 现在我们将使用 DataPipe 函数将转换应用于所有句子。让我们为此定义更多函数。

```python
def applyTransform(sequence_pair):
 """
 Apply transforms to sequence of tokens in a sequence pair
 """

    return (
        getTransform(source_vocab)(engTokenize(sequence_pair[0])),
        getTransform(target_vocab)(deTokenize(sequence_pair[1]))
    )
data_pipe = data_pipe.map(applyTransform) ## Apply the function to each element in the iterator
temp_list = list(data_pipe)
print(temp_list[0])

```

输出：

```txt
([1, 618, 4, 2], [1, 750, 4, 2])

```

## 制作批次（使用存储桶批次） [¶](#make-batches-with-bucket-batch "永久链接到此标题")

 一般情况下，我们都是批量训练模型。在使用序列到序列模型时，
建议保持批次中序列的长度相似。为此，我们将使用 bucketbatch 函数 data_pipe 。 让我们定义一些将由  bucketbatch 函数使用的函数。

```python
def sortBucket(bucket):
 """
 Function to sort a given bucket. Here, we want to sort based on the length of
 source and target sequence.
 """
    return sorted(bucket, key=lambda x: (len(x[0]), len(x[1])))

```

 现在，我们将应用 bucketbatch 函数：

```python
data_pipe = data_pipe.bucketbatch(
    batch_size = 4, batch_num=5,  bucket_num=1,
    use_in_batch_shuffle=False, sort_key=sortBucket
)

```

 在上面的代码块中：

* 我们保持批次大小 = 4。
* batch_num是要保存在存储桶中的批次数量
* Bucket_num\ n> 是池中保留的用于洗牌的存储桶数量
* sort_key 指定获取存储桶并对其进行排序的函数

 现在，让我们将一批源句子视为​​ X 并将一批目标句子视为 y 。通常，在训练模型时，我们会预测批 X 并将结果与 y 进行比较。但是，我们 data_pipe 中的一批的形式为 _[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4) ]_：

```python
print(list(data_pipe)[0])

```

输出：

```txt
[([1, 11016, 17, 4, 2], [1, 505, 29, 24, 2]), ([1, 11016, 17, 4, 2], [1, 7929, 1481, 24, 2]), ([1, 5279, 21, 4, 2], [1, 7307, 32, 24, 2]), ([1, 5279, 21, 4, 2], [1, 15846, 32, 24, 2])]

```

 因此，我们现在将它们转换为以下形式： ((X_1,X_2,X_3,X_4), (y_1,y\ \_2,y_3,y_4))。为此我们将编写一个小函数：

```python
def separateSourceTarget(sequence_pairs):
 """
 input of form: `[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]`
 output of form: `((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))`
 """
    sources,targets = zip(*sequence_pairs)
    return sources,targets

## Apply the function to each element in the iterator
data_pipe = data_pipe.map(separateSourceTarget)
print(list(data_pipe)[0])

```

输出：

```txt
(([1, 6815, 23, 10, 2], [1, 6815, 23, 10, 2], [1, 29, 472, 4, 2], [1, 29, 472, 4, 2]), ([1, 20624, 8, 2], [1, 11009, 8, 2], [1, 31, 1140, 4, 2], [1, 31, 1053, 4, 2]))

```

 现在，我们已经获得了所需的数据。

## 填充 [¶](#padding "此标题的永久链接")

 正如前面在构建词汇时所讨论的，我们需要在一批中填充较短的句子，
以使一批中的所有序列长度相等。我们可以按如下方式执行填充：

```python
def applyPadding(pair_of_sequences):
 """
 Convert sequences to tensors and apply padding
 """
    return (T.ToTensor(0)(list(pair_of_sequences[0])), T.ToTensor(0)(list(pair_of_sequences[1])))
## `T.ToTensor(0)` returns a transform that converts the sequence to `torch.tensor` and also applies
# padding. Here, `0` is passed to the constructor to specify the index of the `<pad>` token in the
# vocabulary.
data_pipe = data_pipe.map(applyPadding)

```

 现在，我们可以使用索引到字符串的映射来查看使用标记\而不是索引时序列的外观：

```python
source_index_to_string = source_vocab.get_itos()
target_index_to_string = target_vocab.get_itos()

def showSomeTransformedSentences(data_pipe):
 """
 Function to show how the sentences look like after applying all transforms.
 Here we try to print actual words instead of corresponding index
 """
    for sources,targets in data_pipe:
        if sources[0][-1] != 0:
            continue # Just to visualize padding of shorter sentences
        for i in range(4):
            source = ""
            for token in sources[i]:
                source += " " + source_index_to_string[token]
            target = ""
            for token in targets[i]:
                target += " " + target_index_to_string[token]
            print(f"Source: {source}")
            print(f"Traget: {target}")
        break

showSomeTransformedSentences(data_pipe)

```

输出：

```txt
Source:  <sos> Freeze ! <eos> <pad>
Traget:  <sos> Stehenbleiben ! <eos> <pad>
Source:  <sos> <unk> ! <eos> <pad>
Traget:  <sos> Zum Wohl ! <eos>
Source:  <sos> Freeze ! <eos> <pad>
Traget:  <sos> Keine Bewegung ! <eos>
Source:  <sos> Got it ! <eos>
Traget:  <sos> Verstanden ! <eos> <pad>

```

 在上面的输出中，我们可以观察到较短的句子用 `<pad>` 填充。现在，我们可以在编写训练函数时使用 data_pipe。

 本教程的某些部分的灵感来自 [本文](https://medium.com/@bitdribble/migrate-torchtext-to-the-new-0-9-0-api-1ff1472b5d71).

**脚本总运行时间：**
（4 分 27.354 秒）
