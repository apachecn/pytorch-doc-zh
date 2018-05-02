from collections import namedtuple
import torch
from torch.autograd import Variable


PackedSequence_ = namedtuple('PackedSequence', ['data', 'batch_sizes'])


class PackedSequence(PackedSequence_):
    r"""保存一个打包序列的 data 和 batch_sizes.

    所有的 RNN 模块都接收这种被包裹后的序列作为它们的输入.

    Note:
        永远不要手动创建这个类的实例. 它们应当被 :func:`pack_padded_sequence` 这样的函数实例化.

    Attributes:
        data (Variable): 包含打包后序列的 Variable
        batch_sizes (list[int]): 包含每个序列步的 batch size 的列表
    """
    pass


def pack_padded_sequence(input, lengths, batch_first=False):
    r"""将填充过的变长序列打包(压紧).

    输入的形状可以是 ``TxBx*`` . T是最长序列长度(等于 ``lengths[0]``), B 是 batch size, *代表任意维度(可以是0). 如果 ``batch_first=True`` , 那么相应的 input size 就是 ``BxTx*`` .

    Variable 中保存的序列, 应该按序列长度的长短排序, 长的在前, 短的在后. 即 input[:,0] 代表的是最长的序列, input[:, B-1] 保存的是最短的序列. 

    Note:
        只要是维度大于等于2的 input 都可以作为这个函数的参数. 你可以用它来打包 labels, 然后用 RNN 的输出和打包后的 labels 来计算 loss. 通过 :class:`PackedSequence` 对象的 ``.data`` 属性可以获取 Variable.

    Arguments:
        input (Variable): 变长序列被填充后的 batch
        lengths (list[int]): Variable 中每个序列的长度.
        batch_first (bool, optional): 如果是 ``True``, input 的形状应该是 BxTx*.

    Returns:
        一个 :class:`PackedSequence` 对象.
    """
    if lengths[-1] <= 0:
        raise ValueError("length of all samples has to be greater than 0, "
                         "but found an element in 'lengths' that is <=0")
    if batch_first:
        input = input.transpose(0, 1)

    steps = []
    batch_sizes = []
    lengths_iter = reversed(lengths)
    batch_size = input.size(1)
    if len(lengths) != batch_size:
        raise ValueError("lengths array has incorrect size")

    prev_l = 0
    for i, l in enumerate(lengths_iter):
        if l > prev_l:
            c_batch_size = batch_size - i
            steps.append(input[prev_l:l, :c_batch_size].contiguous().view(-1, *input.size()[2:]))
            batch_sizes.extend([c_batch_size] * (l - prev_l))
            prev_l = l
        elif prev_l > l:  # remember that new_length is the preceding length in the array
            raise ValueError("lengths array has to be sorted in decreasing order")

    return PackedSequence(torch.cat(steps), batch_sizes)


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0):
    r"""填充打包过的变长序列.

    这是 :func:`pack_padded_sequence` 的逆操作.

    返回的 Varaible 的值的 size 是 TxBx*, T 是最长序列的长度, B 是 batch_size, 如果 ``batch_first=True``, 那么返回值是 BxTx*.

    Batch中的元素将会以它们长度的逆序排列.

    Arguments:
        sequence (PackedSequence): 将要被填充的 batch
        batch_first (bool, optional):  如果为 ``True`` , 返回的数据的格式为 BxTx*.
        padding_value (float, optional): 用来填充元素的值

    Returns:
        一个 tuple, 包含被填充后的序列, 和 batch 中序列的长度列表.
    """
    var_data, batch_sizes = sequence
    max_batch_size = batch_sizes[0]
    output = var_data.data.new(len(batch_sizes), max_batch_size, *var_data.size()[1:]).fill_(padding_value)
    output = Variable(output)

    lengths = []
    data_offset = 0
    prev_batch_size = batch_sizes[0]
    prev_i = 0
    for i, batch_size in enumerate(batch_sizes):
        if batch_size != prev_batch_size:
            l = prev_batch_size * (i - prev_i)
            output[prev_i:i, :prev_batch_size] = var_data[data_offset:data_offset + l]
            data_offset += l
            prev_i = i
        dec = prev_batch_size - batch_size
        if dec > 0:
            lengths.extend((i,) * dec)
        prev_batch_size = batch_size

    l = prev_batch_size * (len(batch_sizes) - prev_i)
    output[prev_i:, :prev_batch_size] = var_data[data_offset:data_offset + l]

    lengths.extend((i + 1,) * batch_size)
    lengths.reverse()

    if batch_first:
        output = output.transpose(0, 1)
    return output, lengths
