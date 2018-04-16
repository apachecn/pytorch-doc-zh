import math
import torch
import warnings

from .module import Module
from ..parameter import Parameter
from ..utils.rnn import PackedSequence


class RNNBase(Module):

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            self._data_ptrs = []
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for l in self.all_weights for p in l)
        if len(unique_data_ptrs) != sum(len(l) for l in self.all_weights):
            self._data_ptrs = []
            return

        with torch.cuda.device_of(any_param):
            # This is quite ugly, but it allows us to reuse the cuDNN code without larger
            # modifications. It's really a low-level API that doesn't belong in here, but
            # let's make this exception.
            from torch.backends.cudnn import rnn
            from torch.backends import cudnn
            from torch.nn._functions.rnn import CudnnRNN
            handle = cudnn.get_handle()
            with warnings.catch_warnings(record=True):
                fn = CudnnRNN(
                    self.mode,
                    self.input_size,
                    self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=self.batch_first,
                    dropout=self.dropout,
                    train=self.training,
                    bidirectional=self.bidirectional,
                    dropout_state=self.dropout_state,
                )

            # Initialize descriptors
            fn.datatype = cudnn._typemap[any_param.type()]
            fn.x_descs = cudnn.descriptor(any_param.new(1, self.input_size), 1)
            fn.rnn_desc = rnn.init_rnn_descriptor(fn, handle)

            # Allocate buffer to hold the weights
            self._param_buf_size = rnn.get_num_weights(handle, fn.rnn_desc, fn.x_descs[0], fn.datatype)
            fn.weight_buf = any_param.new(self._param_buf_size).zero_()
            fn.w_desc = rnn.init_weight_descriptor(fn, fn.weight_buf)

            # Slice off views into weight_buf
            all_weights = [[p.data for p in l] for l in self.all_weights]
            params = rnn.get_parameters(fn, handle, fn.weight_buf)

            # Copy weights and update their storage
            rnn._copyParams(all_weights, params)
            for orig_layer_param, new_layer_param in zip(all_weights, params):
                for orig_param, new_param in zip(orig_layer_param, new_layer_param):
                    orig_param.set_(new_param.view_as(orig_param))

            self._data_ptrs = list(p.data.data_ptr() for p in self.parameters())

    def _apply(self, fn):
        ret = super(RNNBase, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                        num_directions,
                                                        max_batch_size,
                                                        self.hidden_size).zero_(), requires_grad=False)
            if self.mode == 'LSTM':
                hx = (hx, hx)

        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None
        func = self._backend.RNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            batch_sizes=batch_sizes,
            dropout_state=self.dropout_state,
            flat_weight=flat_weight
        )
        output, hidden = func(input, self.all_weights, hx)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def __setstate__(self, d):
        super(RNNBase, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class RNN(RNNBase):
    r"""对于输入序列使用一个多层的 ``Elman RNN``, 它的激活函数为 ``tanh`` 或者 ``ReLU`` .

    对输入序列中每个元素, 每层计算公式为:


    .. math::

        h_t = \tanh(w_{ih} * x_t + b_{ih}  +  w_{hh} * h_{(t-1)} + b_{hh})

    这里 :math:`h_t` 是当前在时刻 `t` 的隐状态, 并且 :math:`x_t` 是之前一层在 `t` 时刻的隐状态, 或者是第一层的输入.
    如果 ``nonlinearity='relu'`` ,那么将使用 relu 代替 tanh 作为激活函数.

    Args:
        input_size: 输入 x 的特征数量
        hidden_size:  隐状态 ``h`` 中的特征数量
        num_layers: RNN 的层数
        nonlinearity: 指定非线性函数使用 ['tanh'|'relu']. 默认: 'tanh'
        bias:  如果是 ``False`` , 那么 RNN 层就不会使用偏置权重 b_ih 和 b_hh, 默认: ``True``
        batch_first: 如果 ``True``, 那么输入 ``Tensor`` 的 shape 应该是 (batch, seq, feature),并且输出也是一样
        dropout:  如果值非零, 那么除了最后一层外, 其它层的输出都会套上一个 ``dropout`` 层
        bidirectional:  如果 ``True`` , 将会变成一个双向 RNN, 默认为 ``False``

    Inputs: input, h_0
        - **input** (seq_len, batch, input_size): 包含输入序列特征的 ``tensor`` ,
          ``input`` 可以是被填充的变长序列.细节请看 :func:`torch.nn.utils.rnn.pack_padded_sequence` .
        - **h_0** (num_layers * num_directions, batch, hidden_size): 包含 ``batch`` 中每个元素保存着初始隐状态的 ``tensor``

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): 包含 RNN 最后一层输出特征 (h_k) 的 ``tensor``
          对于每个 k ,如果输入是一个 :class:`torch.nn.utils.rnn.PackedSequence` , 那么输出也是一个可以是被填充的变长序列.
        - **h_n** (num_layers * num_directions, batch, hidden_size): 包含 k= seq_len 隐状态的 ``tensor``.

    Attributes:
        weight_ih_l[k]: 第 k 层的 input-hidden 权重,可学习, shape 是 `(input_size x hidden_size)`
        weight_hh_l[k]: 第 k 层的 hidden-hidden 权重, 可学习, shape 是 `(hidden_size x hidden_size)`
        bias_ih_l[k]: 第 k 层的 input-hidden 偏置, 可学习, shape 是 `(hidden_size)`
        bias_hh_l[k]: 第 k 层的 hidden-hidden 偏置, 可学习, shape 是 `(hidden_size)`

    Examples::

        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = Variable(torch.randn(5, 3, 10))
        >>> h0 = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError("Unknown nonlinearity '{}'".format(
                    kwargs['nonlinearity']))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(RNN, self).__init__(mode, *args, **kwargs)


class LSTM(RNNBase):
    r"""对于输入序列使用一个多层的 ``LSTM`` ( long short-term memory ).


    对输入序列的每个元素, ``LSTM`` 的每层都会执行以下计算:

    .. math::

            \begin{array}{ll}
            i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
            f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
            o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\
            h_t = o_t * \tanh(c_t)
            \end{array}

    这里 :math:`h_t` 是在时刻 `t` 的隐状态, :math:`c_t` 是在时刻 `t` 的细胞状态 (cell state),
    :math:`x_t` 是上一层的在时刻 `t` 的隐状态或者是第一层的 :math:`input_t` , 而 :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` 分别代表 输入门,遗忘门,细胞和输出门.

    Args:
        input_size: 输入的特征维度
        hidden_size: 隐状态的特征维度
        num_layers: 层数(和时序展开要区分开)
        bias: 如果为 ``False`` ,那么 LSTM 将不会使用 b_ih 和 b_hh ,默认: ``True``
        batch_first: 如果为 ``True`` , 那么输入和输出 Tensor 的形状为 (batch, seq, feature)
        dropout: 如果非零的话, 将会在 RNN 的输出上加个 dropout , 最后一层除外
        bidirectional: 如果为 ``True``,将会变成一个双向 RNN ,默认为 ``False``

    Inputs: input, (h_0, c_0)
        - **input** (seq_len, batch, input_size): 包含输入序列特征的 ``tensor`` .
          也可以是 ``packed variable length sequence``,
          详见 :func:`torch.nn.utils.rnn.pack_padded_sequence` .
        - **h_0** (num_layers \* num_directions, batch, hidden_size): 包含 batch 中每个元素的初始化隐状态的 ``tensor`` .
        - **c_0** (num_layers \* num_directions, batch, hidden_size): 包含 batch 中每个元素的初始化细胞状态的 ``tensor`` .


    Outputs: output, (h_n, c_n)
        - **output** (seq_len, batch, hidden_size * num_directions): 包含 RNN 最后一层的输出特征 `(h_t)` 的 ``tensor`` ,
          对于每个 t . 如果输入是  :class:`torch.nn.utils.rnn.PackedSequence`
          那么输出也是一个可以是被填充的变长序列.
        - **h_n** (num_layers * num_directions, batch, hidden_size): 包含 t=seq_len 隐状态的 ``tensor``.
        - **c_n** (num_layers * num_directions, batch, hidden_size): 包含 t=seq_len 细胞状态的 ``tensor``.

    Attributes:
        weight_ih_l[k] : 第 k 层可学习的 input-hidden 权重
            `(W_ii|W_if|W_ig|W_io)`, shape 是 `(4*hidden_size x input_size)`
        weight_hh_l[k] : 第 k 层可学习的 hidden-hidden 权重
            `(W_hi|W_hf|W_hg|W_ho)`, shape 是 `(4*hidden_size x hidden_size)`
        bias_ih_l[k] :  第 k 层可学习的 input-hidden 偏置
            `(b_ii|b_if|b_ig|b_io)`, shape 是 `(4*hidden_size)`
        bias_hh_l[k] : 第 k 层可学习的 hidden-hidden 偏置
            `(b_hi|b_hf|b_hg|b_ho)`, shape 是 `(4*hidden_size)`

    Examples::

        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = Variable(torch.randn(5, 3, 10))
        >>> h0 = Variable(torch.randn(2, 3, 20))
        >>> c0 = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, (h0, c0))
    """

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)


class GRU(RNNBase):
    r"""对于输入序列使用一个多层的 ``GRU`` (gated recurrent unit).


    对输入序列的每个元素, 每层都会执行以下计算:

    .. math::

            \begin{array}{ll}
            r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\
            \end{array}

    这里 :math:`h_t` 是在时刻 `t` 的隐状态, :math:`x_t` 是前一层在时刻 `t` 的隐状态或者是第一层的 :math:`input_t` ,
    而 :math:`r_t`, :math:`z_t`, :math:`n_t` 分别是重置门,输入门和新门.

    Args:
        input_size: 输入的特征维度
        hidden_size: 隐状态的特征维度
        num_layers: RNN 的层数
        bias: 如果为 ``False``, 那么 RNN 层将不会使用偏置权重 b_ih 和 b_hh
            默认: ``True``
        batch_first: 如果为 ``True``, 那么输入和输出的 ``tensor`` 的形状是 (batch, seq, feature)
        dropout:  如果非零的话,将会在 RNN 的输出上加个 dropout ,最后一层除外
        bidirectional: 如果为 ``True``, 将会变成一个双向 RNN . 默认: ``False``

    Inputs: input, h_0
        - **input** (seq_len, batch, input_size): 包含输入序列特征的 ``tensor`` .
          也可以是 ``packed variable length sequence``,
          详见 :func:`torch.nn.utils.rnn.pack_padded_sequence` .
        - **h_0** (num_layers * num_directions, batch, hidden_size): 包含 batch 中每个元素的初始化隐状态的 ``tensor``

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): 包含 RNN 最后一层的输出特征 `(h_t)` 的 ``tensor`` ,
          对于每个 t . 如果输入是 :class:`torch.nn.utils.rnn.PackedSequence`
          那么输出也是一个可以是被填充的变长序列.
        - **h_n** (num_layers * num_directions, batch, hidden_size): 包含 t=seq_len 隐状态的 ``tensor``.

    Attributes:
        weight_ih_l[k] : 第 k 层可学习的 input-hidden 权重
            (W_ir|W_iz|W_in), shape 为 `(3*hidden_size x input_size)`
        weight_hh_l[k] : 第 k 层可学习的 hidden-hidden 权重
            (W_hr|W_hz|W_hn), shape 为 `(3*hidden_size x hidden_size)`
        bias_ih_l[k] : 第 k 层可学习的 input-hidden 偏置
            (b_ir|b_iz|b_in), shape 为 `(3*hidden_size)`
        bias_hh_l[k] : 第 k 层可学习的 hidden-hidden 偏置
            (b_hr|b_hz|b_hn), shape 为 `(3*hidden_size)`
    Examples::

        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = Variable(torch.randn(5, 3, 10))
        >>> h0 = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__('GRU', *args, **kwargs)


class RNNCellBase(Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class RNNCell(RNNCellBase):
    r"""一个 ``Elan RNN cell`` , 激活函数是 tanh 或 ReLU , 用于输入序列.

    .. math::

        h' = \tanh(w_{ih} * x + b_{ih}  +  w_{hh} * h + b_{hh})

    如果 nonlinearity='relu', 那么将会使用 ReLU 来代替 tanh .

    Args:
        input_size: 输入的特征维度
        hidden_size: 隐状态的特征维度
        bias: 如果为 ``False``, 那么RNN层将不会使用偏置权重 b_ih 和 b_hh.
            默认: ``True``
        nonlinearity: 用于选择非线性激活函数 ['tanh'|'relu']. 默认: 'tanh'

    Inputs: input, hidden
        - **input** (batch, input_size): 包含输入特征的 ``tensor`` .
        - **hidden** (batch, hidden_size): 包含 batch 中每个元素的初始化隐状态的 ``tensor``.

    Outputs: h'
        - **h'** (batch, hidden_size): 保存着 batch 中每个元素的下一层隐状态的 ``tensor`` .

    Attributes:
        weight_ih: ``input-hidden`` 权重, 可学习, shape 为 `(input_size x hidden_size)`
        weight_hh: ``hidden-hidden`` 权重, 可学习, shape 为  `(hidden_size x hidden_size)`
        bias_ih: ``input-hidden`` 偏置,可学习, shape 为 `(hidden_size)`
        bias_hh: ``hidden-hidden`` 偏置,可学习, shape 为 `(hidden_size)`

    Examples::

        >>> rnn = nn.RNNCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        if self.nonlinearity == "tanh":
            func = self._backend.RNNTanhCell
        elif self.nonlinearity == "relu":
            func = self._backend.RNNReLUCell
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        return func(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


class LSTMCell(RNNCellBase):
    r"""LSTM 细胞.

    .. math::

        \begin{array}{ll}
        i = \mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}

    Args:
        input_size: 输入的特征维度
        hidden_size: 隐状态的维度
        bias: 如果为 `False`, 那么RNN层将不会使用偏置权重 b_ih 和 b_hh
             默认: ``True``

    Inputs: input, (h_0, c_0)
        - **input** (batch, input_size): 包含输入特征的 ``tensor`` .
        - **h_0** (batch, hidden_size): 包含 batch 中每个元素的初始化隐状态的 ``tensor``.
        - **c_0** (batch. hidden_size): 包含 batch 中每个元素的初始化细胞状态的 ``tensor``

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): 保存着 batch 中每个元素的下一层隐状态的 ``tensor``
        - **c_1** (batch, hidden_size): 保存着 batch 中每个元素的下一细胞状态的 ``tensor``

    Attributes:
        weight_ih: ``input-hidden`` 权重, 可学习, 形状为 `(4*hidden_size x input_size)`
        weight_hh: ``hidden-hidden`` 权重, 可学习, 形状为 `(4*hidden_size x hidden_size)`
        bias_ih: ``input-hidden`` 偏置, 可学习, 形状为 `(4*hidden_size)`
        bias_hh: ``hidden-hidden`` 偏置, 可学习, 形状为 `(4*hidden_size)`

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> cx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        return self._backend.LSTMCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


class GRUCell(RNNCellBase):
    r""" GRU 细胞

    .. math::

        \begin{array}{ll}
        r = \mathrm{sigmoid}(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \mathrm{sigmoid}(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    Args:
        input_size: 输入的特征维度
        hidden_size: 隐状态的维度
        bias: 如果为 `False`, 那么RNN层将不会使用偏置权重 b_ih 和 b_hh
             默认: ``True``

    Inputs: input, hidden
        - **input** (batch, input_size): 包含输入特征的 ``tensor`` .
        - **hidden** (batch, hidden_size): 包含 batch 中每个元素的初始化隐状态的 ``tensor``.

    Outputs: h'
        - **h'**: (batch, hidden_size): 保存着 batch 中每个元素的下一层隐状态的 ``tensor``

    Attributes:
        weight_ih: ``input-hidden`` 权重, 可学习, shape 为, `(3*hidden_size x input_size)`
        weight_hh: ``hidden-hidden`` 权重, 可学习, shape 为 `(3*hidden_size x hidden_size)`
        bias_ih: ``input-hidden`` 偏置, 可学习, shape 为 `(3*hidden_size)`
        bias_hh: ``hidden-hidden`` 偏置, 可学习, shape 为 `(3*hidden_size)`

    Examples::

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        return self._backend.GRUCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
