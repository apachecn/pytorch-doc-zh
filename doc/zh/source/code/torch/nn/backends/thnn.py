from .backend import FunctionBackend


class THNNFunctionBackend(FunctionBackend):

    def __reduce__(self):
        return (_get_thnn_function_backend, ())

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __copy__(self):
        return self


def _get_thnn_function_backend():
    return backend


def _initialize_backend():
    from .._functions.thnn import _all_functions as _thnn_functions
    from .._functions.rnn import RNN, \
        RNNTanhCell, RNNReLUCell, GRUCell, LSTMCell
    from .._functions.dropout import Dropout, FeatureDropout
    from .._functions.loss import CosineEmbeddingLoss, \
        HingeEmbeddingLoss, HingeEmbeddingLossBackward, MarginRankingLoss

    backend.register_function('RNN', RNN)
    backend.register_function('RNNTanhCell', RNNTanhCell)
    backend.register_function('RNNReLUCell', RNNReLUCell)
    backend.register_function('LSTMCell', LSTMCell)
    backend.register_function('GRUCell', GRUCell)
    backend.register_function('Dropout', Dropout)
    backend.register_function('Dropout2d', FeatureDropout)
    backend.register_function('Dropout3d', FeatureDropout)
    backend.register_function('CosineEmbeddingLoss', CosineEmbeddingLoss)
    backend.register_function('HingeEmbeddingLoss', HingeEmbeddingLoss)
    backend.register_function('HingeEmbeddingLossBackward', HingeEmbeddingLossBackward)
    backend.register_function('MarginRankingLoss', MarginRankingLoss)
    for cls in _thnn_functions:
        name = cls.__name__
        backend.register_function(name, cls)


backend = THNNFunctionBackend()
_initialize_backend()
