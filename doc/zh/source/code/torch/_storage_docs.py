"""Adds docstrings to Storage functions"""

import torch._C
from torch._C import _add_docstr as add_docstr


storage_classes = [
    'DoubleStorageBase',
    'FloatStorageBase',
    'LongStorageBase',
    'IntStorageBase',
    'ShortStorageBase',
    'CharStorageBase',
    'ByteStorageBase',
]


def add_docstr_all(method, docstr):
    for cls_name in storage_classes:
        cls = getattr(torch._C, cls_name)
        try:
            add_docstr(getattr(cls, method), docstr)
        except AttributeError:
            pass


add_docstr_all('from_file',
               """
from_file(filename, shared=False, size=0) -> Storage

如果 shared 为 True , 那么内存将会在所有进程间共享 . 所有的更改都会被写入文件 . 如果 shared 为 False , 
那么对于内存的修改 , 则不会影响到文件 . 

size 是存储中所包含的元素个数 . 如果 shared 为 False 则文件必须包含至少 `size * sizeof(Type)` 字节
( `Type` 是所存储的类型) . 如果 shared 为 True , 文件会在需要的时候被创建 . 

Args:
    filename (str): 要映射到的文件名
    shared (bool): 是否共享内存
    size (int): 存储中包含元素的个数
""")
