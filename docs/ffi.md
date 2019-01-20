# torch.utils.ffi

```py
torch.utils.ffi.create_extension(name, headers, sources, verbose=True, with_cuda=False, package=False, relative_to='.', **kwargs)
```

创建并配置一个 cffi.FFI 对象, 用于构建 PyTorch 的扩展.

| Parameters: | 

*   **name** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.6)")) – 包名. 可以是嵌套模块, 例如. `.ext.my_lib`.
*   **headers** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.6)") _or_ _List__[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.6)")_]_) – 只包含导出函数的头文件列表.
*   **sources** (_List__[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.6)")_]_) – 用于编译的sources列表.
*   **verbose** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.6)")_,_ _optional_) – 如果设置为 `False`, 则不会打印输出 (默认值: True).
*   **with_cuda** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.6)")_,_ _optional_) – 设置为 `True` 以使用 CUDA 头文件进行编译 (默认值: False)
*   **package** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.6)")_,_ _optional_) – 设置为 `True` 以在包模式下构建 (对于要作为 pip 程序包安装的模块) (默认值: False).
*   **relative_to** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.6)")_,_ _optional_) – 构建文件的路径. 当 `package 为 True` 时需要. 最好使用 `__file__` 作为参数.
*   **kwargs** – 传递给 ffi 以声明扩展的附件参数. 参考 [Extension API reference](https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension) 查阅更详细内容.

 |
| --- | --- |