

# torch.utils.cpp_extension
> 译者:  [belonHan](https://github.com/belonHan)

```py
torch.utils.cpp_extension.CppExtension(name, sources, *args, **kwargs)
```

创建一个C++的setuptools.Extension。

便捷地创建一个setuptools.Extension具有最小(但通常是足够）的参数来构建C++扩展的方法。

所有参数都被转发给setuptools.Extension构造函数。

例子

```py
>>> from setuptools import setup
>>> from torch.utils.cpp_extension import BuildExtension, CppExtension
>>> setup(
name='extension',
ext_modules=[
CppExtension(
name='extension',
sources=['extension.cpp'],
extra_compile_args=['-g'])),
],
cmdclass={
'build_ext': BuildExtension
})

```

```py
torch.utils.cpp_extension.CUDAExtension(name, sources, *args, **kwargs)
```

为CUDA/C++创建一个`setuptools.Extension`。

创建一个setuptools.Extension用于构建CUDA/C ++扩展的最少参数(但通常是足够的）的便捷方法。这里包括CUDA路径，库路径和运行库。 所有参数都被转发给setuptools.Extension构造函数。

所有参数都被转发给setuptools.Extension构造函数。

例子

```py
>>> from setuptools import setup
>>> from torch.utils.cpp_extension import BuildExtension, CUDAExtension
>>> setup(
name='cuda_extension',
ext_modules=[
CUDAExtension(
name='cuda_extension',
sources=['extension.cpp', 'extension_kernel.cu'],
extra_compile_args={'cxx': ['-g'],
'nvcc': ['-O2']})
],
cmdclass={
'build_ext': BuildExtension
})

```

```py
torch.utils.cpp_extension.BuildExtension(*args, **kwargs)
```

自定义setuptools构建扩展。

`setuptools.build_ext`子类负责传递所需的最小编译器参数(例如`-std=c++11`）以及混合的C ++/CUDA编译(以及一般对CUDA文件的支持）。

当使用[`BuildExtension`](#torch.utils.cpp_extension.BuildExtension "torch.utils.cpp_extension.BuildExtension")时，它将提供一个用于`extra_compile_args`(不是普通列表）的词典，通过语言(`cxx`或`cuda`）映射到参数列表提供给编译器。这样可以在混合编译期间为C ++和CUDA编译器提供不同的参数。

```py
torch.utils.cpp_extension.load(name, sources, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None, build_directory=None, verbose=False, with_cuda=None, is_python_module=True)
```

即时加载(JIT)PyTorch C ++扩展。

为了加载扩展，会创建一个Ninja构建文件，该文件用于将指定的源编译为动态库。随后将该库作为模块加载到当前Python进程中，并从该函数返回，以备使用。

默认情况下，构建文件创建的目录以及编译结果库是`&lt;tmp&gt;/torch_extensions/&lt;name&gt;`，其中`&lt;tmp&gt;`是当前平台上的临时文件夹以及`&lt;name&gt;`为扩展名。这个位置可以通过两种方式被覆盖。首先，如果`TORCH_EXTENSIONS_DIR`设置了环境变量，它将替换`&lt;tmp&gt;/torch_extensions`并将所有扩展编译到此目录的子文件夹中。其次，如果`build_directory`函数设置了参数，它也将覆盖整个路径，即,库将直接编译到该文件夹中。

要编译源文件，使用默认的系统编译器(`c++`），可以通过设置`CXX`环境变量来覆盖它。将其他参数传递给编译过程，`extra_cflags`或者`extra_ldflags`可以提供。例如，要通过优化来编译您的扩展，你可以传递`extra_cflags=['-O3']`，也可以使用 `extra_cflags`传递进一步包含目录。

提供了混合编译的CUDA支持。只需将CUDA源文件(.cu或.cuh）与其他源一起传递即可。这些文件将被检测，并且使用nvcc而不是C ++编译器进行编译。包括将CUDA lib64目录作为库目录传递并进行cudart链接。您可以将其他参数传递给nvcc extra_cuda_cflags，就像使用C ++的extra_cflags一样。使用了各种原始方法来查找CUDA安装目录，通常情况下可以正常运行。如果不可以，最好设置CUDA_HOME环境变量。


参数:
*   name - 要构建的扩展名。这个必须和`pybind11`模块的名字一样！
*   sources - `C++`源文件的相对或绝对路径列表。
*   extra_cflags - 编译器参数的可选列表，用于转发到构建。
*   extra_cuda_cflags - 编译器标记的可选列表，在构建`CUDA`源时转发给`nvcc`。
*   extra_ldflags - 链接器参数的可选列表，用于转发到构建。
*   extra_include_paths - 转发到构建的包含目录的可选列表。
*   build_directory - 可选路径作为构建区域。
*   verbose - 如果为`True`，打开加载步骤的详细记录。
*   with_cuda – 确定构建是是否包含CUDA头/库. 默认值 `None`, 自动通过`sources`目录是否存在 `.cu` 或 `.cuh`文件确定.  `True`强制包含.
*   is_python_module – 默认值 `True`: python模块方式导入. `False`: 普通动态库方式加载到程序.


| 返回: | `is_python_module` == `True`, 加载`PyTorch`扩展作为`Python`模块。If `is_python_module` == `False` 无返回 (副作用是共享库被加载到进程). |
| --- | --- |

例子

```py
>>> from torch.utils.cpp_extension import load
>>> module = load(
name='extension',
sources=['extension.cpp', 'extension_kernel.cu'],
extra_cflags=['-O2'],
verbose=True)

```

```py
torch.utils.cpp_extension.load_inline(name, cpp_sources, cuda_sources=None, functions=None, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None, build_directory=None, verbose=False, with_cuda=None, is_python_module=True)
```

在运行时编译加载PyTorch C++ 扩展

这个函数很像[`load()`](#torch.utils.cpp_extension.load "torch.utils.cpp_extension.load")，但是它的源文件是字符串而不是文件名。在把这些字符串保存到构建目录后，[`load_inline()`](#torch.utils.cpp_extension.load_inline "torch.utils.cpp_extension.load_inline") 等价于 [`load()`](#torch.utils.cpp_extension.load "torch.utils.cpp_extension.load").

例子： [the tests](https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions.py) 


源代码可能会省略非内联c++扩展的两个必要部分:必要的头文件,以及(pybind11)绑定代码。更准确地说，传递给`cpp_sources`的字符串首先连接成一个单独的`.cpp`文件。然后在这个文件前面加上`#include & lt;torch/extension.h&gt;`

此外，如果提供了`functions`的参数，指定的函数将自动生成绑定。`functions`可以是函数名列表，也可以是{函数名:文档字符串}的字典。如果给定了一个列表，则每个函数的名称用作其文档字符串。

`cuda_sources`中的代码按顺序连接到单独的`.cu`文件,追加`torch/types.h`, `cuda.h` and `cuda_runtime.h`头文件.`.cpp` 和 `.cu` 文件分开编译, 最终连接到一个库中. 注意`cuda_sources`中的函数本身没有绑定,为了绑定CUDA核函数,必须新建一个C++函数来调用它,或者在`cpp_sources` 中声明或定义(并且在`functions`中包含它).


[`load()`](#torch.utils.cpp_extension.load "torch.utils.cpp_extension.load")查看下面忽略的参数.

参数:

*   **cpp_sources** – 字符串, or 字符串列表, 包含C++源代码
*   **cuda_sources** – 字符串, or 字符串列表, 包含CUDA源代码
*   **functions** – 函数名列表 用于生成函数绑定. 如果是字典,key=函数名,value=文档描述.
*   **with_cuda** – 确定是否添加CUDA头/库. 默认值 `None` (default), 取决于参数`cuda_sources` . `True`强制包含CUDA头/库.



例子

```py
>>> from torch.utils.cpp_extension import load_inline
>>> source = '''
at::Tensor sin_add(at::Tensor x, at::Tensor y) {
return x.sin() + y.sin();
}
'''
>>> module = load_inline(name='inline_extension',
cpp_sources=[source],
functions=['sin_add'])

```

```py
torch.utils.cpp_extension.include_paths(cuda=False)
```

获取构建`C++`或`CUDA`扩展所需的路径。


*   参数： `cuda` - 如果为True，则包含`CUDA`特定的包含路径。
*   返回： 包含路径字符串的列表。

```py
torch.utils.cpp_extension.check_compiler_abi_compatibility(compiler)
```

验证给定的编译器是否与`PyTorch` ABI兼容。


*   参数：compiler([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) - 要检查可执行的编译器文件名(例如`g++`),必须在`shell`进程中可执行。
*   返回：如果编译器(可能）与`PyTorch`ABI不兼容，则为`False`，否则返回`True`。

```py
torch.utils.cpp_extension.verify_ninja_availability()
```

如果可以在[ninja](https://ninja-build.org/)上运行则返回`True`。

