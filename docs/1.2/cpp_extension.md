# torch.utils.cpp_extension

`torch.utils.cpp_extension.``CppExtension`( _name_ , _sources_ , _*args_ ,
_**kwargs_ )[[source]](_modules/torch/utils/cpp_extension.html#CppExtension)

    

创建`setuptools.Extension`为C ++。

一个创建`setuptools.Extension`与最低限度(但通常​​足以）参数建立一个C ++扩展的便捷方法。

所有参数被转发到`setuptools.Extension`构造。

例

    
    
    >>> from setuptools import setup
    >>> from torch.utils.cpp_extension import BuildExtension, CppExtension
    >>> setup(
            name='extension',
            ext_modules=[
                CppExtension(
                    name='extension',
                    sources=['extension.cpp'],
                    extra_compile_args=['-g']),
            ],
            cmdclass={
                'build_ext': BuildExtension
            })
    

`torch.utils.cpp_extension.``CUDAExtension`( _name_ , _sources_ , _*args_ ,
_**kwargs_ )[[source]](_modules/torch/utils/cpp_extension.html#CUDAExtension)

    

创建`setuptools.Extension  [HTG3用于CUDA / C ++。`

一个创建`setuptools.Extension`与最低限度(但通常​​足以）参数建立一个CUDA / C
++扩展的便捷方法。这包括CUDA包括路径，库路径和运行时库。

All arguments are forwarded to the `setuptools.Extension`constructor.

Example

    
    
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
    

`torch.utils.cpp_extension.``BuildExtension`( _*args_ , _**kwargs_
)[[source]](_modules/torch/utils/cpp_extension.html#BuildExtension)

    

自定义`setuptools的 `构建扩展。

此`setuptools.build_ext`亚类需要经过所需要的最小编译器标志的护理(例如，`-std = C ++ 11`）以及作为混合的C
++ / CUDA汇编(和一般为CUDA文件的支持）。

当使用 `BuildExtension`，它被允许提供一个字典`extra_compile_args`(而不是通常的列表）从语言映射(`
CXX`或`CUDA`）来的额外的编译标志来提供给编译器的列表。这使得可以为混合编译期间提供不同的标记到C ++和CUDA编译器。

`torch.utils.cpp_extension.``load`( _name_ , _sources_ , _extra_cflags=None_ ,
_extra_cuda_cflags=None_ , _extra_ldflags=None_ , _extra_include_paths=None_ ,
_build_directory=None_ , _verbose=False_ , _with_cuda=None_ ,
_is_python_module=True_
)[[source]](_modules/torch/utils/cpp_extension.html#load)

    

加载PyTorch C ++扩展刚刚在时间(JIT）。

要加载的扩展，一个忍者构建文件被发射时，其被用来编译该给定源集成到一个动态库。该库随后被加载到当前的Python程序作为一个模块，并从该函数返回，以备使用。

默认情况下，目录到构建文件发出并编译得到的库来为`& LT ; TMP & GT ; / torch_extensions / [ - - ] LT
;名称& GT ;`，其中`& LT ; TMP & GT ;`为当前平台上的临时文件夹并`& LT ;名称& GT ;
`扩展的名称。这个位置可以通过两种方式来覆盖。首先，如果`TORCH_EXTENSIONS_DIR`环境变量被设置，它取代`& LT ; TMP &
GT ; / torch_extensions`和所有的扩展会被编译成这个目录的子文件夹。其次，如果被提供的`BUILD_DIRECTORY
`参数给此函数，它会覆盖整个路径，即，库将被直接编译到该文件夹​​。

编译源代码，默认的系统编译器(`C ++`）被使用，其可以通过设置`CXX`环境变量被重写。传递额外的参数来编译过程，`
EXTRA_CFLAGS`或``可以提供EXTRA_LDFLAGS。例如，为了与编译优化您的扩展，通过`EXTRA_CFLAGS = [ ' -
O3']`。您也可以使用`EXTRA_CFLAGS`进一步通过包括目录。

提供CUDA支持混合编译。简单地传递CUDA源文件(`.CU`或`.cuh`）与其他来源的沿。这些文件将被检测并与NVCC，而不是C
++编译器编译。这包括使CUDA lib64目录作为一个库的目录，和链接`cudart`。可以传递附加标志通过`至NVCC
extra_cuda_cflags`，就像`EXTRA_CFLAGS`为C
++。为寻找CUDA安装目录各种试探被使用，通常做工精细。如果不是，设置`CUDA_HOME`环境变量是最安全的选择。

Parameters

    

  * **名** \- 扩展的名称来构建。这必须是相同pybind11模块的名称！

  * **来源** \- 相对或绝对路径到C ++源文件的列表。

  * **EXTRA_CFLAGS** \- 编译器标志的可选列表转发到构建。

  * **extra_cuda_cflags** \- 编译器标志的可选列表了建设CUDA源时，NVCC。

  * **EXTRA_LDFLAGS** \- 连接标志的可选列表转发到构建。

  * **extra_include_paths** \- 包括目录的可选列表转发到构建。

  * **BUILD_DIRECTORY** \- 可选路径为构建工作空间使用。

  * **冗长** \- 若`真 `，接通的负载的步骤详细日志记录。

  * **with_cuda** \- 确定CUDA头和库是否被添加到该生成。如果设置为`无 `(默认），该值被自动确定基于的`.CU`或`[HTG11存在] .cuh `在`来源 `。其设置为 TRUE`给力CUDA头文件和库包括在内。

  * **is_python_module** \- 若`真 `(默认），出口所产生的共享库的Python模块。如果`假 `，将其加载到处理作为一个纯动态库。

Returns

    

如果`is_python_module`是`真 `，返回加载PyTorch扩展作为一个Python模块。如果`is_python_module
`是`假 `返回任何(共享库加载到过程作为副作用）。

Example

    
    
    >>> from torch.utils.cpp_extension import load
    >>> module = load(
            name='extension',
            sources=['extension.cpp', 'extension_kernel.cu'],
            extra_cflags=['-O2'],
            verbose=True)
    

`torch.utils.cpp_extension.``load_inline`( _name_ , _cpp_sources_ ,
_cuda_sources=None_ , _functions=None_ , _extra_cflags=None_ ,
_extra_cuda_cflags=None_ , _extra_ldflags=None_ , _extra_include_paths=None_ ,
_build_directory=None_ , _verbose=False_ , _with_cuda=None_ ,
_is_python_module=True_
)[[source]](_modules/torch/utils/cpp_extension.html#load_inline)

    

装载来自串源的PyTorch C ++扩展刚刚在时间(JIT）。

此函数的行为完全一样 `负载(） `，但需要它的来源字符串而不是文件名。这些字符串存储到文件中生成目录，之后， `load_inline的行为(） `
是相同的 `负载(） `。

参见[测试](https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions.py)使用此功能的好例子。

源可省略典型的非直列C ++扩展的两个必需的部分：必要的头包括，以及所述(pybind11）绑定代码。更精确地，字符串传递给`cpp_sources
`首先连接成单个`的.cpp`文件。该文件然后用`前缀的#include  & LT ;torch/ extension.h & GT ;`。

此外，如果`功能 `提供参数，绑定将被自动指定为每个功能产生。 `功能
`可以是函数名的列表，或者从功能名称来文档字符串的字典映射。如果给出一个列表，每个函数的名称作为它的文档字符串。

cuda_sources 被连接到一个单独的`.CU`文件，并通过`torch/ types.h中预先考虑在`来源 `，`cuda.h`和`
cuda_runtime.h`包括。的`的.cpp`和`.CU`的文件被单独编译，但最终连接到单个库。注意，没有绑定在`
cuda_sources`本身为函数生成的。绑定到一个CUDA内核，你必须创建一个C ++函数调用它，无论是申报或`cpp_sources
`的一个定义这个C ++函数(且在HTG36包括它的名字] 功能 `）。

参见 `负载(） `为以下省略的参数的描述。

Parameters

    

  * **cpp_sources** \- 一个字符串或字符串的列表中，含有C ++源代码。

  * **cuda_sources** \- 一个字符串或字符串的列表，包含CUDA源代码。

  * **功能** \- 要为其生成功能绑定函数名称的列表。如果字典中给出，它应该映射函数名的文档字符串(否则只是函数名）。

  * **with_cuda** \- 确定CUDA头和库是否被添加到该生成。如果设置为`无 `(默认），该值被自动确定基于是否`cuda_sources提供 `。其设置为 TRUE`给力CUDA头文件和库包括在内。

Example

    
    
    >>> from torch.utils.cpp_extension import load_inline
    >>> source = '''
    at::Tensor sin_add(at::Tensor x, at::Tensor y) {
      return x.sin() + y.sin();
    }
    '''
    >>> module = load_inline(name='inline_extension',
                             cpp_sources=[source],
                             functions=['sin_add'])
    

`torch.utils.cpp_extension.``include_paths`( _cuda=False_
)[[source]](_modules/torch/utils/cpp_extension.html#include_paths)

    

获取包括建立一个C ++或CUDA扩展所需的路径。

Parameters

    

**CUDA** \- CUDA专用如果真，包括包括路径。

Returns

    

名单包括路径字符串。

`torch.utils.cpp_extension.``check_compiler_abi_compatibility`( _compiler_
)[[source]](_modules/torch/utils/cpp_extension.html#check_compiler_abi_compatibility)

    

验证给定的编译器与PyTorch ABI兼容。

Parameters

    

**编译** ([ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in
Python v3.7\)")） - 编译器可执行文件的名称来检查(例如，`克++`）。必须在shell进程可执行文件。

Returns

    

FALSE如果编译器(可能）ABI-不符合PyTorch，否则真。

`torch.utils.cpp_extension.``verify_ninja_availability`()[[source]](_modules/torch/utils/cpp_extension.html#verify_ninja_availability)

    

返回`真 `如果[忍者](https://ninja-build.org/)打造系统可在系统上。

[Next ![](_static/images/chevron-right-orange.svg)](data.html
"torch.utils.data") [![](_static/images/chevron-right-orange.svg)
Previous](checkpoint.html "torch.utils.checkpoint")

* * *

©版权所有2019年，Torch 贡献者。