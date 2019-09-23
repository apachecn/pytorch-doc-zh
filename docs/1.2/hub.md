# torch.hub

Pytorch中心是一个旨在促进研究再现性的预训练模型库。

## 出版模式

Pytorch中心通过添加一个简单的`hubconf.py`文件支持发布预先训练模型（模型定义和预训练的权重）到GitHub的库;

`hubconf.py`可以有多个入口点。每个入口点被定义为一个Python函数（例如：您要发布一个预先训练模型）。

    
    
    def entrypoint_name(*args, **kwargs):
        # args & kwargs are optional, for models which take positional/keyword arguments.
        ...
    

### 如何实现一个入口点？

下面的代码片段指定`resnet18`如果我们扩大在`pytorch /视觉/ hubconf.py`实现模型的入口点。在大多数情况下，在导入`
hubconf.py`右边的功能就足够了。在这里，我们只是想用扩展版本作为一个例子来说明它是如何工作的。你可以看到在[ pytorch
/视觉回购的完整剧本](https://github.com/pytorch/vision/blob/master/hubconf.py)

    
    
    dependencies = ['torch']
    from torchvision.models.resnet import resnet18 as _resnet18
    
    # resnet18 is the name of entrypoint
    def resnet18(pretrained=False, **kwargs):
        """ # This docstring shows up in hub.help()
        Resnet18 model
        pretrained (bool): kwargs, load pretrained weights into the model
        """
        # Call the model, load pretrained weights
        model = _resnet18(pretrained=pretrained, **kwargs)
        return model
    

  * `依赖性 `变量是需要 **负载** 该模型包名的 **列表。请注意，这可能是从训练模型需要的依赖略有不同。**

  * `ARGS`和`kwargs`沿着到真正的可调用的函数传递。

  * 该函数的文档字符串可以作为一个帮助信息。它说明了什么呢模型做什么都允许的位置/关键字参数。我们强烈建议在这里补充几个例子。

  * 入口函数可以返回一个模型（nn.module），或辅助工具，使所述用户的工作流平滑，例如断词。

  * 以下划线前缀可调用被认为是辅助功能，这将不能在`torch.hub.list（） `显示。

  * 预训练的权重可以是在GitHub库存储在本地，或者通过`torch.hub.load_state_dict_from_url（） `装载。如果小于2GB，建议将其连接到[项目发布](https://help.github.com/en/articles/distributing-large-binaries)和释放使用的URL。在上述`torchvision.models.resnet.resnet18`手柄`预训练 `，或者可以把下面的逻辑在入口点定义的例子。

    
    
    if pretrained:
        # For checkpoint saved in local github repo, e.g. <RELATIVE_PATH_TO_CHECKPOINT>=weights/save.pth
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, <RELATIVE_PATH_TO_CHECKPOINT>)
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
    
        # For checkpoint saved elsewhere
        checkpoint = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
    

### 重要提示

  * 已公布的车型应该是至少的一个分支/标签。它不能是一个随机的承诺。

## 从轮毂承载量模型

Pytorch集线器提供了方便的API通过`torch.hub.list探索毂所有可用的模型（）到`torch.hub.help
`，显示文档字符串和实施例（ ） `和负载使用`torch.hub.load（）预先训练的模型 `

`torch.hub.``list`( _github_ , _force_reload=False_
)[[source]](_modules/torch/hub.html#list)

    

列出 github上 hubconf所有可用的入口点。

Parameters

    

  * **的github** \- 所需的，具有格式“repo_owner / repo_name [：TAG_NAME]”的字符串与任选的标记/分支。默认分支为主如果未指定。例如：“pytorch /视力[：毂]”

  * **force_reload** \- 可选，是否放弃现有缓存并强制新鲜下载。默认为假[HTG3。

Returns

    

可用的入口点名称的列表

Return type

    

入口点

例

    
    
    >>> entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
    

`torch.hub.``help`( _github_ , _model_ , _force_reload=False_
)[[source]](_modules/torch/hub.html#help)

    

显示入口点模型的文档字符串。

Parameters

    

  * **的github** \- 所需的，与格式的字符串& LT ; repo_owner / repo_name [：TAG_NAME] & GT ;具有任选的标记/分支。默认分支为主如果未指定。例如：“pytorch /视力[：毂]”

  * **模型** \- 必须在回购的hubconf.py定义的入口点名称的字符串

  * **force_reload** – Optional, whether to discard the existing cache and force a fresh download. Default is False.

Example

    
    
    >>> print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))
    

`torch.hub.``load`( _github_ , _model_ , _*args_ , _**kwargs_
)[[source]](_modules/torch/hub.html#load)

    

从GitHub库加载模式，与预训练的权重。

Parameters

    

  * **github** – Required, a string with format “repo_owner/repo_name[:tag_name]” with an optional tag/branch. The default branch is master if not specified. Example: ‘pytorch/vision[:hub]’

  * **model** – Required, a string of entrypoint name defined in repo’s hubconf.py

  * *** ARGS** \- 可选，用于可调用模型相应ARGS 。

  * **force_reload** \- 可选，是否强制GitHub库的新鲜下载无条件。默认为假[HTG3。

  * **** kwargs** \- 可选，用于可调用模型相应kwargs 。

Returns

    

一个单一的模型与对应的预训练的权重。

Example

    
    
    >>> model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
    

### 运行加载的模型：HTG0]

注意，`*指定参数时， ** kwargs`在`torch.load（） `用于 **实例**
的模型。您加载模型之后，你怎么能找出你可以与模型做什么呢？一个建议的工作流程

  * `DIR（模型） `以查看模型的所有可用的方法。

  * `帮助（model.foo）HTG2] `检查哪些参数`model.foo`需要运行

为了帮助用户浏览，而不指的文档来回，我们强烈建议回购业主进行函数帮助信息清晰而简洁。这也有利于包括最小工作示例。

### 我下载的模型保存？

的位置是在的顺序使用

  * 主叫`hub.set_dir（& LT ; PATH_TO_HUB_DIR & GT ;） `

  * `$ TORCH_HOME /集线器 `时，如果环境变量`TORCH_HOME`被设置。

  * `$ XDG_CACHE_HOME /torch/集线器 `时，如果环境变量`XDG_CACHE_HOME`被设置。

  * `〜/ .cache /torch/集线器 `

`torch.hub.``set_dir`( _d_ )[[source]](_modules/torch/hub.html#set_dir)

    

（可选）设置hub_dir到本地目录保存下载的模型&放;权重。

如果`set_dir`不叫，缺省路径为`$ TORCH_HOME /集线器 `其中环境变量`$ TORCH_HOME`默认为`$
XDG_CACHE_HOME /torch `。 `$ XDG_CACHE_HOME`遵循了Linux
filesytem布局的X设计集团说明书中，具有缺省值`〜/ .cache`如果环境变量未设置。

Parameters

    

**d** \- 路径到本地文件夹来保存下载的模型&放;权重。

### 高速缓存逻辑

默认情况下，我们不会加载它清理干净后的文件。集线器默认使用的缓存，如果它已经在`hub_dir`存在。

用户可以通过调用`hub.load强迫重载（...， force_reload =真）
`。这将删除现有GitHub的文件夹和下载的权重，重新初始化一个新的下载。当更新发布到同一分支，这非常有用，用户可以使用最新版本跟上。

### 已知的限制：

Torch 中心的工作原理是，如果它是安装包导入。还有就是通过导入Python中引入了一些副作用。例如，你可以看到新的项目在Python缓存`
sys.modules中 `和`sys.path_importer_cache`这是正常的Python行为。

已知的限制，即这里值得一提是用户 **不能** 负载在 **相同蟒过程**
相同回购两个不同的分支。这就像在Python中相同的名称，这是不好的安装两个包。缓存可能入党，给你惊喜，如果你真的尝试。当然，这是完全正常加载它们在单独的进程。

[Next ![](_static/images/chevron-right-orange.svg)](jit.html "TorchScript")
[![](_static/images/chevron-right-orange.svg) Previous](distributions.html
"Probability distributions - torch.distributions")

* * *

©版权所有2019年，Torch 贡献者。