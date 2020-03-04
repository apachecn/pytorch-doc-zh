# torch.utils.tensorboard

> 译者：[shuziP](https://github.com/shuziP)
> 
> 校验：[shuziP](https://github.com/shuziP)

在进一步讨论之前，可以在[https://www.tensorflow.org/tensorboard/](https://www.tensorflow.org/tensorboard/)上找到有关TensorBoard的更多详细信息。

一旦你安装TensorBoard，这些工具让您登录PyTorch模型和指标纳入了TensorBoard
UI中的可视化的目录。标量，图像，柱状图，曲线图，和嵌入可视化都支持PyTorch模型和张量以及Caffe2网和斑点。

SummaryWriter类是记录TensorBoard使用和可视化数据的主入口。例如：​ 

    import torch
    import torchvision
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import datasets, transforms
    
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model = torchvision.models.resnet50(False)
    # Have ResNet model take in grayscale rather than RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    images, labels = next(iter(trainloader))
    
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)
    writer.close()


然后可以用TensorBoard可视化，这应该是安装和运行的有：

```
pip install tensorboard
tensorboard --logdir=runs
```


一次实验可以记录很多信息。为了避免混乱的UI，并有更好的聚类的结果，我们可以通过分层命名来对图进行分组。例如，“Loss/train”和“Loss/test”将被分组在一起，而“Accuracy/train”和“Accuracy/test”将分别在TensorBoard接口分组。

    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    
    writer = SummaryWriter()
    
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)


预期结果：

![_images/hier_tags.png](https://pytorch.org/docs/stable/_images/hier_tags.png)

_class_`torch.utils.tensorboard.writer.``SummaryWriter`( _log_dir=None_ ,
_comment=''_ , _purge_step=None_ , _max_queue=10_ , _flush_secs=120_ ,
_filename_suffix=''_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter)

将条目直接写入log_dir中的事件文件中，供TensorBoard使用。

在 SummaryWriter
SummaryWriter类提供了一个高级API，可以在给定的目录中创建事件文件，并向其中添加摘要和事件。该类异步更新文件内容。训练程序调用方法直接从训练循环中向文件添加数据，而不会减慢训练速度。

`__init__`( _log_dir=None_ , _comment=''_ , _purge_step=None_ , _max_queue=10_
, _flush_secs=120_ , _filename_suffix=''_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.__init__)

创建 SummaryWriter 将写出事件和摘要的事件文件。

Parameters

​    

  * **log_dir** (*string*) - 保存目录位置。缺省值是运行/ **CURRENT_DATETIME_HOSTNAME** ，在每次运行后，其改变。采用分层文件夹结构容易运行之间的比较。例如通过在 ‘runs/exp1’, ‘runs/exp2’等，对每一个新实验进行比较。
* **comment** (*string*)  - 添加到默认log_dir后缀的注释log_dir。如果分配了log_dir，则此参数无效
* **purge_step** ([*int*](https://docs.python.org/3/library/functions.html#int))  -当日志记录在步骤T+XT+X崩溃并在步骤TT重新启动时，global_step大于或等于TT的任何事件将被清除并从TensorBoard中隐藏。注意，崩溃和恢复的实验应该具有相同的log_dir。
* **max_queue** ([*int*](https://docs.python.org/3/library/functions.html#int))  - 在其中一个“add”调用强制刷新到磁盘之前，挂起事件和摘要的队列大小。默认为10项。
* **flush_secs** ([*int*](https://docs.python.org/3/library/functions.html#int))   - 将事件挂起和将摘要刷新到磁盘的频率(秒）。默认值是每两分钟一次。
* **filename_suffix** (*string*)   - 添加到日志目录中所有事件文件名的后缀。在tensorboard.summary.writer.event_file_writer.EventFileWriter中有更多细节。

例子：

    from torch.utils.tensorboard import SummaryWriter
    
    # create a summary writer with automatically generated folder name.
    writer = SummaryWriter()
    # folder location: runs/May04_22-14-54_s-MacBook-Pro.local/
    
    # create a summary writer using the specified folder name.
    writer = SummaryWriter("my_experiment")
    # folder location: my_experiment
    
    # create a summary writer with comment appended.
    writer = SummaryWriter(comment="LR_0.1_BATCH_16")
    # folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/


`add_scalar`( _tag_ , _scalar_value_ , _global_step=None_ , _walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_scalar)

​    

标量数据添加到汇总。

Parameters

​    

  * **tag** (*string*)  - 数据标识符
* **scalar_value** ([*float*](https://docs.python.org/3/library/functions.html#float) *or* *string/blobname*)  - 值保存
* **global_step** ([*int*](https://docs.python.org/3/library/functions.html#int)) –- 记录的全局步长值
* **walltime** ([*float*](https://docs.python.org/3/library/functions.html#float)) – 可选覆盖默认的walltime (time.time())，在事件一轮后的几秒内覆盖

Examples:


​    
​    from torch.utils.tensorboard import SummaryWriter
​    writer = SummaryWriter()
​    x = range(100)
​    for i in x:
​        writer.add_scalar('y=2x', i * 2, i)
​    writer.close()


Expected result:

![_images/add_scalar.png](https://pytorch.org/docs/stable/_images/add_scalar.png)

`add_scalars`( _main_tag_ , _tag_scalar_dict_ , _global_step=None_ ,
_walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_scalars)

​    

向摘要添加许多标量数据。

注意，此函数还将记录标量保存在内存中。在极端情况下，它会让你的内存爆满。

Parameters

​    

  * **main_tag** (*string*) – 标记的父名称
* **tag_scalar_dict** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – 存储标记和相应值的键值对
* **global_step** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 要记录的全局步骤值
* **walltime** ([*float*](https://docs.python.org/3/library/functions.html#float)) – 可选的覆盖默认的walltime (time.time())事件历元后

Examples:

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
r = 5
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)
writer.close()
# This call adds three values to the same scalar plot with the tag
# 'run_14h' in TensorBoard's scalar section.
```


Expected result:

![_images/add_scalars.png](https://pytorch.org/docs/stable/_images/add_scalars.png)

`add_histogram`( _tag_ , _values_ , _global_step=None_ , _bins='tensorflow'_ ,
_walltime=None_ , _max_bins=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_histogram)

​    

添加柱状图总结。

Parameters

​    

  * **tag** (*string*) – 数据标识符
* **values** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *numpy.array**, or* *string/blobname*) – 值构建直方图
* **global_step** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 要记录的全局步长值
* **bins** (*string*) – One of {‘tensorflow’,’auto’, ‘fd’, …}. 这决定了bins的制作方式。您可以在以下地址找到其他选项: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
* **walltime** ([*float*](https://docs.python.org/3/library/functions.html#float)) – 可选覆盖默认的walltime (time.time())事件历元后的

Examples:

​    

```
from torch.utils.tensorboard import SummaryWriter
import numpy as np
writer = SummaryWriter()
for i in range(10):
    x = np.random.random(1000)
    writer.add_histogram('distribution centers', x + i, i)
writer.close()
```


Expected result:

![_images/add_histogram.png](https://pytorch.org/docs/stable/_images/add_histogram.png)

`add_image`( _tag_ , _img_tensor_ , _global_step=None_ , _walltime=None_ ,
_dataformats='CHW'_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_image)

​    

将图像数据添加到摘要中。

请注意，这需要 `pillow` 包装。

Parameters

​    

  * **tag** ( _string_ ) – Data identifier

  * **img_tensor** ([ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _numpy.array_ _或_ _串/ blobname_ ) - 图像数据

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Shape:

​    

img_tensor:默认值为(3,H,W) (3,H,W)。您可以使用 `torchvision.utils.make_grid()` 将一批张量转换成3xHxW格式，或者调用 `add_images` ，让我们来完成这项工作。(1,H,W) (1,H,W) (H,W) (H,W) (H,W) (H,W) (H,W,3) (H,W,3)张量也是可以的，只要传递了相应的 `dataformats` 参数。例如:CHW, HWC, HW。

Examples:



    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    img = np.zeros((3, 100, 100))
    img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    
    img_HWC = np.zeros((100, 100, 3))
    img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    
    writer = SummaryWriter()
    writer.add_image('my_image', img, 0)
    
    # If you have non-default dimension setting, set the dataformats argument.
    writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
    writer.close()


Expected result:

![_images/add_image.png](https://pytorch.org/docs/stable/_images/add_image.png)

`add_images`( _tag_ , _img_tensor_ , _global_step=None_ , _walltime=None_ ,
_dataformats='NCHW'_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_images)

​    

成批的图像数据添加到汇总。

Note that this requires the `pillow`package.

请注意，这需要`pillow`package。

Parameters

​    

  * **tag** ( _string_ ) – Data identifier

  * **img_tensor** ([ _torch.Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _numpy.array_ _, or_ _string/blobname_ ) – Image data

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

  * **dataformats**  (_串_ ) - 形式的NCHW，NHWC，CHW，HWC，HW，WH等的图像数据格式规范

Shape:

​    

img_tensor：默认为  (N  ， 3  ， H  ， W  ) (N，3，H，W）   (N  ， 3  ， H  ， W  ) 。如果`
dataformats`被指定，其他形状将被接受。例如NCHW或NHWC。

Examples:


​    

    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    
    img_batch = np.zeros((16, 3, 100, 100))
    for i in range(16):
        img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
        img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i
    
    writer = SummaryWriter()
    writer.add_images('my_image_batch', img_batch, 0)
    writer.close()


Expected result:

![_images/add_images.png](https://pytorch.org/docs/stable/_images/add_images.png)

`add_figure`( _tag_ , _figure_ , _global_step=None_ , _close=True_ ,
_walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_figure)

​    

渲染matplotlib图成图像并将其添加到汇总。

注意，这需要的`matplotlib`包。

Parameters

​    

  * **tag** ( _string_ ) – Data identifier

  * [HTG0图 (_matplotlib.pyplot.figure_ ) - 图或数字的列表

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **关闭** ([ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 标志自动关闭该图

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

`add_video`( _tag_ , _vid_tensor_ , _global_step=None_ , _fps=4_ ,
_walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_video)

​    

视频数据添加到汇总。

注意，这需要的`moviepy`包。

Parameters

​    

  * **tag** (*string*) – Data identifier
* **vid_tensor** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Video data
* **global_step** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Global step value to record
* **fps** ([*float*](https://docs.python.org/3/library/functions.html#float) *or* [*int*](https://docs.python.org/3/library/functions.html#int)) – Frames per second
* **walltime** ([*float*](https://docs.python.org/3/library/functions.html#float)) – Optional override default walltime (time.time()) seconds after epoch of event

Shape:

​    

vid_tensor：  (N  ， T  ， C  ， H  ， W  ) (N，T，C，H，W）  (N  ， T  ， C  ， H  ， W  )
。的值应该位于[0,255]为式 UINT8 或[0,1]类型浮动。

`add_audio`( _tag_ , _snd_tensor_ , _global_step=None_ , _sample_rate=44100_ ,
_walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_audio)

​    

音频数据添加到汇总。

Parameters

​    

  * **tag** ( _string_ ) – Data identifier

  * **snd_tensor** ([ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 声音数据

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **SAMPLE_RATE** ([ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 以Hz采样率

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Shape:

​    

snd_tensor：  (1  ， L  ) (1，L）  (1  ， L  ) 。值应该[-1,1]之间。

`add_text`( _tag_ , _text_string_ , _global_step=None_ , _walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_text)

​    

文本数据添加到汇总。

Parameters

​    

  * **tag** ( _string_ ) – Data identifier

  * **text_string的**  (_串_ ) - 字符串，以节省

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Examples:


​    
​    writer.add_text('lstm', 'This is an lstm', 0)
​    writer.add_text('rnn', 'This is an rnn', 10)


`add_graph`( _model_ , _input_to_model=None_ , _verbose=False_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_graph)

​    

图数据添加到汇总。

Parameters

​    

  * **模型** ([ _torch.nn.Module_ ](nn.html#torch.nn.Module "torch.nn.Module")） - 模型绘制。

  * **input_to_model** ([ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ _torch.Tensor_ 的列表中） - 的变量或变量的元组被输送。

  * **冗长** ([ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 是否打印图形结构在控制台。

`add_embedding`( _mat_ , _metadata=None_ , _label_img=None_ ,
_global_step=None_ , _tag='default'_ , _metadata_header=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_embedding)

​    

添加投影数据嵌入到总结。

Parameters

​    

  * **垫** ([ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ _numpy.array_ ) - 甲矩阵，每一行都是特征向量数据点

  * **元数据** ([ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)")） - 标签的列表，每个元件将转换为串

  * **label_img** ([ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 图像对应于每个数据点

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **标记**  (_串_ ) - 名称为嵌入

Shape:

​    

垫：  (N  ， d  ) (N，d）  (N  ， d  ) ，其中N是数据的数和d是特征尺寸

label_img：  (N  ， C  ， H  ， W  ) (N，C，H，W）  (N  ， C  ， H  ， W  )

Examples:



    import keyword
    import torch
    meta = []
    while len(meta)<100:
        meta = meta+keyword.kwlist # get some strings
    meta = meta[:100]
    
    for i, v in enumerate(meta):
        meta[i] = v+str(i)
    
    label_img = torch.rand(100, 3, 10, 32)
    for i in range(100):
        label_img[i]*=i/100.0
    
    writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
    writer.add_embedding(torch.randn(100, 5), label_img=label_img)
    writer.add_embedding(torch.randn(100, 5), metadata=meta)

`add_pr_curve`( _tag_ , _labels_ , _predictions_ , _global_step=None_ ,
_num_thresholds=127_ , _weights=None_ , _walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_pr_curve)

   



添加精确召回曲线。绘制精确召回曲线可以了解模型在不同阈值设置下的性能。使用此函数，可以为每个目标提供基本真实值标记 (T/F) 和预测置信度(通常是模型的输出）TensorBoard UI将允许您交互地选择阈值。

Parameters

​    

  * **tag** (*string*) –– Data identifier

  * **labels** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *numpy.array**, or* *string/blobname*) – 地面实测数据。每个元素的二进制标签。

  * **predictions** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *numpy.array**, or* *string/blobname*) –  该元素被分类为真概率。值应在[0，1]

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **num_thresholds** ([*int*](https://docs.python.org/3/library/functions.html#int)) –  用于绘制曲线的阈值的数量。

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Examples:


​    

```
from torch.utils.tensorboard import SummaryWriter
import numpy as np
labels = np.random.randint(2, size=100)  # binary label
predictions = np.random.rand(100)
writer = SummaryWriter()
writer.add_pr_curve('pr_curve', labels, predictions, 0)
writer.close()
```

`add_custom_scalars`( _layout_)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_custom_scalars)

通过收集“scalars”中的图表标记创建特殊图表。请注意，对于每个summarywriter(）对象，此函数只能调用一次。因为它只向tensorboard提供元数据，所以可以在训练循环之前或之后调用该函数。

Parameters

​    

layout (dict) - {categoryName: charts}，其中charts也是一个字典{chartName: ListOfProperties}。ListOfProperties中的第一个元素是图表的类型(多行或空白中的一个)，第二个元素应该是包含add_scalar函数中使用的标记的列表，这些标记将被收集到新图表中。

Examples:


​    
​    layout = {'Taiwan':{'twse':['Multiline',['twse/0050', 'twse/2330']]},
​                 'USA':{ 'dow':['Margin',   ['dow/aaa', 'dow/bbb', 'dow/ccc']],
​                      'nasdaq':['Margin',   ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}
​    
​    writer.add_custom_scalars(layout)

`add_mesh`( _tag_ , _vertices_ , _colors=None_ , _faces=None_ ,
_config_dict=None_ , _global_step=None_ , _walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_mesh)

​    

将网格或三维点云添加到TensorBoard。可视化基于three.js，因此它允许用户与呈现的对象交互。除了顶点、面等基本定义外，用户还可以进一步提供相机参数、照明条件等，高级使用请查看https://threejs.org/docs/index.html manual/en/introduction/creating-a-scene。

Parameters

​    

  * **tag** ( _string_ ) – Data identifier

  * **顶点** ([ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 三维坐标列表的顶点。

  * **颜色** ([ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 为每个顶点的色彩

  * **面** ([ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 每个三角形内的顶点指数。 (可选的）

  * **config_dict** \- 字典与ThreeJS类的名称和结构。

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Shape:

​    

vertices: (B, N, 3)  (B  ， N  ， 3  ) (B，N，3）  (B  ， N  ， 3  ) 。 (分批，number_of_vertices，通道）

colors: (B, N, 3)  (B  ， N  ， 3  ) (B，N，3）  (B  ， N  ， 3  ) 。的值应该位于[0,255]为式 UINT8
或[0,1]类型浮动。

faces: (B, N, 3) (B  ， N  ， 3  ) (B，N，3）  (B  ， N  ， 3  ) 。的值应该位于[0，number_of_vertices]为式
UINT8 。

Examples:



​    

    from torch.utils.tensorboard import SummaryWriter
    vertices_tensor = torch.as_tensor([
        [1, 1, 1],
        [-1, -1, 1],
        [1, -1, -1],
        [-1, 1, -1],
    ], dtype=torch.float).unsqueeze(0)
    colors_tensor = torch.as_tensor([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 255],
    ], dtype=torch.int).unsqueeze(0)
    faces_tensor = torch.as_tensor([
        [0, 2, 3],
        [0, 3, 1],
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=torch.int).unsqueeze(0)
    
    writer = SummaryWriter()
    writer.add_mesh('my_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)
    
    writer.close()


`flush`()[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.flush)

​    

刷新事件文件到磁盘。调用此方法，以确保所有未决事件已被写入磁盘。

`close`()[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.close)
