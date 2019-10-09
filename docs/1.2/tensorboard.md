# torch.utils.tensorboard

在进一步讨论之前，可以在[https://www.tensorflow.org/tensorboard/](https://www.tensorflow.org/tensorboard/)上找到有关TensorBoard的更多详细信息。

一旦你安装TensorBoard，这些工具让您登录PyTorch模型和指标纳入了TensorBoard
UI中的可视化的目录。标量，图像，柱状图，曲线图，和嵌入可视化都支持PyTorch模型和张量以及Caffe2网和斑点。

该SummaryWriter类是TensorBoard登录消费和可视化数据的主入口。例如：

    
    
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

    
    
    pip install tb-nightly  # Until 1.14 moves to the release channel
    tensorboard --logdir=runs
    

的大量信息可以记录一个实验。为了避免混乱的UI，并有更好的结果的聚类，通过分级命名它们，我们可以组地块。例如，“损失/火车”和“损耗/试验”将被分组在一起，而“准确度/火车”和“准确度/试验”将分别在TensorBoard接口分组。

    
    
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

    

直接将条目写入事件文件在LOG_DIR由TensorBoard消耗。

在 SummaryWriter
类提供了一个高层次的API来创建指定目录的事件文件，并添加摘要和事件给它。类异步更新文件内容。这使得培训计划，调用方法将数据添加到直接从训练循环的文件，而不会减慢培训。

`__init__`( _log_dir=None_ , _comment=''_ , _purge_step=None_ , _max_queue=10_
, _flush_secs=120_ , _filename_suffix=''_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.__init__)

    

创建 SummaryWriter 将写出事件和摘要的事件文件。

Parameters

    

  * **LOG_DIR** （ _串_ ） - 保存目录位置。缺省值是运行/ **CURRENT_DATETIME_HOSTNAME** ，在每次运行后，其改变。采用分层文件夹结构容易运行之间的比较。例如通过在“运行/ EXP1”，“运行/ EXP2”等，为每一个新的实验在它们之间进行比较。

  * **留言** （ _串_ ） - 评LOG_DIR后缀附加到缺省`LOG_DIR`。如果`LOG_DIR`被分配，这种说法没有任何效果。

  * **purge_step** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 如果记录在步骤 T，崩溃 \+  X  T + X  T  \+  X  和在重新开始步骤 T  T  T  ，其global_step大于或等于 [任何事件HTG71 ]  T  T  T  将被清除和hidde n在TensorBoard。需要注意的是死机了，重新开始实验，应该有相同的`LOG_DIR  [HTG97。`

  * **max_queue** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 为的“添加”前一个未决事件和摘要的队列的大小调用迫使冲洗到磁盘。默认为十个项目。

  * **flush_secs** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 多久，在几秒钟内，冲洗未决事件和摘要到磁盘。默认为每两分钟。

  * **filename_suffix** （ _串_ ） - 后缀添加到在LOG_DIR目录中的所有事件文件名。在文件名施工tensorboard.summary.writer.event_file_writer.EventFileWriter更多细节。

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

    

标量数据添加到汇总。

Parameters

    

  * **标记** （ _串_ ） - 数据标识符

  * **scalar_value** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ _串/ blobname_ ） - 值保存

  * **global_step** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 全球步长值来记录

  * 事件的时期后的可选覆盖默认walltime（了time.time（）），与秒 - **walltime** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")）

Examples:

    
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    x = range(100)
    for i in x:
        writer.add_scalar('y=2x', i * 2, i)
    writer.close()
    

Expected result:

![_images/add_scalar.png](https://pytorch.org/docs/stable/_images/add_scalar.png)

`add_scalars`( _main_tag_ , _tag_scalar_dict_ , _global_step=None_ ,
_walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_scalars)

    

增加了许多标量数据汇总。

请注意，此功能也保持记录的标量在内存中。在极端情况下，它爆炸的RAM。

Parameters

    

  * **main_tag** （ _串_ ） - 为对标签父名

  * **tag_scalar_dict** （[ _DICT_ ](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.7\)")） - 键 - 值对存储标签和相应的值

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **walltime** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 可选覆盖默认walltime（了time.time（））秒事件的时期后

Examples:

    
    
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
    

Expected result:

![_images/add_scalars.png](https://pytorch.org/docs/stable/_images/add_scalars.png)

`add_histogram`( _tag_ , _values_ , _global_step=None_ , _bins='tensorflow'_ ,
_walltime=None_ , _max_bins=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_histogram)

    

添加柱状图总结。

Parameters

    

  * **tag** ( _string_ ) – Data identifier

  * **值** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _numpy.array_ _或_ _串/ blobname_ ） - 值来构建直方图

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **仓** （ _串[HTG3） - 酮的{“tensorflow”，”自动”，‘的fd’，...}。这决定了容器是如何制作。你可以找到其他选项：[ https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html ](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html)_

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Examples:

    
    
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    writer = SummaryWriter()
    for i in range(10):
        x = np.random.random(1000)
        writer.add_histogram('distribution centers', x + i, i)
    writer.close()
    

Expected result:

![_images/add_histogram.png](https://pytorch.org/docs/stable/_images/add_histogram.png)

`add_image`( _tag_ , _img_tensor_ , _global_step=None_ , _walltime=None_ ,
_dataformats='CHW'_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_image)

    

图像数据添加到汇总。

注意，这需要的`枕 `包。

Parameters

    

  * **tag** ( _string_ ) – Data identifier

  * **img_tensor** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _numpy.array_ _或_ _串/ blobname_ ） - 图像数据

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Shape:

    

img_tensor：默认为 （ 3  ， H  ， W  ） （3，H，W） （ 3  ， H  ， W  ） 。您可以使用`
torchvision.utils.make_grid（） `对批量张量转换成3xHxW格式或致电`add_images`，让我们做工作。张量 （ 1
， H  ， W  ） （1，H，W） （ 1  ， H  ， W  ） ， （ H  ， W  ） （H，W） （ H  ， W  ） ， [H
TG158]  （ H  ， W  ， 3  ） （H，W，3） （ H  ， W  ， 3  ） 也只要suitible作为对应`
`参数传递dataformats。例如CHW，HWC，HW。

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

    

成批的图像数据添加到汇总。

Note that this requires the `pillow`package.

Parameters

    

  * **tag** ( _string_ ) – Data identifier

  * **img_tensor** ([ _torch.Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _numpy.array_ _, or_ _string/blobname_ ) – Image data

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

  * **dataformats** （ _串_ ） - 形式的NCHW，NHWC，CHW，HWC，HW，WH等的图像数据格式规范

Shape:

    

img_tensor：默认为 （ N  ， 3  ， H  ， W  ） （N，3，H，W）  （ N  ， 3  ， H  ， W  ） 。如果`
dataformats`被指定，其他形状将被接受。例如NCHW或NHWC。

Examples:

    
    
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

    

渲染matplotlib图成图像并将其添加到汇总。

注意，这需要的`matplotlib`包。

Parameters

    

  * **tag** ( _string_ ) – Data identifier

  * [HTG0图（ _matplotlib.pyplot.figure_ ） - 图或数字的列表

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **关闭** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 标志自动关闭该图

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

`add_video`( _tag_ , _vid_tensor_ , _global_step=None_ , _fps=4_ ,
_walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_video)

    

视频数据添加到汇总。

注意，这需要的`moviepy`包。

Parameters

    

  * **tag** ( _string_ ) – Data identifier

  * **vid_tensor** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 视频数据

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **FPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 框子每秒

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Shape:

    

vid_tensor： （ N  ， T  ， C  ， H  ， W  ） （N，T，C，H，W） （ N  ， T  ， C  ， H  ， W  ）
。的值应该位于[0,255]为式 UINT8 或[0,1]类型浮动。

`add_audio`( _tag_ , _snd_tensor_ , _global_step=None_ , _sample_rate=44100_ ,
_walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_audio)

    

音频数据添加到汇总。

Parameters

    

  * **tag** ( _string_ ) – Data identifier

  * **snd_tensor** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 声音数据

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **SAMPLE_RATE** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 以Hz采样率

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Shape:

    

snd_tensor： （ 1  ， L  ） （1，L） （ 1  ， L  ） 。值应该[-1,1]之间。

`add_text`( _tag_ , _text_string_ , _global_step=None_ , _walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_text)

    

文本数据添加到汇总。

Parameters

    

  * **tag** ( _string_ ) – Data identifier

  * **text_string的** （ _串_ ） - 字符串，以节省

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Examples:

    
    
    writer.add_text('lstm', 'This is an lstm', 0)
    writer.add_text('rnn', 'This is an rnn', 10)
    

`add_graph`( _model_ , _input_to_model=None_ , _verbose=False_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_graph)

    

图数据添加到汇总。

Parameters

    

  * **模型** （[ _torch.nn.Module_ ](nn.html#torch.nn.Module "torch.nn.Module")） - 模型绘制。

  * **input_to_model** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ _torch.Tensor_ 的列表中） - 的变量或变量的元组被输送。

  * **冗长** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 是否打印图形结构在控制台。

`add_embedding`( _mat_ , _metadata=None_ , _label_img=None_ ,
_global_step=None_ , _tag='default'_ , _metadata_header=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_embedding)

    

添加投影数据嵌入到总结。

Parameters

    

  * **垫** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ _numpy.array_ ） - 甲矩阵，每一行都是特征向量数据点

  * **元数据** （[ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)")） - 标签的列表，每个元件将转换为串

  * **label_img** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 图像对应于每个数据点

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **标记** （ _串_ ） - 名称为嵌入

Shape:

    

垫： （ N  ， d  ） （N，d） （ N  ， d  ） ，其中N是数据的数和d是特征尺寸

label_img： （ N  ， C  ， H  ， W  ） （N，C，H，W） （ N  ， C  ， H  ， W  ）

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

    

增加精度召回曲线。绘制精确召回曲线，让你了解下不同的阈值设置模型的性能。有了这个功能，你所提供的地面实况标签（T /
F），并为每个目标预测置信（通常是模型的输出）。该TensorBoard UI会让你选择的门槛交互。

Parameters

    

  * **tag** ( _string_ ) – Data identifier

  * **标签** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _numpy.array_ _或_ _串/ blobname_ ） - 地面实测数据。每个元素的二进制标签。

  * **的预测** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _numpy.array_ _或_ _串/ blobname_ ） - 该元素被分类为真概率。值应在[0，1]

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **num_thresholds** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 用于绘制曲线的阈值的数量。

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Examples:

    
    
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    labels = np.random.randint(2, size=100)  # binary label
    predictions = np.random.rand(100)
    writer = SummaryWriter()
    writer.add_pr_curve('pr_curve', labels, predictions, 0)
    writer.close()
    

`add_custom_scalars`( _layout_)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_custom_scalars)

    

在“标量”收集图表标签创建专题图。请注意，此功能只能调用一次，每个SummaryWriter（）对象。因为它仅提供元数据tensorboard，该功能可以前或训练后循环调用。

Parameters

    

**布局** （[ _DICT_ ](https://docs.python.org/3/library/stdtypes.html#dict "\(in
Python v3.7\)")） - {类别名称： _图表_ }，其中 _图表_ 也是一个字典{chartName： _ListOfProperties_
}。在 _ListOfProperties_ 的第一个元素是图表的类型（ **一个多行** 或 **保证金**
）和所述第二元件应为包含的标签列表已在add_scalar使用功能，这将被收集到新的图表。

Examples:

    
    
    layout = {'Taiwan':{'twse':['Multiline',['twse/0050', 'twse/2330']]},
                 'USA':{ 'dow':['Margin',   ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                      'nasdaq':['Margin',   ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}
    
    writer.add_custom_scalars(layout)
    

`add_mesh`( _tag_ , _vertices_ , _colors=None_ , _faces=None_ ,
_config_dict=None_ , _global_step=None_ , _walltime=None_
)[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_mesh)

    

添加网格或三维点云TensorBoard。可视化是基于three.js所，所以它允许用户与描绘对象进行交互。除了诸如顶点，脸上的基本定义，用户可以进一步提供摄像机参数，照明条件等，请参见[
https://threejs.org/docs/index.html#manual/en/introduction/Creating-a
-scene用于高级用途](https://threejs.org/docs/index.html#manual/en/introduction/Creating-
a-scene)。需要注意的是目前这取决于TB-夜间显示。

Parameters

    

  * **tag** ( _string_ ) – Data identifier

  * **顶点** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 三维坐标列表的顶点。

  * **颜色** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 为每个顶点的色彩

  * **面** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 每个三角形内的顶点指数。 （可选的）

  * **config_dict** \- 字典与ThreeJS类的名称和结构。

  * **global_step** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Global step value to record

  * **walltime** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Optional override default walltime (time.time()) seconds after epoch of event

Shape:

    

顶点： （ B  ， N  ， 3  ） （B，N，3） （ B  ， N  ， 3  ） 。 （分批，number_of_vertices，通道）

颜色： （ B  ， N  ， 3  ） （B，N，3） （ B  ， N  ， 3  ） 。的值应该位于[0,255]为式 UINT8
或[0,1]类型浮动。

面： （ B  ， N  ， 3  ） （B，N，3） （ B  ， N  ， 3  ） 。的值应该位于[0，number_of_vertices]为式
UINT8 。

Examples:

    
    
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

    

刷新事件文件到磁盘。调用此方法，以确保所有未决事件已被写入磁盘。

`close`()[[source]](_modules/torch/utils/tensorboard/writer.html#SummaryWriter.close)
