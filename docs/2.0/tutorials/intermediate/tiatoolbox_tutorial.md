# 使用TIAToolbox和pytorch 分类医疗扫描图像 [¶](#Whole-Slide-Image-Classification-Using-PyTorch-and-TIAToolbox "永久链接到此标题")  


> 译者：[歌尔股东](https://github.com/sanxincao)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/Whole_Slide_Image_Classification_Using_PyTorch_and_TIAToolbox>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/tiatoolbox_tutorial.html>  


为了充分利用本教程，我们建议使用[Colab版本](https://colab.research.google.com/github/pytorch/tutorials/blob/main/_static/tiatoolbox_tutorial.ipynb)。这将允许您对照下面提供的信息进行实验。  
  
## 介绍  
在本教程中，我们将展示如何在TIAToolbox的帮助下使用PyTorch深度学习模型对整个CT扫描图像(WSIs)进行分类。WSI是通过手术或活检获得的人体组织样本的图像，并使用专门的扫描仪进行扫描。病理学家和计算病理学研究人员使用它们在微观水平上研究癌症等疾病，以了解例如肿瘤的生长并帮助改善患者的治疗效果。  

<img src='https://pytorch.org/tutorials/_images/read_bounds_tissue.webp' width=50% />


使扫描图像难以处理的是它们的巨大尺寸。例如，典型的扫描图像大约有100,000x100,000像素，其中每个像素对应于扫描图像上的0.25x0.25微米。这给加载和处理这些图像带来了挑战，更不用说在单个研究中使用数百甚至数千个扫描图像(更大的研究产生更好的结果)。

传统的图像处理管道不适合WSI处理所以我们需要更好的工具。这就是TIAToolbox可以提供帮助的地方，因为它提供了一套有用的工具，以快速和计算高效的方式导入和处理组织幻灯片。通常，wsi保存在金字塔结构中，具有为可视化而优化的不同放大级别的同一图像的多个副本。金字塔的0级(或最底层)包含最高放大倍数或缩放级别的图像，而金字塔中较高的级别具有较低分辨率的基础图像副本.  

Tiatoolbox使我们能够自动化常见的下游分析任务，例如组织分类。在本教程中，我们将展示你能做到什么：1。使用tiatoolbox加载WSI图像；2.使用不同的pytorch模型WSI分类为块级。在本教程中，我们将提供一个使用Torchvision Resnet18模型和自定义HistoEncoder <https://github.com/jopo666/histoencododer>`__模型的示例。

## 设置环境  
要运行本教程中提供的示例，需要以下包作为先决条件。  
* OpenJpeg
* OpenSlide
* Pixman
* TIAToolbox
* HistoEncoder (for a custom model example)

请在终端上运行以下命令来安装这些软件包  
```
apt-get -y -qq install libopenjp2-7-dev libopenjp2-tools openslide-tools libpixman-1-dev pip install -q 'tiatoolbox<1.5' histoencoder && echo "Installation is done."
```
或者，您可以运行brew install openjpeg openslide在MacOS上安装必备软件包，而不是apt-get。[关于安装的更多信息可以在这里找到。](https://tia-toolbox.readthedocs.io/en/latest/installation.html)  


## 导入相关包
```
"""Import modules required to run the Jupyter notebook."""
from __future__ import annotations

# Configure logging
import logging
import warnings
if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

# Downloading data and files
import shutil
from pathlib import Path
from zipfile import ZipFile

# Data processing and visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
import PIL
import contextlib
import io
from sklearn.metrics import accuracy_score, confusion_matrix

# TIAToolbox for WSI loading and processing
from tiatoolbox import logger
from tiatoolbox.models.architecture import vanilla
from tiatoolbox.models.engine.patch_predictor import (
    IOPatchPredictorConfig,
    PatchPredictor,
)
from tiatoolbox.utils.misc import download_data, grab_files_from_dir
from tiatoolbox.utils.visualization import overlay_prediction_mask
from tiatoolbox.wsicore.wsireader import WSIReader

# Torch-related
import torch
from torchvision import transforms

# Configure plotting
mpl.rcParams["figure.dpi"] = 160  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode

# If you are not using GPU, change ON_GPU to False
ON_GPU = True

# Function to suppress console output for overly verbose code blocks
def suppress_console_output():
    return contextlib.redirect_stderr(io.StringIO())
```

## 运行前的清理工作  
为了确保正确的清理，在此运行中下载或创建的所有文件均保存在单个目录global_save_dir中，我们将其设置为“ ./ TMP/”。为了简化维护，目录的名称仅在这个地方发生，因此如果需要，可以轻松更改它。

```
warnings.filterwarnings("ignore")
global_save_dir = Path("./tmp/")


def rmdir(dir_path: str | Path) -> None:
    """Helper function to delete directory."""
    if Path(dir_path).is_dir():
        shutil.rmtree(dir_path)
        logger.info("Removing directory %s", dir_path)


rmdir(global_save_dir)  # remove  directory if it exists from previous runs
global_save_dir.mkdir()
logger.info("Creating new directory %s", global_save_dir)
```
## 下载数据
```
wsi_path = global_save_dir / "sample_wsi.svs"
patches_path = global_save_dir / "kather100k-validation-sample.zip"
weights_path = global_save_dir / "resnet18-kather100k.pth"

logger.info("Download has started. Please wait...")

# Downloading and unzip a sample whole-slide image
download_data(
    "https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs",
    wsi_path,
)

# Download and unzip a sample of the validation set used to train the Kather 100K dataset
download_data(
    "https://tiatoolbox.dcs.warwick.ac.uk/datasets/kather100k-validation-sample.zip",
    patches_path,
)
with ZipFile(patches_path, "r") as zipfile:
    zipfile.extractall(path=global_save_dir)

# Download pretrained model weights for WSI classification using ResNet18 architecture
download_data(
    "https://tiatoolbox.dcs.warwick.ac.uk/models/pc/resnet18-kather100k.pth",
    weights_path,
)

logger.info("Download is complete.")
```

## 读取数据  
源文档写了一堆废话，总结，将文件路径与类别标签对应起来（省流！）
```
# Read the patch data and create a list of patches and a list of corresponding labels
dataset_path = global_save_dir / "kather100k-validation-sample"

# Set the path to the dataset
image_ext = ".tif"  # file extension of each image

# Obtain the mapping between the label ID and the class name
label_dict = {
    "BACK": 0, # Background (empty glass region)
    "NORM": 1, # Normal colon mucosa
    "DEB": 2,  # Debris
    "TUM": 3,  # Colorectal adenocarcinoma epithelium
    "ADI": 4,  # Adipose
    "MUC": 5,  # Mucus
    "MUS": 6,  # Smooth muscle
    "STR": 7,  # Cancer-associated stroma
    "LYM": 8,  # Lymphocytes
}

class_names = list(label_dict.keys())
class_labels = list(label_dict.values())

# Generate a list of patches and generate the label from the filename
patch_list = []
label_list = []
for class_name, label in label_dict.items():
    dataset_class_path = dataset_path / class_name
    patch_list_single_class = grab_files_from_dir(
        dataset_class_path,
        file_types="*" + image_ext,
    )
    patch_list.extend(patch_list_single_class)
    label_list.extend([label] * len(patch_list_single_class))

# Show some dataset statistics
plt.bar(class_names, [label_list.count(label) for label in class_labels])
plt.xlabel("Patch types")
plt.ylabel("Number of patches")

# Count the number of examples per class
for class_name, label in label_dict.items():
    logger.info(
        "Class ID: %d -- Class Name: %s -- Number of images: %d",
        label,
        class_name,
        label_list.count(label),
    )

# Overall dataset statistics
logger.info("Total number of patches: %d", (len(patch_list)))
```

<img src='https://pytorch.org/tutorials/_images/tiatoolbox_tutorial_001.png
' width=50% />  
如您所见，对于此数据集，我们有9个具有IDS 0-8和关联类名称的类/标签。描述斑块中的主要组织类型：  
* BACK  背景（无意义空区域）
* LYM   淋巴
* NORM  正常结肠黏膜(???)
* DEB   凋零
* MUS   平滑肌
* STR   癌症组织
* ADI   脂肪
* MUC   粘液
* TUM   结直肠腺癌

## 分类图像块

译者注：原文档中出现较多次的patch（块）。在医疗图像处理中，"patches"通常指的是从大型图像（如全切片图像）中提取的较小的图像区域。这些区域通常是正方形或矩形，并且大小是固定的。例如，你可能会从一个10000x10000像素的图像中提取出多个512x512像素的patches。

在这个特定的例子中，每个patch都被标记为一种特定的组织类型（如背景、淋巴、正常结肠黏膜等）。这些标签可以用于训练一个分类模型，该模型的任务是预测新的patch属于哪种组织类型。

我们演示了如何首先使用分块模式获得WSI中的每个块的预测，然后使用wsi模式获得整个WSI的预测。

## 定义PatchPredictor模型
PatchPredictor类运行一个用PyTorch编写的基于cnn的分类器。

* model可以是任何经过训练的PyTorch模型，其约束是：它应该遵循tiatoolbox.models.abc.ModelABC <https://tia-toolbox.readthedocs.io/en/latest/_autosummary/tiatoolbox.models.models_abc.ModelABC.html>`。有关此的更多信息，请参考[our example notebook on advanced model techniques](https://github.com/TissueImageAnalytics/tiatoolbox/blob/develop/examples/07-advanced-modeling.ipynb)
为了加载自定义模型，您需要编写一个小的预处理函数，如在preproc函数(img)中，它确保输入张量是加载网络的正确格式。  
* 或者，您可以将预训练模型作为字符串参数传递。这指定了执行预测的CNN模型，并且它必须是这里列出的[模型之一](https://tia-toolbox.readthedocs.io/en/latest/usage.html?highlight=pretrained%20models#tiatoolbox.models.architecture.get_pretrained_model)。代码看起来像这样 predictor = PatchPredictor(pretrained_model='resnet18-kather100k', pretrained_weights=weights_path, batch_size=32).
* pretrained_weights:使用预训练模型时，默认也会下载相应的预训练权值。您可以通过预训练的权重参数用自己的权重集覆盖默认值。
* batch_size:每次输入模型的图像数量。该参数值越大，要求GPU的内存容量越大。
```
# Importing a pretrained PyTorch model from TIAToolbox
predictor = PatchPredictor(pretrained_model='resnet18-kather100k', batch_size=32)

# Users can load any PyTorch model architecture instead using the following script
model = vanilla.CNNModel(backbone="resnet18", num_classes=9) # Importing model from torchvision.models.resnet18
model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
def preproc_func(img):
    img = PIL.Image.fromarray(img)
    img = transforms.ToTensor()(img)
    return img.permute(1, 2, 0)
model.preproc_func = preproc_func
predictor = PatchPredictor(model=model, batch_size=32)
 ```
## 预测块标签 
我们创建一个预测器对象，然后使用块模式调用预测方法。然后计算分类精度和混淆矩阵。
```
with suppress_console_output():
    output = predictor.predict(imgs=patch_list, mode="patch", on_gpu=ON_GPU)

acc = accuracy_score(label_list, output["predictions"])
logger.info("Classification accuracy: %f", acc)

# Creating and visualizing the confusion matrix for patch classification results
conf = confusion_matrix(label_list, output["predictions"], normalize="true")
df_cm = pd.DataFrame(conf, index=class_names, columns=class_names)
df_cm
```
## 预测整个图片的贴片标签
现在我们引入IOPatchPredictorConfig，这个类为模型预测引擎指定图像读取和预测写入的配置。这需要通知分类器应该读取WSI金字塔的哪一层，处理数据并生成输出。
IOPatchPredictorConfig的参数定义如下：
* input_resolutions: 字典形式的列表，指定每个输入的解析。列表元素必须与目标模型中的顺序相同。forward()如果你的模型只接受一个输入，你只需要放入一个字典指定“单位”和“分辨率”。请注意，TIAToolbox支持具有多个输入的模型。有关单位和分辨率的更多信息。[TIAToolbox documentation](https://tia-toolbox.readthedocs.io/en/latest/_autosummary/tiatoolbox.wsicore.wsireader.WSIReader.html#tiatoolbox.wsicore.wsireader.WSIReader.read_rect)
* patch_input_shape:最大输入(高度，宽度)格式。
* stride_shape: 在块提取过程中使用的两个连续块之间的步幅（步长）的大小。如果用户将stride_shape等于patch_input_shape设置，则将提取和处理块而不会重叠。
```
wsi_ioconfig = IOPatchPredictorConfig(
    input_resolutions=[{"units": "mpp", "resolution": 0.5}],
    patch_input_shape=[224, 224],
    stride_shape=[224, 224],
)
```
预测方法将CNN应用于输入patch上，得到预测结果。以下是论点及其描述:  
* mode: 要处理的输入类型。根据您的应用程序选择patch, tile或wsi。
* imgs: 输入列表，它应该是到输入块或wsi的路径列表。
* return_probabilities: 将其设置为True，可以获取输入贴片的每个类别的概率以及预测标签。如果你希望合并预测结果以生成tile或wsi模式的预测map，你可以设置return_probabilities=True。
* ioconfig:使用IOPatchPredictorConfig类设置IO配置信息。
* resolution and unit (并无演示):这些参数指定我们计划从中提取块的WSI级别的级别或微米/像素分辨率，可以代替ioconfig使用。这里我们将WSI级别指定为“基线”，相当于级别0。一般来说，这是分辨率最高的水平。在本例中，图像只有一个级别。更多信息可以在 [文档](https://tia-toolbox.readthedocs.io/en/latest/usage.html?highlight=WSIReader.read_rect#tiatoolbox.wsicore.wsireader.WSIReader.read_rect)中找到。
* masks: 对应于imgs列表中WSIs的掩码的路径列表。这些掩码指定了我们想从原始WSIs中提取贴片的区域。如果特定WSI的掩码被指定为None，那么该WSI的所有贴片（甚至包括背景区域）的标签都将被预测。这可能会导致不必要的计算。
* merge_predictions:如果需要生成贴片分类结果的二维地图，你可以将此参数设置为True。然而，对于大型WSIs，这将需要大量的可用内存。另一种（默认）解决方案是将merge_predictions设置为False，然后使用merge_predictions函数生成二维预测地图，你稍后会看到。
  
由于我们使用的是大型WSI，块提取和预测过程可能需要一些时间(如果您可以访问启用Cuda的GPU和PyTorch+Cuda，请确保将ON GPU设置为True)
```
with suppress_console_output():
    wsi_output = predictor.predict(
        imgs=[wsi_path],
        masks=None,
        mode="wsi",
        merge_predictions=False,
        ioconfig=wsi_ioconfig,
        return_probabilities=True,
        save_dir=global_save_dir / "wsi_predictions",
        on_gpu=ON_GPU,
    )
```
我们通过可视化wsi_output来看预测模型如何在我们的全切片图像上工作。我们首先需要合并贴片预测输出，然后将它们作为原始图像的覆盖层进行可视化。和之前一样，merge_predictions方法被用来合并贴片预测。这里我们设置参数resolution=1.25, units='power'来生成1.25倍放大的预测地图。如果你想要有更高/更低分辨率（更大/更小）的预测地图，你需要相应地改变这些参数。当预测被合并时，使用overlay_patch_prediction函数将预测地图覆盖在WSI缩略图上，这应该在用于预测合并的分辨率下提取。
```
overview_resolution = (
    4  # the resolution in which we desire to merge and visualize the patch predictions
)
# the unit of the `resolution` parameter. Can be "power", "level", "mpp", or "baseline"
overview_unit = "mpp"
wsi = WSIReader.open(wsi_path)
wsi_overview = wsi.slide_thumbnail(resolution=overview_resolution, units=overview_unit)
plt.figure(), plt.imshow(wsi_overview)
plt.axis("off")
```


在此图像上叠加预测图如下所示:  
```
# Visualization of whole-slide image patch-level prediction
# first set up a label to color mapping
label_color_dict = {}
label_color_dict[0] = ("empty", (0, 0, 0))
colors = cm.get_cmap("Set1").colors
for class_name, label in label_dict.items():
    label_color_dict[label + 1] = (class_name, 255 * np.array(colors[label]))

pred_map = predictor.merge_predictions(
    wsi_path,
    wsi_output[0],
    resolution=overview_resolution,
    units=overview_unit,
)
overlay = overlay_prediction_mask(
    wsi_overview,
    pred_map,
    alpha=0.5,
    label_info=label_color_dict,
    return_ax=True,
)
plt.show()
```

<img src='https://pytorch.org/tutorials/_images/tiatoolbox_tutorial_003.png' width=20% />  

## 基于特定病理模型的特征提取
在本节中，我们将展示如何使用TIAToolbox提供的WSI推理引擎，从存在于TIAToolbox之外的预训练PyTorch模型中提取特征。为了说明这一点，我们将使用HistoEncoder，这是一种以自监督方式训练的计算病理学特定模型，用于从组织学图像中提取特征。该模型已在这里提供:  
‘HistoEncoder: Foundation models for digital pathology’ (https://github.com/jopo666/HistoEncoder) by Pohjonen, Joona and team at the University of Helsinki.  
我们将绘制特征图的3D（RGB）的umap降维，以可视化特征如何捕捉上述一些组织类型之间的差异。
```
# Import some extra modules
import histoencoder.functional as F
import torch.nn as nn

from tiatoolbox.models.engine.semantic_segmentor import DeepFeatureExtractor, IOSegmentorConfig
from tiatoolbox.models.models_abc import ModelABC
import umap
```
TIAToolbox定义了一个ModelABC，这是一个继承了PyTorch nn.Module的类，并指定了要在TIAToolbox推理引擎中使用的模型应该是什么样子。histoencoder模型并不遵循这种结构，所以我们需要将它包装在一个类中，这个类的输出和方法是TIAToolbox引擎期望的。
```
class HistoEncWrapper(ModelABC):
    """Wrapper for HistoEnc model that conforms to tiatoolbox ModelABC interface."""

    def __init__(self: HistoEncWrapper, encoder) -> None:
        super().__init__()
        self.feat_extract = encoder

    def forward(self: HistoEncWrapper, imgs: torch.Tensor) -> torch.Tensor:
        """Pass input data through the model.

        Args:
            imgs (torch.Tensor):
                Model input.

        """
        out = F.extract_features(self.feat_extract, imgs, num_blocks=2, avg_pool=True)
        return out

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        *,
        on_gpu: bool,
    ) -> list[np.ndarray]:
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (torch.Tensor):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        """
        img_patches_device = batch_data.to('cuda') if on_gpu else batch_data
        model.eval()
        # Do not compute the gradient (not training)
        with torch.inference_mode():
            output = model(img_patches_device)
        return [output.cpu().numpy()]
```
现在我们有了包装器，我们将创建我们的特征提取模型并实例化一个DeepFeatureExtractor，以允许我们在WSI上使用这个模型。我们将使用与上面相同的WSI，但这次我们将使用HistoEncoder模型从WSI的块中提取特征，而不是为每个块预测一些标签。
```
# create the model
encoder = F.create_encoder("prostate_medium")
model = HistoEncWrapper(encoder)

# set the pre-processing function
norm=transforms.Normalize(mean=[0.662, 0.446, 0.605],std=[0.169, 0.190, 0.155])
trans = [
    transforms.ToTensor(),
    norm,
]
model.preproc_func = transforms.Compose(trans)

wsi_ioconfig = IOSegmentorConfig(
    input_resolutions=[{"units": "mpp", "resolution": 0.5}],
    patch_input_shape=[224, 224],
    output_resolutions=[{"units": "mpp", "resolution": 0.5}],
    patch_output_shape=[224, 224],
    stride_shape=[224, 224],
)
```
当我们创建DeepFeatureExtractor时，我们将传递auto generate mask=True参数。这将使用otsu阈值自动创建组织区域的掩码，以便提取器只处理那些包含组织的斑块。  
```
# create the feature extractor and run it on the WSI
extractor = DeepFeatureExtractor(model=model, auto_generate_mask=True, batch_size=32, num_loader_workers=4, num_postproc_workers=4)
with suppress_console_output():
    out = extractor.predict(imgs=[wsi_path], mode="wsi", ioconfig=wsi_ioconfig, save_dir=global_save_dir / "wsi_features",)
```
这些特征可以用来训练下游模型，但在这里，为了直观地了解特征代表什么，我们将使用UMAP约简来可视化RGB空间中的特征。标记为相似颜色的点应该具有相似的特征，因此当我们在WSI缩略图上覆盖UMAP还原时，我们可以检查这些特征是否自然地分离到不同的组织区域。我们将将其与上面的块级预测图一起绘制，以查看这些特征与以下单元格中的块级预测的比较情况。  
```
# First we define a function to calculate the umap reduction
def umap_reducer(x, dims=3, nns=10):
    """UMAP reduction of the input data."""
    reducer = umap.UMAP(n_neighbors=nns, n_components=dims, metric="manhattan", spread=0.5, random_state=2)
    reduced = reducer.fit_transform(x)
    reduced -= reduced.min(axis=0)
    reduced /= reduced.max(axis=0)
    return reduced

# load the features output by our feature extractor
pos = np.load(global_save_dir / "wsi_features" / "0.position.npy")
feats = np.load(global_save_dir / "wsi_features" / "0.features.0.npy")
pos = pos / 8 # as we extracted at 0.5mpp, and we are overlaying on a thumbnail at 4mpp

# reduce the features into 3 dimensional (rgb) space
reduced = umap_reducer(feats)

# plot the prediction map the classifier again
overlay = overlay_prediction_mask(
    wsi_overview,
    pred_map,
    alpha=0.5,
    label_info=label_color_dict,
    return_ax=True,
)

# plot the feature map reduction
plt.figure()
plt.imshow(wsi_overview)
plt.scatter(pos[:,0], pos[:,1], c=reduced, s=1, alpha=0.5)
plt.axis("off")
plt.title("UMAP reduction of HistoEnc features")
plt.show()
```
<img src='https://pytorch.org/tutorials/_images/tiatoolbox_tutorial_004.png
' width=20% />
<img src='https://pytorch.org/tutorials/_images/tiatoolbox_tutorial_005.png
' width=20% />  
我们看到来自我们的斑块级预测器的预测图和来自我们的自监督特征编码器的特征图，捕获了关于WSI中组织类型的相似信息。这是一个很好的完整性检查，我们的模型是否按预期工作。这也表明，由HistoEncoder模型提取的特征捕获了组织类型之间的差异，因此它们编码了组织学相关的信息。
注意事项: 

