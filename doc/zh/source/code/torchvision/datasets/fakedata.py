import torch
import torch.utils.data as data
from .. import transforms


class FakeData(data.Dataset):
    """A fake dataset that returns randomly generated images and returns them as PIL images

    Args:
        size (int, optional): 数据集的大小. 默认: 1000 张图片
        image_size(tuple, optional): 返回图像的尺寸. 默认: (3, 224, 224)
        num_classes(int, optional): 数据集中类别的数量. 默认: 10
        transform (callable, optional): 一个 transform 函数, 它输入 PIL image 并且返回 
        transformed 版本. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): 一个 transform 函数, 输入 target 并且
            转换它.


    """

    def __init__(self, size=1000, image_size=(3, 224, 224), num_classes=10, transform=None, target_transform=None):
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        rng_state = torch.get_rng_state()
        torch.manual_seed(index)
        img = torch.randn(*self.image_size)
        target = torch.Tensor(1).random_(0, self.num_classes)[0]
        torch.set_rng_state(rng_state)

        # convert to PIL Image
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.size
