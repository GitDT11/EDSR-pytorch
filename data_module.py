import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import functional as F


class DIV2K_x2(Dataset):
    def __init__(self, root_dir, im_size, scale, transform=None):

        self.root_dir = root_dir
        self.im_size = im_size
        self.scale = scale
        self.transform = transform

        images = []
        labels = []
        for file in os.listdir(self.root_dir + '/img'):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                continue
            images.append(file)
        images.sort()

        for file in os.listdir(self.root_dir + '/label'):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                continue
            labels.append(file)
        labels.sort()

        self.images = images
        self.labels = labels

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_path = os.path.join(self.root_dir + '/img', self.images[idx])
        label_path = os.path.join(self.root_dir + '/label', self.labels[idx])

        img = Image.open(img_path)
        img = img.resize((int(self.im_size / self.scale), int(self.im_size / self.scale)))
        label = Image.open(label_path)
        label = label.resize((int(self.im_size), int(self.im_size)))

        if self.transform:
            img, label = self.transform(img, label)
            # label = self.transform(label)

        return img, label


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label


class RandomVerticalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM), label.transpose(Image.FLIP_TOP_BOTTOM)
        return img, label


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, label):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(img, self.mean, self.std), normalize(label, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(img), F.to_tensor(label)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
            # label = t(label)
        return img, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
