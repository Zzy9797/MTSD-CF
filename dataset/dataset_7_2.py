
import torch.utils.data as data
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random
import math
from collections import defaultdict


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.path=data_path
        self.images=[]
        self.label=[]

        self.transform = transform
       
        for i in range(len(os.listdir(self.path))):
            self.img_dir=os.path.join(self.path,str(i))

            self.imageslist=os.listdir(self.img_dir)
            for item in self.imageslist:
                self.images.append(os.path.join(self.path,str(i),item))
                self.label.append(i)  


    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        if self.label[idx]==0:
            label_2=torch.tensor(0)
        else:
            label_2=torch.tensor(1)
        
        if self.label[idx]==0:
            label_4=torch.tensor(0)
        elif self.label[idx]==1:
            label_4=torch.tensor(1)
        elif self.label[idx]==3:
            label_4=torch.tensor(2)
        else:
            label_4=torch.tensor(3)
        label = torch.tensor(self.label[idx])
        return img, label,label_2,label_4

    def __len__(self):
        return len(self.images)
    
    def get_labels(self):
        return self.label
    
    # For Class_Balance
    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """

        return (int(self.images[idx].split('/')[-2]), )
    
class ClassBalancedDataset():
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in [1], in each epoch, an image may appear multiple
    times based on its "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids(idx)` to support
    ClassBalancedDataset.
    The repeat factor is computed as followed.
    1. For each category c, compute the fraction # of images
        that contain it: f(c)
    2. For each category c, compute the category-level repeat factor:
        r(c) = max(1, sqrt(t/f(c)))
    3. For each image I and its labels L(I), compute the image-level repeat
    factor:
        r(I) = max_{c in L(I)} r(c)

    References:
        .. [1]  https://arxiv.org/pdf/1908.03195.pdf

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with `f_c` >= `oversample_thr`, there is
            no oversampling. For categories with `f_c` < `oversample_thr`, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
    """

    def __init__(self, dataset, oversample_thr=0.15, method='sqrt', **kwargs):
        assert(method in ('sqrt', 'reciprocal'))    # reciprocal，倒数，绝对 balance， repeat_factor = f(c) / max( fc )
        self.method = method
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.CLASSES = FER_CLASSES = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Happiness', 'Surprise', 'Neutral']

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))    #遍历dataset，查询每个图片的标签
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            assert v > 0, f'caterogy {k} does not contain any images'
            category_freq[k] = v / num_images   #每个类别出现频率

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        if self.method == 'sqrt':
            category_repeat = {
                cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
                for cat_id, cat_freq in category_freq.items()
            }
        elif self.method == 'reciprocal':
            cat_freq_max = 0
            for cat_id, cat_freq in category_freq.items():
                cat_freq_max = max(cat_freq_max, cat_freq)
            print('cat_freq_max: ', cat_freq_max)
            category_repeat = {
                cat_id: cat_freq_max / cat_freq
                for cat_id, cat_freq in category_freq.items()
            }   #每个类别的重复因数
        # 3. For each image I and its labels L(I), compute the image-level
        # repeat factor:
        #    r(I) = max_{c in L(I)} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            repeat_factor = max(
                {category_repeat[cat_id]
                 for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        return len(self.repeat_indices)