import csv
import random
import torch.nn.functional as F
from torchdata.datapipes.iter import IterableWrapper, Shuffler, Mapper
from torchvision.transforms.functional import to_tensor
from PIL import Image

def split_csv(csv_path, split_ratio=0.8, seed=42):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    random.seed(seed)
    random.shuffle(samples)

    split_point = int(len(samples) * split_ratio)
    return samples[:split_point], samples[split_point:]

def resize_tensor(tensor, size, mode):
    return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode, align_corners=False).squeeze(0)

from torchdata.datapipes.iter import IterableWrapper, Shuffler, Mapper, ShardingFilter
from torchvision.transforms.functional import to_tensor
from PIL import Image
import torch.nn.functional as F
import numpy as np

def resize_tensor(tensor, size, mode):
    return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode, align_corners=False).squeeze(0)

def build_iterable_datapipe(sample_list, resize_to=(736, 1280), shuffle=True, seed=42):
    """
    Constructs a worker-aware, optionally shuffled IterableDataPipe with preprocessing.

    Args:
        sample_list (list): List of sample dicts with 'rgb', 'gt', 'init_mask'.
        resize_to (tuple): Target size (H, W) for all images.
        shuffle (bool): Whether to shuffle sample_list.
        seed (int): Random seed for reproducible shuffling.

    Returns:
        DataPipe yielding (rgb_tensor, init_mask_tensor, gt_tensor).
    """
    if shuffle:
        rng = np.random.default_rng(seed)
        sample_list = rng.permutation(sample_list).tolist()

    pipe = IterableWrapper(sample_list)
    pipe = ShardingFilter(pipe)  # Ensures proper multi-worker splitting

    def load_resize_tensorize(sample):
        rgb = to_tensor(Image.open(sample['rgb']).convert('RGB'))          # [3, H, W]
        mask = to_tensor(Image.open(sample['init_mask']).convert('L'))     # [1, H, W]
        gt = to_tensor(Image.open(sample['gt']).convert('L'))              # [1, H, W]

        target_h, target_w = resize_to

        rgb = resize_tensor(rgb, (target_h, target_w), mode='bilinear')
        mask = resize_tensor(mask, (target_h, target_w), mode='bilinear')
        gt = resize_tensor(gt, (target_h, target_w), mode='bilinear')

        return rgb, mask, gt

    return Mapper(pipe, load_resize_tensorize)

