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

def build_iterable_datapipe(sample_list, resize_to=(736, 1280), shuffle=True):
    pipe = IterableWrapper(sample_list)
    if shuffle:
        pipe = Shuffler(pipe)

    def load_resize_tensorize(sample):
        rgb = to_tensor(Image.open(sample['rgb']).convert('RGB'))          # [3, H, W]
        mask = to_tensor(Image.open(sample['init_mask']).convert('L'))     # [1, H, W]
        gt = to_tensor(Image.open(sample['gt']).convert('L'))              # [1, H, W]

        target_h, target_w = resize_to

        rgb = resize_tensor(rgb, (target_h, target_w), mode='bilinear')
        mask = resize_tensor(mask, (target_h, target_w), mode='bilinear')  # or 'nearest' if binary
        gt = resize_tensor(gt, (target_h, target_w), mode='bilinear')

        return rgb, mask, gt

    return Mapper(pipe, load_resize_tensorize)
