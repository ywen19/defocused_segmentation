import csv
import random
from torchdata.datapipes.iter import IterableWrapper, Shuffler, Mapper
from torchvision.transforms.functional import to_tensor
from PIL import Image

def split_csv(csv_path, split_ratio=0.8, seed=42):
    """Split csv rows into train and val lists."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    random.seed(seed)
    random.shuffle(samples)

    split_point = int(len(samples) * split_ratio)
    return samples[:split_point], samples[split_point:]

def build_iterable_datapipe(sample_list, resize_to=(720, 1280), shuffle=True):
    """Create a TorchData pipeline from a list of dicts with image paths."""
    pipe = IterableWrapper(sample_list)
    if shuffle:
        pipe = Shuffler(pipe)

    def load_resize_tensorize(sample):
        rgb = Image.open(sample['rgb']).convert('RGB')
        mask = Image.open(sample['init_mask']).convert('L')
        gt = Image.open(sample['gt']).convert('L')

        # Explicitly resize using PIL to (width, height)
        rgb = rgb.resize(resize_to[::-1], resample=Image.BILINEAR)
        mask = mask.resize(resize_to[::-1], resample=Image.BILINEAR)
        gt = gt.resize(resize_to[::-1], resample=Image.BILINEAR)

        return to_tensor(rgb), to_tensor(mask), to_tensor(gt)

    return Mapper(pipe, load_resize_tensorize)
