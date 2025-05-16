import csv
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchdata.datapipes.iter import IterableWrapper, ShardingFilter, Mapper
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
    return F.interpolate(
        tensor.unsqueeze(0), size=size, mode=mode, align_corners=False
    ).squeeze(0)


def make_trimap_from_alpha(alpha_np,
                           fg_thresh=0.99,
                           kernel_size_fg=11,
                           iter_fg=1):
    """
    Generate a dilated foreground-only trimap from GT alpha matte.

    Only dilates the foreground to define the unknown band;
    background is implicitly everything else.

    Args:
        alpha_np: HxW float32 numpy array in [0,1]
        fg_thresh: threshold to binarize GT foreground
        kernel_size_fg: dilation kernel size for foreground
        iter_fg: number of dilation iterations

    Returns:
        HxW uint8 trimap mask: 0=Background, 128=Unknown, 255=Foreground
    """
    # Binary foreground mask
    fg = (alpha_np > fg_thresh).astype(np.uint8)
    # Structuring element
    ker_fg = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size_fg, kernel_size_fg)
    )
    # Dilate FG to get expanded region
    fg_d = fg.copy()
    for _ in range(iter_fg):
        fg_d = cv2.dilate(fg_d, ker_fg)
    # Build trimap: FG_dilated minus FG is unknown
    trimap = np.zeros_like(alpha_np, dtype=np.uint8)
    trimap[fg_d == 1] = 128      # unknown or FG region
    trimap[fg == 1] = 255        # original FG overrides
    # Background remains 0
    return trimap


def build_iterable_datapipe(
    sample_list,
    resize_to=(736, 1280),
    fg_thresh=0.5,
    kernel_size_fg=11,
    iter_fg=5,
    shuffle=True,
    seed=42
):
    """
    Constructs a worker-aware IterableDataPipe with preprocessing and trimap generation.

    Returns DataPipe yielding: (rgb_tensor, init_mask_tensor, gt_tensor, trimap_tensor)
    """
    if shuffle:
        rng = np.random.default_rng(seed)
        sample_list = rng.permutation(sample_list).tolist()

    pipe = IterableWrapper(sample_list)
    pipe = ShardingFilter(pipe)  # multi-worker split

    def load_resize_tensorize(sample):
        # Load and convert
        rgb = to_tensor(Image.open(sample['rgb']).convert('RGB'))    # [3,H,W]
        mask = to_tensor(Image.open(sample['init_mask']).convert('L')) # [1,H,W]
        gt   = to_tensor(Image.open(sample['gt']).convert('L'))        # [1,H,W]

        # Resize
        th, tw = resize_to
        rgb = resize_tensor(rgb, (th, tw), mode='bilinear')
        mask = resize_tensor(mask, (th, tw), mode='bilinear')
        gt   = resize_tensor(gt,   (th, tw), mode='bilinear')

        # Generate trimap from alpha: dilate only FG
        gt_np = gt.squeeze(0).cpu().numpy().astype(np.float32)
        trimap_np = make_trimap_from_alpha(
            gt_np,
            fg_thresh=fg_thresh,
            kernel_size_fg=kernel_size_fg,
            iter_fg=iter_fg
        )
        # to tensor [1,H,W]
        trimap = torch.from_numpy(trimap_np).unsqueeze(0).float() / 255.0

        return rgb, mask, gt, trimap

    return Mapper(pipe, load_resize_tensorize)
