import csv
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchdata.datapipes.iter import IterableWrapper, ShardingFilter, Mapper
from torchvision.transforms.functional import to_tensor
from PIL import Image
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def split_csv(csv_path, split_ratio=0.8, seed=42):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    random.seed(seed)
    random.shuffle(samples)

    split_point = int(len(samples) * split_ratio)
    return samples[:split_point], samples[split_point:]


def make_trimap_distance(
    alpha_np,
    fg_thresh=0.95,
    bg_thresh=0.05,
    unknown_radius=20
):
    """
    用距离变换（Distance Transform）生成 Trimap，使得 Unknown 带尽量宽。
    
    Args:
        alpha_np:       HxW 的 float32 numpy 数组，取值范围在 [0,1]，代表 GT alpha。
        fg_thresh:      阈值，大于它的像素视为前景；默认 0.95。
        bg_thresh:      阈值，小于它的像素视为背景；默认 0.05。
        unknown_radius: 控制 Unknown 带的“宽度”，单位是像素。这个值越大，
                        Unknown 区越宽；如果想要一个很宽的灰带，可以把它设得大一些。

    Returns:
        trimap: HxW 的 uint8 数组，只含 {0,128,255} 三种值，
                分别表示 {BG, Unknown, FG}。
    """
    h, w = alpha_np.shape

    # —— Step 1: 根据阈值得到最初的二值 FG / BG
    fg_bin = (alpha_np > fg_thresh).astype(np.uint8)   # 前景二值图 (0/1)
    bg_bin = (alpha_np < bg_thresh).astype(np.uint8)   # 背景二值图 (0/1)

    # —— Step 2: 对 “非 FG” 做距离变换，得到每个像素到最初 FG 区域的欧氏距离
    # distanceTransform 要求输入是 0/非0 (uint8)，所以用 1 - fg_bin，让 FG 区为 0，其它位置为 1
    dist_to_fg = cv2.distanceTransform(1 - fg_bin, cv2.DIST_L2, 5)

    # —— Step 3: 对 “非 BG” 做距离变换，得到每个像素到最初 BG 区域的欧氏距离
    dist_to_bg = cv2.distanceTransform(1 - bg_bin, cv2.DIST_L2, 5)

    # —— Step 4: 构造 trimap，先全部设为 Unknown (128)
    trimap = np.full((h, w), 128, dtype=np.uint8)

    # —— Step 5: 距离 > unknown_radius 的像素算“最确定 FG(255)”或“最确定 BG(0)”
    #     —— 如果离最初 BG 区距离超过 unknown_radius，说明离 BG 较远，可以认定为 FG
    trimap[dist_to_bg > unknown_radius] = 255
    #     —— 如果离最初 FG 区距离超过 unknown_radius，说明离 FG 较远，可以认定为 BG
    trimap[dist_to_fg > unknown_radius] = 0

    return trimap


def resize_tensor(tensor, size, mode):
    return F.interpolate(
        tensor.unsqueeze(0), size=size, mode=mode, align_corners=False
    ).squeeze(0)


def build_iterable_datapipe(
    sample_list,
    resize_to=(736, 1280),
    fg_thresh=0.95,
    bg_thresh=0.05,
    unknown_radius=20,
    shuffle=True,
    seed=42,
    do_crop=False,
    crop_size=(512, 512)
):
    """
    构造一个 IterableDataPipe，并在其中对每个样本生成“距离变换版”的 Trimap。
    
    Returns: DataPipe，yield (rgb_tensor, init_mask_tensor, gt_tensor, trimap_tensor)
    """
    if shuffle:
        rng = np.random.default_rng(seed)
        sample_list = rng.permutation(sample_list).tolist()

    pipe = IterableWrapper(sample_list)
    pipe = ShardingFilter(pipe)  # 在多 worker 下做分片

    def load_resize_tensorize(sample):
        # ------ 1. 读取 RGB、初始 Mask、GT alpha
        rgb = to_tensor(Image.open(sample['rgb']).convert('RGB'))      # [3, H, W]
        mask = to_tensor(Image.open(sample['init_mask']).convert('L')) # [1, H, W]
        gt   = to_tensor(Image.open(sample['gt']).convert('L'))        # [1, H, W]

        # ------ 2. Resize 到固定大小
        th, tw = resize_to
        rgb = resize_tensor(rgb, (th, tw), mode='bilinear')
        mask = resize_tensor(mask, (th, tw), mode='bilinear')
        gt   = resize_tensor(gt,   (th, tw), mode='bilinear')

        # ------ 3. 可选地随机 Crop
        if do_crop:
            ch, cw = crop_size
            _, H, W = rgb.shape
            top  = random.randint(0, max(0, H - ch))
            left = random.randint(0, max(0, W - cw))
            rgb  = rgb[:,  top:top+ch,   left:left+cw]
            mask = mask[:, top:top+ch,   left:left+cw]
            gt   = gt[:,   top:top+ch,   left:left+cw]
            
        rgb = TF.normalize(rgb, IMAGENET_MEAN, IMAGENET_STD)
        # ------ 4. 用距离变换生成 Trimap
        #    首先把 GT alpha 转成 numpy 数组，方便做距离变换
        gt_np = gt.squeeze(0).cpu().numpy().astype(np.float32)  # [H, W]，float32，值在 [0,1]
        trimap_np = make_trimap_distance(
            alpha_np=gt_np,
            fg_thresh=fg_thresh,
            bg_thresh=bg_thresh,
            unknown_radius=unknown_radius
        )
        # 再转回 [1, H, W] 的 Tensor，值归一化到 [0,1]
        trimap = torch.from_numpy(trimap_np).unsqueeze(0).float() / 255.0

        return rgb, mask, gt, trimap

    return Mapper(pipe, load_resize_tensorize)


