import cv2
import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
import numpy as np


def generate_trimap_from_gt(gt, fg_thresh=0.95, bg_thresh=0.05, dilation=20):
    """
    从 ground truth alpha matte 生成 trimap（0: 背景，0.5: unknown，1: 前景）
    Args:
        gt: (B, 1, H, W) 的 float tensor，范围 [0, 1]
        fg_thresh: 超过该值为前景
        bg_thresh: 低于该值为背景
        dilation: unknown 区域宽度（像素）

    Returns:
        trimap: 同 shape 的 float tensor，值域为 {0.0, 0.5, 1.0}
    """
    fg = (gt >= fg_thresh).float()
    bg = (gt <= bg_thresh).float()
    unknown = 1.0 - fg - bg

    # 对 fg/bg 做膨胀，扩大 unknown 区域
    kernel_size = dilation * 2 + 1
    pad = dilation
    fg_dilated = F.max_pool2d(fg, kernel_size=kernel_size, stride=1, padding=pad)
    bg_dilated = F.max_pool2d(bg, kernel_size=kernel_size, stride=1, padding=pad)

    new_unknown = ((fg_dilated - fg) + (bg_dilated - bg)).clamp(0, 1)

    trimap = fg * 1.0 + new_unknown * 0.5 + bg * 0.0
    return trimap

def extract_unknown_mask(trimap, strategy='auto', threshold=0.1):
    if trimap.ndim == 3:
        trimap = trimap.unsqueeze(1)

    is_float = trimap.dtype in [torch.float32, torch.float64]
    
    if strategy == 'fixed':
        unknown_mask = (trimap == 128).float()
    
    elif strategy == 'range':
        if is_float:
            lower, upper = 0.5 - threshold, 0.5 + threshold
        else:
            lower, upper = 128 - threshold, 128 + threshold
        unknown_mask = ((trimap >= lower) & (trimap <= upper)).float()

    elif strategy == 'auto':
        unique_vals = torch.unique(trimap)
        if is_float:
            # 强化：使用 threshold 控制范围，而不是固定0.1/0.9
            lower, upper = threshold, 1.0 - threshold
            unknown_mask = ((trimap > lower) & (trimap < upper)).float()
        elif unique_vals.numel() == 3:
            sorted_vals = unique_vals.sort()[0]
            unknown_val = sorted_vals[1]
            unknown_mask = (trimap == unknown_val).float()
        else:
            unknown_mask = ((trimap > 25) & (trimap < 230)).float()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return unknown_mask


def compute_mse_with_trimap(pred, gt, trimap):
    pred, gt, trimap = pred.float(), gt.float(), trimap.float()
    unknown_mask = extract_unknown_mask(trimap)
    diff = (pred - gt) ** 2 * unknown_mask
    mse_per_image = diff.sum(dim=[1, 2, 3]) / (unknown_mask.sum(dim=[1,2,3]) + 1e-6)
    return mse_per_image.mean()

def compute_sad_with_trimap(pred, gt, trimap, normalize=True):
    pred, gt, trimap = pred.float(), gt.float(), trimap.float()
    unknown_mask = extract_unknown_mask(trimap, strategy='auto')
    diff = torch.abs(pred - gt) * unknown_mask
    sad_per_image = diff.sum(dim=[1, 2, 3])

    if normalize:
        area = unknown_mask.sum(dim=[1, 2, 3])
        sad_per_image = sad_per_image / (area + 1e-6) * 1000.0

    return sad_per_image.mean()

def get_sobel_gradients(x):
    x = x.float()
    kernel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=torch.float32,
        device=x.device
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=torch.float32,
        device=x.device
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    return gx, gy

def largest_connected_component(mask):
    B, _, H, W = mask.shape
    result = torch.zeros_like(mask)

    for i in range(B):
        bin_mask = (mask[i, 0] > 0.5).float().cpu().numpy().astype('uint8')
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        if num_labels <= 1:
            continue
        largest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
        result[i, 0] = torch.from_numpy((labels == largest).astype('float32')).to(mask.device)

    return result

def compute_conn_with_unknown_mask(pred, gt, trimap, threshold=0.5):
    B, C, H, W = pred.shape
    assert C == 1
    pred, gt, trimap = pred.float(), gt.float(), trimap.float()
    unknown_mask = extract_unknown_mask(trimap)
    losses = []

    with torch.no_grad():
        gt_bin = (gt > threshold).float()
        main_region_mask = largest_connected_component(gt_bin)

    for i in range(B):
        gt_main_tensor = main_region_mask[i:i+1]
        eval_mask = (gt_main_tensor * unknown_mask[i:i+1]).clamp(0, 1)

        if eval_mask.sum() < 1:
            losses.append(torch.tensor(0.0, device=pred.device))
            continue

        bce = F.binary_cross_entropy(
            pred[i:i+1] * eval_mask,
            gt_main_tensor * eval_mask,
            reduction='sum'
        )
        norm = eval_mask.sum().clamp(min=1.0)
        losses.append(bce / norm)

    return torch.stack(losses).mean()

def compute_grad_metric(pred_alpha, gt_alpha, trimap, eps=1e-6):
    pred_alpha, gt_alpha = pred_alpha.float(), gt_alpha.float()
    def _grad_xy(x):
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3)
        return F.conv2d(x,kx,padding=1), F.conv2d(x,ky,padding=1)

    pgx, pgy = _grad_xy(pred_alpha)
    ggx, ggy = _grad_xy(gt_alpha)
    unk = extract_unknown_mask(trimap)
    dx = torch.abs(pgx - ggx) * unk
    dy = torch.abs(pgy - ggy) * unk
    area = unk.sum(dim=[1,2,3]) + eps
    per_err = (dx.sum(dim=[1,2,3]) + dy.sum(dim=[1,2,3])) / (2 * area)
    return per_err.mean()

def compute_grad_loss_soft_only(pred_alpha, gt_alpha, coarse_mask=None, trimap=None, use_trimap=False,
                                lower=0.05, upper=0.95):
    pred_alpha, gt_alpha = pred_alpha.float(), gt_alpha.float()

    def _grad(x):
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
        return F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)

    pgx, pgy = _grad(pred_alpha)
    ggx, ggy = _grad(gt_alpha)

    soft = ((gt_alpha > lower) & (gt_alpha < upper)).float()
    if coarse_mask is not None:
        soft *= ((coarse_mask > lower) & (coarse_mask < upper)).float()
    if use_trimap and trimap is not None:
        from utils import extract_unknown_mask
        soft *= extract_unknown_mask(trimap, strategy='auto', threshold=0.1)

    edge = torch.sqrt(ggx**2 + ggy**2 + 1e-6)
    weight = edge / (edge.max() + 1e-6)
    wm = soft * weight

    dx = torch.abs(pgx - ggx) * wm
    dy = torch.abs(pgy - ggy) * wm

    return (dx.sum() + dy.sum()) / (wm.sum() + 1e-6) / 2


def compute_conn_mismatch_loss(pred_alpha, gt_alpha, threshold=0.5):
    pred_alpha, gt_alpha = pred_alpha.float(), gt_alpha.float()
    gt_bin = (gt_alpha > threshold).to(pred_alpha.dtype)
    pred_bin = (pred_alpha > threshold).float()
    mismatch = torch.abs(gt_bin - pred_bin)
    bce = F.binary_cross_entropy(pred_alpha, gt_bin, reduction='none')
    masked = bce * mismatch
    return masked.sum()/(mismatch.sum()+1e-6)

def compute_cf_loss(pred_alpha, gt_alpha, trimap=None, use_trimap=False,
                    lower=0.05, upper=0.95, eps=1e-6):
    pred_alpha, gt_alpha = pred_alpha.float(), gt_alpha.float()
    soft = ((gt_alpha>lower)&(gt_alpha<upper)).float()
    if use_trimap and trimap is not None:
        soft *= extract_unknown_mask(trimap)
    loss_map = (pred_alpha - gt_alpha)**2
    masked = loss_map * soft
    return masked.sum()/(soft.sum()+eps)

def compute_ctr_loss(anchor_feat, positive_feat, pred_mask, gt_mask, margin=0.2, use_triplet=True, debug=False):
    B, C, H, W = anchor_feat.shape

    # Downsample pred and gt to match feature resolution
    pred_mask = F.interpolate(pred_mask, size=(H, W), mode='bilinear', align_corners=False)
    gt_mask = F.interpolate(gt_mask, size=(H, W), mode='bilinear', align_corners=False)

    with torch.no_grad():
        false_pos = (pred_mask > 0.5) & (gt_mask <= 0.5)
        false_neg = (pred_mask <= 0.5) & (gt_mask > 0.5)

    def extract_feat(feat, mask):
        masked_feat = feat * mask.unsqueeze(1)
        return F.adaptive_avg_pool2d(masked_feat, 1).view(B, C)

    anchor_fp = extract_feat(anchor_feat, false_pos.float())
    anchor_fn = extract_feat(anchor_feat, false_neg.float())
    pos_fp = extract_feat(positive_feat, false_pos.float())
    pos_fn = extract_feat(positive_feat, false_neg.float())

    sim_pos_fp = F.cosine_similarity(anchor_fp, pos_fp, dim=1)
    sim_pos_fn = F.cosine_similarity(anchor_fn, pos_fn, dim=1)

    if use_triplet:
        loss_fp = F.relu(1.0 - sim_pos_fp + margin)
        loss_fn = F.relu(1.0 - sim_pos_fn + margin)
    else:
        loss_fp = 1.0 - sim_pos_fp
        loss_fn = 1.0 - sim_pos_fn

    loss = (loss_fp.mean() + loss_fn.mean()) / 2.0

    if debug:
        print(f"[CTR Spatial] sim_fp: {sim_pos_fp.mean().item():.4f}, sim_fn: {sim_pos_fn.mean().item():.4f}, loss: {loss.item():.4f}")

    return loss

# ------------------- Embedding Contrastive Loss ----------------------
def compute_embedding_ctr_loss(anchor_embed, positive_embed, margin=0.2, use_triplet=True, debug=False):
    anchor_embed = F.normalize(anchor_embed, dim=1)
    positive_embed = F.normalize(positive_embed, dim=1)

    sim = F.cosine_similarity(anchor_embed, positive_embed, dim=1)

    if use_triplet:
        loss = F.relu(1.0 - sim + margin).mean()
    else:
        loss = (1 - sim).mean()

    if debug:
        print(f"[Embed CTR] sim: {sim.mean().item():.4f}, loss: {loss.item():.4f}")

    return loss


def compute_sad_loss(pred_alpha, gt_alpha, trimap=None, use_trimap=False, lower=0.1, upper=0.9):
    pred_alpha, gt_alpha = pred_alpha.float(), gt_alpha.float()
    soft = ((gt_alpha > lower) & (gt_alpha < upper)).float()
    if use_trimap and trimap is not None:
        from utils import extract_unknown_mask
        soft *= extract_unknown_mask(trimap, strategy='auto', threshold=0.1)
    absd = torch.abs(pred_alpha - gt_alpha)
    loss = absd * soft
    return loss.sum() / (soft.sum() + 1e-6)

