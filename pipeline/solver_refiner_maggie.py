#!/usr/bin/env python
import os
import sys
import glob
import re
import gc
import itertools

import math
import numpy as np
import pywt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.distributed as dist
import torchvision.utils as vutils

import kornia
from kornia.losses import SSIMLoss
from pytorch_wavelets import DWTForward           
from torchvision.transforms.functional import gaussian_blur

# 开启异常检测，以便在反向传播时立即定位第一个 NaN/Inf 的来源
torch.autograd.set_detect_anomaly(True)

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
from model.refiner_dwt_maggie import XNetDeep

# --------------------
# 配置项
# --------------------
cfg = {
    'num_epochs':       20,
    'transition_epochs': 2,    # epochs 11-12 => length 2
    'batch_size':       1,
    'accum_steps':      4,
    'print_interval':   50,
    'checkpoint_dir':   'checkpoints_xnet_v2',
    'log_dir':          'log_xnet_v2',
    'csv_path':         '../data/pair_for_refiner.csv',
    'vis_dir':          'vis_xnet_aux1loss',
    'low_res':          (360, 640),
    'high_res':         (720, 1280),
    'num_workers':      6,
    'lr':               5e-5,
    'weight_decay':     1e-5,
    'seed':             42,
    'lambda_gate':      5e-4,
    'lambda_fill':      1.0,
    'lambda_art':       1.0,
    'lambda_feather':   0.2,
    'alpha_mode': 'auto',
    'alpha_k': 1.0,
    'coarse_epochs': 4,
    'gate_start_epoch':  11,
}
train_loss_json_path = os.path.join(cfg['log_dir'], 'train_loss.jsonl')
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
loss_history = []
EPS = 1e-6  # 全局小常数，用于 clamp、除以 0 保护等

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------
# （一）“Safe” 计算函数合集
# --------------------

def safe_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor = None,
    eps: float = EPS
) -> torch.Tensor:
    """
    更鲁棒的 BCE-with-logits，用于 target ∈ [0,1] 连续值。
    logits: 网络输出（未经过 sigmoid），shape=[B,1,H,W]
    target: 连续 gt，shape=[B,1,H,W]
    weight_map: 像素级权重图，shape 同 target。如果 sum(weight_map)<eps 或含 NaN，则替换成全 eps。
    eps: 防止 log(0) 或除 0 的小常数。
    """
    # 形状不匹配：返回 0
    if logits.shape != target.shape:
        return torch.tensor(0.0, device=logits.device)

    tgt = target.clamp(eps, 1.0 - eps)

    if weight_map is not None:
        wm = weight_map.clone().float()
        if (not torch.isfinite(wm).all()) or wm.sum() < eps or wm.shape != tgt.shape:
            wm = torch.full_like(tgt, eps)
        weight_map = wm
    else:
        weight_map = None

    loss = F.binary_cross_entropy_with_logits(logits, tgt, weight=weight_map, reduction='mean')
    return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)


def safe_mse(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = EPS
) -> torch.Tensor:
    """
    更鲁棒的 MSE，用于 target ∈ [0,1] 连续值。
    logits: 网络输出（未经过 sigmoid），shape=[B,1,H,W]
    target: 连续 gt，shape=[B,1,H,W]
    mask: 区域掩码 (0/1)，shape 同 target。如果 sum(mask)<eps 或形状不匹配，则替换为全 eps张量。
    eps: 防止除以 0 的小常数。
    """
    if logits.shape != target.shape:
        return torch.tensor(0.0, device=logits.device)

    pred = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    tgt  = target.clamp(eps, 1.0 - eps)

    if mask is None:
        loss = F.mse_loss(pred, tgt, reduction='mean')
    else:
        m = mask.clone().float()
        if (not torch.isfinite(m).all()) or m.sum() < eps or m.shape != tgt.shape:
            m = torch.full_like(tgt, eps)
        area = m.sum().clamp(min=eps)
        diff2 = (pred - tgt).pow(2) * m
        loss = diff2.sum() / area

    return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)


def safe_ssim_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    mask: torch.Tensor = None,
    eps: float = EPS
) -> torch.Tensor:
    """
    更鲁棒的 SSIM Loss (1 - SSIM)，用于 target ∈ [0,1] 连续值。
    logits: 网络输出（未经过 sigmoid），shape=[B,1,H,W]
    target: 连续 gt，shape=[B,1,H,W]
    window_size: SSIM 滑动窗口大小
    mask: 可选区域掩码 (0/1)，shape 同 target。如果 sum(mask)<eps 或形状不匹配，则替换为全 eps。
    eps: 防止除以 0 的小常数。
    """
    if logits.shape != target.shape:
        return torch.tensor(0.0, device=logits.device)
    B, C, H, W = logits.shape
    assert C == 1, "safe_ssim_loss 仅支持单通道"

    pred = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    tgt  = target.clamp(eps, 1.0 - eps)

    if mask is not None:
        m = mask.clone().float()
        if (not torch.isfinite(m).all()) or m.sum() < eps or m.shape != tgt.shape:
            m = torch.full_like(tgt, eps)
    else:
        m = None

    # 构造 2D 高斯窗
    def _gaussian(win_size, sigma):
        coords = torch.arange(win_size, device=pred.device).float() - win_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return (g / g.sum()).to(pred.device)

    _1d = _gaussian(window_size, sigma=1.5).unsqueeze(1)  # [win,1]
    _2d = _1d @ _1d.t()                                   # [win,win]
    window = _2d.expand(C, 1, window_size, window_size).contiguous()
    window = window.to(pred.dtype)

    pad = window_size // 2
    mu_x = F.conv2d(pred, window, padding=pad, groups=C)
    mu_y = F.conv2d(tgt,  window, padding=pad, groups=C)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy   = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(tgt  * tgt, window, padding=pad, groups=C) - mu_y_sq
    sigma_xy   = F.conv2d(pred * tgt, window, padding=pad, groups=C) - mu_xy

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    sigma_x_sq = sigma_x_sq.clamp(min=eps)
    sigma_y_sq = sigma_y_sq.clamp(min=eps)

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    den = den + eps  # 防止除零

    ssim_map = num / den
    ssim_map = torch.clamp(ssim_map, min=0.0, max=1.0)
    ssim_map = torch.nan_to_num(ssim_map, nan=1.0, posinf=1.0, neginf=0.0)

    if m is None:
        ssim_per_img = ssim_map.view(B, -1).mean(dim=1)  # [B]
    else:
        weighted = (ssim_map * m).view(B, -1).sum(dim=1)
        area = m.sum().clamp(min=eps)
        ssim_per_img = weighted / area  # [B]

    loss = 1.0 - ssim_per_img.mean()
    return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)


def safe_edge_map(x: torch.Tensor) -> torch.Tensor:
    """
    计算 Sobel 边缘幅值，并确保中间不会有 NaN/Inf：
    1) clamp(x, 0, 1)
    2) conv2d → gx, gy → nan_to_num(0)
    3) val = gx^2 + gy^2 → clamp(min=0) → sqrt
    4) nan_to_num
    最终返回 [B,H,W]
    """
    x_clamped = x.clamp(0.0, 1.0)
    B, C, H, W = x_clamped.shape
    if C > 1:
        x_gray = x_clamped.mean(dim=1, keepdim=True)
    else:
        x_gray = x_clamped

    sobel_kernel_x = torch.tensor(
        [[[-1., 0., 1.],
           [-2., 0., 2.],
           [-1., 0., 1.]]], device=x_gray.device
    ).unsqueeze(0)  # [1,1,3,3]
    sobel_kernel_y = torch.tensor(
        [[[-1., -2., -1.],
           [ 0.,  0.,  0.],
           [ 1.,  2.,  1.]]], device=x_gray.device
    ).unsqueeze(0)  # [1,1,3,3]

    gx = F.conv2d(x_gray, sobel_kernel_x, padding=1)
    gy = F.conv2d(x_gray, sobel_kernel_y, padding=1)

    gx = torch.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
    gy = torch.nan_to_num(gy, nan=0.0, posinf=0.0, neginf=0.0)

    val = gx.pow(2) + gy.pow(2)
    val = val.clamp(min=0.0)

    mag = torch.sqrt(val)
    mag = torch.nan_to_num(mag, nan=0.0, posinf=1e3, neginf=0.0)

    return mag.squeeze(1)  # [B,H,W]


def safe_wavelet_loss(mu: torch.Tensor, gt: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    计算多尺度 DWT 下的低频 & 高频 L1 损失。
    mu, gt: shape=[B,1,H,W]
    1) 如果形状不一致或含 NaN/Inf，返回 (0,0)
    2) wave1, wave2 分别计算低/高频
    3) lf = lf1 + lf2; hf = 平均所有高频分量
    返回 (lf, hf)，均做 nan_to_num
    """
    if mu.shape != gt.shape or (not torch.isfinite(mu).all()) or (not torch.isfinite(gt).all()):
        zero = torch.tensor(0.0, device=mu.device)
        return zero, zero

    try:
        low_mu1, highs_mu1 = wave1(mu)
        low_gt1, highs_gt1 = wave1(gt)
        low_mu2, highs_mu2 = wave2(mu)
        low_gt2, highs_gt2 = wave2(gt)

        lf1 = F.l1_loss(low_mu1, low_gt1)
        lf2 = F.l1_loss(low_mu2, low_gt2)
        lf = lf1 + lf2

        hf_list = []
        # highs_mu1, highs_mu2 是列表，需要把所有分量合并
        for hmu, hgt in zip(highs_mu1 + highs_mu2, highs_gt1 + highs_gt2):
            hf_list.append(F.l1_loss(hmu, hgt))
        if len(hf_list) > 0:
            hf = sum(hf_list) / len(hf_list)
        else:
            hf = torch.tensor(0.0, device=mu.device)

        lf = torch.nan_to_num(lf, nan=0.0, posinf=None, neginf=None)
        hf = torch.nan_to_num(hf, nan=0.0, posinf=None, neginf=None)
        return lf, hf

    except Exception:
        zero = torch.tensor(0.0, device=mu.device)
        return zero, zero


def safe_guided_structure_loss(
    gt: torch.Tensor,
    guided: torch.Tensor,
    threshold: float = 0.15
) -> torch.Tensor:
    """
    guided_structure_loss = mean( |gt - guided| * soft_mask )，
    soft_mask = clamp((|gt - guided|-threshold)/(1-threshold), 0,1)
    1) If shape mismatch or contains NaN/Inf, return 0
    2) clamp inputs to [0,1]
    3) compute residual & soft_mask
    4) nan_to_num
    """
    if gt.shape != guided.shape or (not torch.isfinite(gt).all()) or (not torch.isfinite(guided).all()):
        return torch.tensor(0.0, device=gt.device)

    g = guided.clamp(0.0, 1.0)
    t = gt.clamp(0.0, 1.0)

    residual = (t - g).abs()
    soft_mask = torch.clamp((residual - threshold) / (1 - threshold), min=0.0, max=1.0)
    loss_val = (residual * soft_mask).mean()
    return torch.nan_to_num(loss_val, nan=0.0, posinf=None, neginf=None)


def compute_safe_losses_at_scale(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    *,
    scale_factor: float = None,
    size: tuple = None,
    weight_map: torch.Tensor = None,
    mask:       torch.Tensor = None,
    window_size: int = 11,
    eps: float = EPS
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    在 pred/gt 的某个缩放分辨率下，返回 (BCE, MSE, SSIM) 三种损失。
    1) If pred.shape != gt.shape: return (0,0,0)
    2) 根据 scale_factor 或 size 缩放 pred/gt
    3) 同步缩放 weight_map/mask
    4) 分别调用 safe_bce_with_logits, safe_mse, safe_ssim_loss
    """
    if pred.shape != gt.shape:
        zero = torch.tensor(0.0, device=pred.device)
        return zero, zero, zero

    if scale_factor is not None:
        p_scaled = F.interpolate(pred, scale_factor=scale_factor,
                                 mode='bilinear', align_corners=False)
        g_scaled = F.interpolate(gt,   scale_factor=scale_factor,
                                 mode='bilinear', align_corners=False)
    elif size is not None:
        p_scaled = F.interpolate(pred, size=size,
                                 mode='bilinear', align_corners=False)
        g_scaled = F.interpolate(gt,   size=size,
                                 mode='bilinear', align_corners=False)
    else:
        p_scaled = pred
        g_scaled = gt

    if weight_map is not None:
        w_map = weight_map.clone()
        if w_map.shape != g_scaled.shape:
            w_map = F.interpolate(w_map, size=g_scaled.shape[-2:],
                                  mode='bilinear', align_corners=False)
    else:
        w_map = None

    if mask is not None:
        m = mask.clone()
        if m.shape != g_scaled.shape:
            m = F.interpolate(m, size=g_scaled.shape[-2:],
                              mode='bilinear', align_corners=False)
    else:
        m = None

    bce_loss  = safe_bce_with_logits(logits=p_scaled, target=g_scaled, weight_map=w_map, eps=eps)
    mse_loss  = safe_mse(logits=p_scaled, target=g_scaled, mask=m, eps=eps)
    ssim_loss = safe_ssim_loss(logits=p_scaled, target=g_scaled, window_size=window_size, mask=m, eps=eps)

    return bce_loss, mse_loss, ssim_loss

# --------------------
# （二） 其余辅助函数（保持原样或稍作 nan_to_num 处理）
# --------------------

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_training_phase(epoch, cfg):
    w = cfg['coarse_epochs']
    g = cfg['gate_start_epoch']
    if epoch < w:
        return 'coarse'
    elif epoch < g:
        return 'full'
    else:
        return 'gate'


def set_gate_grad(model, requires_grad: bool):
    m = model.module if hasattr(model, 'module') else model
    for p in m.res_gate.parameters():
        p.requires_grad = requires_grad
    m.alpha.requires_grad = requires_grad


def set_trunk_aux_grad(model, requires_grad: bool):
    m = model.module if hasattr(model, 'module') else model
    for p in m.trunk_aux.parameters():
        p.requires_grad = requires_grad


def save_visualization(rgb, init_mask, gt, trimap, outputs, loss_curve, save_path):
    orig = rgb.shape[-2:]
    def to_np(x):
        x_t = x.detach().cpu()
        if x_t.dim() == 3:  # (C,H,W)→(1,C,H,W)
            x_t = x_t.unsqueeze(0)
        x_t = x_t.mean(dim=1, keepdim=True)  # 单通道
        x_t = F.interpolate(x_t, size=orig, mode='bilinear', align_corners=False)
        return x_t[0,0].numpy()

    def denormalize_rgb(x_norm):
        mean = IMAGENET_MEAN.to(x_norm.device)
        std = IMAGENET_STD.to(x_norm.device)
        x = x_norm.clone().detach()
        x = x * std + mean
        return x.clamp(0.0, 1.0)

    rgb_denorm = denormalize_rgb(rgb)

    (main_up, aux1_up, aux2_up,
     lf1, hf1, lf2, hf2, a1_cb, b1_cb, fused,
     trunk_up, trunk_aux_up, g) = outputs

    fused_vis = fused.mean(dim=1, keepdim=True)
    im_list = [
        rgb_denorm[0].permute(1,2,0).cpu().numpy(),  # 0: RGB
        init_mask[0,0].cpu().numpy(),               # 1: Init
        gt[0,0].cpu().numpy(),                      # 2: GT
        trimap[0,0].cpu().numpy(),                  # 3: Trimap
        to_np(main_up),                              # 4: Main
        to_np(aux1_up),                              # 5: Aux1
        to_np(aux2_up),                              # 6: Aux2
        to_np(lf1),                                  # 7: LF1
        to_np(hf1),                                  # 8: HF1
        to_np(lf2),                                  # 9: LF2
        to_np(hf2),                                  #10: HF2
        to_np(a1_cb),                                #11: A1_CB
        to_np(b1_cb),                                #12: B1_CB
        to_np(fused_vis),                            #13: Fused
        to_np(trunk_up),                             #14: Trunk
        to_np(trunk_aux_up),                         #15: Trunk_aux_up
        to_np(g),                                    #16: Gate
        to_np(g * (init_mask - trunk_up))            #17: g*res
    ]
    titles = ['RGB','Init','GT','Trimap',
              'Main','Aux1','Aux2',
              'LF1','HF1','LF2','HF2',
              'A1_CB','B1_CB',
              'Fused','Trunk','Trunk_aux_up','Gate','g*res']
    assert len(im_list) == len(titles)

    cols = 5
    rows = math.ceil((len(im_list)+1) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axs = axs.flatten()
    for i, (im, title) in enumerate(zip(im_list, titles)):
        axs[i].imshow(im, cmap='gray' if im.ndim==2 else None)
        axs[i].set_title(title)
        axs[i].axis('off')

    axs[len(im_list)].plot(loss_curve[-500:])
    axs[len(im_list)].set_title('Loss')
    axs[len(im_list)].axis('off')

    for j in range(len(im_list)+1, rows*cols):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def get_aux1_weight(epoch, cfg):
    c = cfg['coarse_epochs']
    g = cfg['gate_start_epoch']
    mid = c + (g - c)//2
    if epoch < c:
        return 0.0, 0.0
    if epoch <= mid:
        frac = float(epoch - c) / float(max(mid - c, 1))
        w_1_8 = (1.0 - frac) * 0.9
        w_1_2 = 0.1 + frac * (1.0 - 0.1)
        return w_1_8, w_1_2
    return 0.0, 1.0


def get_aux1_coef(epoch, cfg):
    c = cfg['coarse_epochs']
    g = cfg['gate_start_epoch']
    mid = c + (g - c)//2
    if epoch < c:
        return 0.0
    if epoch >= mid:
        return 0.0
    start_coef = 0.1
    end_coef   = 0.3
    frac = float(epoch - c) / float(max(mid - c, 1))
    return start_coef + frac * (end_coef - start_coef)


def compute_warmup_loss(mu, gt, trimap, weights, epoch, cfg):
    """
    计算一个 batch 的各项 Loss（包括 1/8 下的各类安全 Loss，以及 Full/Gate 阶段的各种 Loss）。
    返回 tuple(total, l1, soft_iou, struct, low_freq, high_freq,
                 bce_fg, l1_bg, bce_bg, grad_bg, bce8, mse8, ssim8)
    """
    if (not torch.isfinite(mu).all()) or (not torch.isfinite(gt).all()):
        zero = torch.tensor(0.0, device=mu.device)
        return (zero,)*13

    # Coarse 1/8 分辨率
    p8 = F.interpolate(mu, scale_factor=1/8, mode='bilinear', align_corners=False)
    g8 = F.interpolate(gt, scale_factor=1/8, mode='bilinear', align_corners=False)
    w8 = torch.ones_like(g8)
    w8[g8 > 0.5] = 2.0

    bce8, mse8, ssim8 = compute_safe_losses_at_scale(
        pred=p8, gt=g8, weight_map=w8, mask=None, window_size=11, eps=EPS
    )

    # Full 分辨率
    fg_mask = (gt > 0.1).float()
    bg_mask = (trimap == 0.0).float()
    unk_mask = (trimap == 0.5).float()

    # 4.1 L1 on alpha
    weight_l1_map = fg_mask * weights['fg_weight_l1'] + (bg_mask + unk_mask)*weights['bg_weight_l1']
    l1_map = torch.abs(mu - gt) * weight_l1_map
    l1 = torch.nan_to_num(l1_map.mean(), nan=0.0, posinf=None, neginf=None)

    # 4.2 Soft-IoU
    inter = (mu * gt).sum(dim=[1,2,3])
    uni = (mu + gt - mu*gt).sum(dim=[1,2,3]).clamp(min=EPS)
    soft_iou = torch.nan_to_num((1.0 - inter/uni).mean(), nan=0.0, posinf=None, neginf=None)

    # 4.3 Guided Structure
    struct = safe_guided_structure_loss(gt.squeeze(1), mu.squeeze(1))

    # 4.4 Wavelet
    lf, hf = safe_wavelet_loss(mu, gt)
    low_freq = lf
    high_freq = hf * weights['unk_weight']

    # 4.6 Masked-BCE(前景)
    mu_cl = mu.clamp(EPS, 1.0 - EPS)
    logit = torch.log(mu_cl / (1.0 - mu_cl))
    logit = torch.nan_to_num(logit, nan=0.0, posinf=10.0, neginf=-10.0)
    gt_cl = gt.clamp(0.05, 0.95)
    bce_map_fg = F.binary_cross_entropy_with_logits(logit, gt_cl, reduction='none')
    sum_fg = fg_mask.sum().clamp(min=EPS)
    bce_fg = torch.nan_to_num((bce_map_fg * fg_mask).sum() / sum_fg, nan=0.0, posinf=None, neginf=None)

    # 4.7 Background L1/BCE/Grad
    l1_bg_map = torch.abs(mu - 0.0) * bg_mask
    l1_bg = torch.nan_to_num(l1_bg_map.mean(), nan=0.0, posinf=None, neginf=None)

    bce_map_bg = F.binary_cross_entropy_with_logits(logit, torch.zeros_like(mu), reduction='none')
    sum_bg = bg_mask.sum().clamp(min=EPS)
    bce_bg = torch.nan_to_num((bce_map_bg * bg_mask).sum()/sum_bg, nan=0.0, posinf=None, neginf=None)

    grad_x = torch.abs(mu[:,:,:,1:] - mu[:,:,:,:-1])
    grad_x = F.pad(grad_x, (0,1,0,0), mode='constant', value=0)
    grad_y = torch.abs(mu[:,:,1:,:] - mu[:,:,:-1,:])
    grad_y = F.pad(grad_y, (0,0,0,1), mode='constant', value=0)
    grad_map = grad_x + grad_y
    grad_bg_map = grad_map * bg_mask
    grad_bg = torch.nan_to_num(grad_bg_map.mean(), nan=0.0, posinf=None, neginf=None)

    phase = get_training_phase(epoch, cfg)
    if phase == 'coarse':
        total = weights['bce8_weight']*bce8 + weights['mse8_weight']*mse8 + weights['ssim8_weight']*ssim8
    elif phase == 'full':
        full_part = (
            weights['fg_weight_l1']*l1 +
            weights['soft_iou_weight']*soft_iou +
            weights['structure_weight']*struct +
            weights['wavelet_lf_weight']*low_freq +
            weights['wavelet_hf_weight']*high_freq +
            weights['bce_fg_weight']*bce_fg +
            weights['bg_bce_weight']*bce_bg +
            weights['bg_grad_weight']*grad_bg +
            weights['bg_l1_weight']*l1_bg
        )
        w = cfg['coarse_epochs']; g = cfg['gate_start_epoch']
        progress = (epoch - w) / float(max(g - w, 1))
        alpha_coarse = 1.0 - progress
        alpha_full = progress
        total = alpha_coarse*(weights['bce8_weight']*bce8 + weights['mse8_weight']*mse8 + weights['ssim8_weight']*ssim8) \
                + alpha_full*full_part
    else:  # gate
        full_part = (
            weights['fg_weight_l1']*l1 +
            weights['soft_iou_weight']*soft_iou +
            weights['structure_weight']*struct +
            weights['wavelet_lf_weight']*low_freq +
            weights['wavelet_hf_weight']*high_freq +
            weights['bce_fg_weight']*bce_fg +
            weights['bg_bce_weight']*bce_bg +
            weights['bg_grad_weight']*grad_bg +
            weights['bg_l1_weight']*l1_bg
        )
        total = full_part

    return (
        total,
        l1, soft_iou, struct, low_freq, high_freq,
        bce_fg, l1_bg, bce_bg, grad_bg,
        bce8, mse8, ssim8
    )


def get_stage(epoch):
    if epoch < cfg['coarse_epochs']:
        return 'coarse'
    elif epoch < cfg['gate_start_epoch']:
        return 'full'
    else:
        return 'gate'


def compute_warmup_loss_weights(stage, epoch, cfg):
    coarse_base = cfg['coarse_weights']
    full_base   = cfg['full_weights']
    c = cfg['coarse_epochs']
    g = cfg['gate_start_epoch']

    alpha_coarse = 1.0
    alpha_full   = 0.0
    if stage == 'coarse':
        alpha_coarse = 1.0; alpha_full = 0.0
    elif stage == 'full':
        transition_len = g - c
        mid_epoch = c + transition_len // 2
        if epoch <= mid_epoch:
            prog = float(epoch - c) / float(max(mid_epoch - c, 1))
            alpha_coarse = 1.0 - prog
            alpha_full   = prog
        else:
            alpha_coarse = 0.0; alpha_full = 1.0
    else:
        alpha_coarse = 0.0; alpha_full = 1.0

    weights = {}
    weights['alpha_coarse'] = alpha_coarse
    weights['alpha_full']   = alpha_full
    for k,v in coarse_base.items():
        weights[k] = v * alpha_coarse
    for k,v in full_base.items():
        weights[k] = v * alpha_full
    return weights


# --------------------
# （三） 训练与验证流程
# --------------------

def train_one_epoch(epoch, loader, optimizer, scaler, stage, weights, mode='low'):
    model.train()
    optimizer.zero_grad()
    use_gate = model.module.use_gate

    for step, batch in enumerate(loader):
        if step == 0:
            print(f"Total Step: {len(loader)} \n")
        rgb, init_mask, gt, trimap = [x.to(device) for x in batch]

        # 检查输入数据 NaN/Inf
        if not (torch.isfinite(rgb).all() and torch.isfinite(init_mask).all()
                and torch.isfinite(gt).all() and torch.isfinite(trimap).all()):
            print(f"[Warning] 数据异常：出现 NaN/Inf，跳过本 batch")
            optimizer.zero_grad()
            continue

        with autocast(device_type=device.type):
            outputs = model(rgb, init_mask)
            main_up, aux1_up, aux2_up, *_, trunk_up, trunk_aux_up, g = outputs
            aux1_up = torch.sigmoid(aux1_up)

        total, l1, soft_iou, struct, low_freq, high_freq, \
        bce_fg, l1_bg, bce_bg, grad_bg, bce8, mse8, ssim8 = compute_warmup_loss(
            main_up, gt, trimap, weights, epoch, cfg
        )

        total_aux1 = 0.0
        if stage == 'full' and epoch < (cfg['coarse_epochs'] + (cfg['gate_start_epoch'] - cfg['coarse_epochs']) // 2):
            weights_aux1 = {
                'bce8_weight': weights['bce8_weight'],
                'mse8_weight': weights['mse8_weight'],
                'ssim8_weight': weights['ssim8_weight'],
                'fg_weight_l1': 0.0, 'bg_weight_l1': 0.0,
                'soft_iou_weight': 0.0, 'structure_weight': 0.0,
                'wavelet_lf_weight': 0.0, 'wavelet_hf_weight': 0.0,
                'edge_weight': 0.0, 'bce_fg_weight': 0.0,
                'bg_l1_weight': 0.0, 'bg_bce_weight': 0.0,
                'bg_grad_weight': 0.0, 'unk_weight': 0.0,
            }
            total_aux1_1_8, *_ = compute_warmup_loss(aux1_up, gt, trimap, weights_aux1, epoch, cfg)
            total_aux1_1_8 = torch.nan_to_num(total_aux1_1_8, nan=0.0, posinf=None, neginf=None)

            w_h = torch.ones_like(gt)
            w_h[gt > 0.5] = 2.0

            bce_aux1, mse_aux1, ssim_aux1 = compute_safe_losses_at_scale(
                pred=aux1_up, gt=gt, weight_map=w_h, window_size=11, eps=EPS
            )
            total_aux1_1_2 = (
                weights_aux1['bce8_weight'] * bce_aux1 +
                weights_aux1['mse8_weight'] * mse_aux1 +
                weights_aux1['ssim8_weight'] * ssim_aux1
            )
            total_aux1_1_2 = torch.nan_to_num(total_aux1_1_2, nan=0.0, posinf=None, neginf=None)
        else:
            total_aux1, *_ = compute_warmup_loss(aux1_up, gt, trimap, weights, epoch, cfg)
            total_aux1_1_8 = total_aux1_1_2 = total_aux1

        w_1_8, w_1_2 = get_aux1_weight(epoch, cfg)
        total_aux1 = w_1_8 * total_aux1_1_8 + w_1_2 * total_aux1_1_2
        aux1_coef = get_aux1_coef(epoch, cfg)

        if not torch.isfinite(total):
            print(f"[Warning] Epoch{epoch} Step{step}: 主 loss 非有限值，跳过本次更新")
            optimizer.zero_grad()
            continue

        if not torch.isfinite(total_aux1):
            total_aux1 = torch.nan_to_num(total_aux1, nan=0.0, posinf=1e3, neginf=-1e3)

        loss = (total + aux1_coef * total_aux1) / cfg['accum_steps']
        loss_history.append(loss.mean().item())

        scaled_loss = scaler.scale(loss)
        try:
            scaled_loss.backward()
        except RuntimeError as e:
            print(f"[Error] AMP 反向时出错（Epoch{epoch} Step{step}），跳过本次更新：{e}")
            optimizer.zero_grad()
            scaler.update()
            continue

        is_final_step = (step + 1) % cfg['accum_steps'] == 0
        if is_final_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            for p in model.parameters():
                p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e3, neginf=-1e3)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if (step + 1) % cfg['print_interval'] == 0:
            print(f"[E{epoch}-{mode}] Step{step}: Loss={(loss.item() * cfg['accum_steps']):.4f} \
                Gate alpha={model.module.alpha.item():.3f}")
            print('[debug] g max=', g.max().item(),
                  ' delta max=', (model.module.alpha * g * (init_mask - trunk_up)).max().item())

            print(f"[E{epoch}-{mode}] Step{step}: Loss={(loss.item() * cfg['accum_steps']):.4f}  |  total_aux1={(total_aux1.item() * 0.4 * cfg['accum_steps']):.4f}")
            print(f"Main | BCE8={bce8.item():.4f}  |  MSE8={mse8.item():.4f}  |  SSIM8={ssim8.item():.4f}")
            print(f"Aux1(1/8)={(total_aux1_1_8.item() * w_1_8 * 0.4):.5f}  "
                  f"Aux1(1/2)={(total_aux1_1_2.item() * w_1_2 * 0.4):.5f} "
                  f"Aux1(1/8) weight = {w_1_8} "
                  f"Aux1(1/2) weight = {w_1_2}")

        if (step + 1) % (cfg['print_interval'] * 2) == 0:
            save_visualization(
                rgb, init_mask, gt, trimap, outputs, loss_history,
                os.path.join(cfg['vis_dir'], f"e{epoch}_{mode}_{step}.png")
            )


def run_validation(model, eval_loader, device, epoch, vis_dir, max_images=6):
    """
    在高分裁剪验证集上跑可视化，并保存对比图。
    """
    model.eval()
    os.makedirs(vis_dir, exist_ok=True)
    cnt = 0

    with torch.no_grad():
        for batch in eval_loader:
            x, init_mask, gt, _trimap = [t.to(device) for t in batch]
            pred, *_ = model(x, init_mask)

            grid = vutils.make_grid(
                torch.cat([
                    x,
                    init_mask.repeat(1, 3, 1, 1),
                    gt.repeat(1, 3, 1, 1),
                    pred.repeat(1, 3, 1, 1),
                ], dim=0),
                nrow=4,
                normalize=True,
                scale_each=True
            )

            save_path = os.path.join(vis_dir, f"eval_epoch{epoch}_{cnt}.png")
            print(f"[val] saving {save_path}")
            vutils.save_image(grid, save_path)

            cnt += 1
            if cnt >= max_images:
                break

    model.train()


# --------------------
# 主训练流程
# --------------------
if __name__ == '__main__':

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    os.makedirs(cfg['vis_dir'], exist_ok=True)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)
    open(train_loss_json_path, 'w').close()

    low_loader, high_loader, high_val_loader = None, None, None
    prev_stage = None

    model = XNetDeep(
        base_channels=128,
        window_size=16,
        num_heads=4,
        chunk_size=8
    ).to(device)
    model.apply(init_weights)
    model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['num_epochs'],
        eta_min=cfg['lr'] * 0.1
    )
    scaler = GradScaler()

    # 恢复检查点
    cks = glob.glob(os.path.join(cfg['checkpoint_dir'], 'epoch*.pth'))
    if cks:
        cks.sort(key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)))
        start_epoch = int(re.search(r'epoch(\d+)', cks[-1]).group(1)) + 1
        model.load_state_dict(torch.load(cks[-1], map_location=device))
        print(f"Resume from {cks[-1]}, epoch {start_epoch}")
    else:
        start_epoch = 0
        print("Start training from scratch")

    low_loader, low_val_loader = build_dataloaders(
        cfg['csv_path'], cfg['low_res'], cfg['batch_size'],
        cfg['num_workers'], split_ratio=0.8, seed=cfg['seed'],
        sample_fraction=0.3, do_crop=False
    )

    wave1 = DWTForward(J=1, mode='zero', wave='haar').to(device)
    wave2 = DWTForward(J=2, mode='zero', wave='haar').to(device)

    # Warmup 阶段：关闭 Gate，冻结 trunk_aux
    model.module.use_gate = False
    model.module.use_trunk_aux = False
    set_gate_grad(model, False)
    set_trunk_aux_grad(model, False)

    # warmup 大结构阶段的权重
    cfg['coarse_weights'] = {
        'bce8_weight': 1.0,
        'mse8_weight': 1.0,
        'ssim8_weight': 1.0,
    }
    cfg['full_weights'] = {
        'fg_weight_l1':       6.0,
        'bg_weight_l1':       1.0,
        'soft_iou_weight':    1.0,
        'structure_weight':   2.0,
        'wavelet_lf_weight':  1.5,
        'wavelet_hf_weight':  0.8,
        'edge_weight':        0.5,
        'unk_weight':         3.0,
        'bce_fg_weight':      0.3,
        'bg_l1_weight':       1.0,
        'bg_bce_weight':      1.0,
        'bg_grad_weight':     0.2,
    }

    for epoch in range(start_epoch, cfg['num_epochs']):
        stage = get_stage(epoch)
        weights = compute_warmup_loss_weights(stage, epoch, cfg)

        print(f"Currently in stage: {stage} \n")
        if epoch == cfg['coarse_epochs']:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.3
            print(">>> Entering Full Phase, LR lowered to", optimizer.param_groups[0]['lr'])

        train_one_epoch(epoch, low_loader, optimizer, scaler, stage, weights, mode='low')
        run_validation(model, low_val_loader, device, epoch, cfg['vis_dir'])

        scheduler.step()
        ckpt = os.path.join(cfg['checkpoint_dir'], f'epoch{epoch}.pth')
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")
        gc.collect()
        torch.cuda.empty_cache()

        loss_history.clear()

    print('✅ Training complete!')
