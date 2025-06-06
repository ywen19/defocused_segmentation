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
from pytorch_wavelets import DWTForward           # 你已有
from torchvision.transforms.functional import gaussian_blur

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
    'warmup_epochs':    5,
    'correction_epochs': 10,    # epochs 5-10 inclusive => length 6
    'transition_epochs': 2,    # epochs 11-12 => length 2
    'batch_size':       2,
    'accum_steps':      4,
    'print_interval':   50,
    'checkpoint_dir':   'checkpoints_xnet_v2',
    'log_dir':          'log_xnet_v2',
    'csv_path':         '../data/pair_for_refiner.csv',
    'vis_dir':          'vis_xnet_largerW',
    'low_res':          (368, 640),
    'high_res':         (720, 1280),
    'num_workers':      6,
    'lr':               1e-4,
    'weight_decay':     1e-5,
    'seed':             42,
    'lambda_gate':      1e-4,
    'lambda_fill':      1.0,
    'lambda_art':       1.0,
    'lambda_feather':   0.2,
    'alpha_mode': 'auto',
    'alpha_k': 1.0,
}
train_loss_json_path = os.path.join(cfg['log_dir'], 'train_loss.jsonl')

def get_stage(epoch):
    if   epoch < cfg['warmup_epochs']:
        return 'warmup'
    elif epoch < cfg['warmup_epochs'] + cfg['correction_epochs']:   
        return 'correction'
    elif epoch < cfg['warmup_epochs'] + cfg['correction_epochs'] + cfg['transition_epochs']: 
        return 'transition'
    else:                  
        return 'fine'


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)

def set_gate_grad(model, requires_grad: bool):
    m = model.module if hasattr(model, 'module') else model
    for p in m.res_gate.parameters(): p.requires_grad = requires_grad
    m.alpha.requires_grad = requires_grad

def set_trunk_aux_grad(model, requires_grad: bool):
    """
    将 model.trunk_aux 所有参数的 requires_grad 设为 requires_grad。
    """
    m = model.module if hasattr(model, 'module') else model
    for p in m.trunk_aux.parameters():
        p.requires_grad = requires_grad


# --------------------
# 损失函数定义
# --------------------
ssim_loss = SSIMLoss(window_size=11, reduction='mean')

def soft_iou_loss(pred, target, eps=1e-6):
    inter = (pred * target).sum((1,2,3))
    union = (pred + target - pred*target).sum((1,2,3)) + eps
    return 1.0 - (inter / union).mean()

def tv_loss(x):
    vert = (x[...,1:,:] - x[...,:-1,:]).abs().mean()
    hori = (x[..., :,1:] - x[..., :,:-1]).abs().mean()
    return vert + hori

def dwt_high_freq_loss_torch(pred: torch.Tensor, gt: torch.Tensor):
    dec_lo = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=pred.dtype, device=pred.device)
    dec_hi = torch.tensor([-1/math.sqrt(2), 1/math.sqrt(2)], dtype=pred.dtype, device=pred.device)
    filters = torch.stack([
        dec_lo[:, None] @ dec_lo[None, :],
        dec_lo[:, None] @ dec_hi[None, :],
        dec_hi[:, None] @ dec_lo[None, :],
        dec_hi[:, None] @ dec_hi[None, :],
    ]).unsqueeze(1)
    coeffs_p = F.conv2d(pred, filters, stride=2, padding=0)
    coeffs_g = F.conv2d(gt,   filters, stride=2, padding=0)
    hf_p = coeffs_p[:, 1:4, :, :]
    hf_g = coeffs_g[:, 1:4, :, :]
    return F.l1_loss(hf_p, hf_g)


def wavelet_hf_loss(p: torch.Tensor, g: torch.Tensor):
    """
    p, g:  (B,1,H,W) or (B,3,H,W)
    1. 确保 wavelet 模块在当前 device
    2. 禁用 autocast → 全 FP32，从而 input/weight dtype 一致
    """
    # 如果 dataparallel 换 GPU，需要动态迁移
    if wave1.h0_row.device != p.device:
        wave1.to(p.device)
        wave2.to(p.device)

    with autocast(device_type=p.device.type, enabled=False):             # —— 关键：停用 autocast
        p32, g32 = p.float(), g.float()

        # gt 部分可以 no_grad，节省显存；对 p 仍需梯度
        with torch.no_grad():
            _, gh1 = wave1(g32)
            _, gh2 = wave2(g32)

        _, ph1 = wave1(p32)
        _, ph2 = wave2(p32)

        l1 = sum(F.l1_loss(ph, gh) for ph, gh in zip(ph1, gh1)) / len(ph1)
        l2 = sum(F.l1_loss(ph, gh) for ph, gh in zip(ph2, gh2)) / len(ph2)

    return l1, l2


def fourier_loss(x, y, debug=False):
    # cast to float32 so cuFFT has no shape restrictions
    x_f = x[:, 0].to(torch.float32)
    y_f = y[:, 0].to(torch.float32)

    fx = torch.fft.rfft2(x_f, norm='forward')
    fy = torch.fft.rfft2(y_f, norm='forward')

    B, H, W2 = fx.shape

    # create a [H×W2] mask that zeros out the first row and first column
    mask = torch.ones(H, W2, device=fx.device, dtype=fx.dtype)
    mask[0, :] = 0.0  # zero out all j where i=0
    mask[:, 0] = 0.0  # zero out all i where j=0

    # compute per-batch absolute difference
    diff = torch.abs(fx - fy) * mask  # shape: (B, H, W2)
    loss = diff.mean()

    if debug:
        # 打印整体 loss
        print(f"[Fourier] mean diff over all B×H×W2 = {loss.item():.6f}")
        # 也可以打印每个样本上的差异情况
        per_sample = diff.view(B, -1).mean(dim=1)
        for i, d in enumerate(per_sample):
            print(f"[Fourier] sample {i} mean diff = {d.item():.6f}")
        # 打印频域尺寸，确保正确
        print(f"[Fourier] FFT output shape: B={B}, H={H}, W2={W2}")

    return loss.to(x.dtype)

# --------------------
# Laplacian Pyramid Loss
# --------------------
def laplacian_pyramid_loss(x, y, levels=3, debug=False):
        loss_lp = 0.0
        curr_x, curr_y = x, y
        for i in range(levels):
            gx = F.avg_pool2d(curr_x, 2, 2)
            gy = F.avg_pool2d(curr_y, 2, 2)
            lx = curr_x - F.interpolate(gx, scale_factor=2,
                                        mode='bilinear', align_corners=False)
            ly = curr_y - F.interpolate(gy, scale_factor=2,
                                        mode='bilinear', align_corners=False)
            # 这里计算本层 loss（平均绝对差）
            l = F.l1_loss(lx, ly, reduction='mean')
            if debug:
                # 打印本层的平均误差
                print(f"[Laplacian Level {i}] mean |lx-ly| = {l.item():.6f}")
            loss_lp += l
            curr_x, curr_y = gx, gy
        avg_loss = loss_lp / levels
        if debug:
            print(f"[Laplacian Pyramid] averaged over {levels} levels: {avg_loss.item():.6f}")
        return avg_loss


def gate_losses(g, init_mask, trunk, gt,
                w_pos=0.8, w_neg=0.4,        # L1 权重
                w_iou_pos=0.4, w_iou_neg=0.2,
                lam_reg=cfg['lambda_gate'],
                w_cons=0.1,
                lam_tv=0.02):  # 新增 TV 正则 的权重
    """
    g: (B,1,H,W)  Gate 输出
    init_mask, trunk, gt: (B,1,H,W)
    w_pos, w_neg: 正向/负向 L1 损失权重
    w_iou_pos, w_iou_neg: 正向/负向 Soft-IoU 权重
    lam_reg: Gate L1 正则权重
    w_cons: consistency (trunk vs. gt) 权重
    lam_tv: TV 正则的权重（本次新增）。
    """

    # 1) 计算残差与目标
    pos_res = F.relu(init_mask - trunk)   # 需要“填充”的区域
    neg_res = F.relu(trunk - init_mask)   # 需要“抠掉”的区域
    tgt_pos = F.relu(gt - trunk)          # 真实应该填充的量
    tgt_neg = F.relu(trunk - gt)          # 真实应该抠掉的量

    # 2) 正向/负向 L1 校正
    L_pos = (g * pos_res - tgt_pos).abs().mean()
    L_neg = (g * neg_res - tgt_neg).abs().mean()

    # 3) Soft-IoU (可选)
    def soft_iou(a, b):
        inter = (a * b).sum((1,2,3))
        union = (a + b - a*b).sum((1,2,3)) + 1e-6
        return 1 - (inter / union).mean()

    L_iou_pos = soft_iou(g * pos_res, tgt_pos)
    L_iou_neg = soft_iou(g * neg_res, tgt_neg)

    # 4) 汇总正向/负向损失
    L_gate_raw = (w_pos * L_pos +
                  w_neg * L_neg +
                  w_iou_pos * L_iou_pos +
                  w_iou_neg * L_iou_neg)

    # 5) Gate 本身的 L1 正则（让 g 尽可能接近 0/1，但不全部为 1）
    L_reg = g.abs().mean() * lam_reg

    # 6) Trunk vs. GT 的 consistency（保持 trunk 本身也接近 gt）
    L_cons = (trunk - gt).abs().mean() * w_cons

    # 7) 新增：Total‐Variation (TV) 正则，让 Gate g 在空间上更平滑
    #    TV(g) = 垂直像素差 + 水平像素差 的绝对值之和
    #    你也可以写成一个小函数，这里直接 inline:
    vert = (g[...,1:,:] - g[...,:-1,:]).abs().mean()
    hori = (g[..., :,1:] - g[..., :,:-1]).abs().mean()
    L_tv = (vert + hori) * lam_tv

    # 8) 最终把所有项加起来，作为 Gate 的损失
    L_gate_total = L_gate_raw + L_reg + L_cons + L_tv

    return L_gate_total, L_reg, L_cons


def get_soft_edges(mask):
    lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                       device=mask.device, dtype=mask.dtype).view(1,1,3,3)
    e = torch.abs(F.conv2d(mask, lap, padding=1))
    return e / (e.max() + 1e-6)


def soft_boundary_loss(p, g, sigma=1.0):
    return F.l1_loss(gaussian_blur(p, [5,5], sigma=sigma),
                     gaussian_blur(g, [5,5], sigma=sigma))

def gradient_loss(p, g):
    return (F.l1_loss(p[:,:,:,1:]-p[:,:,:,:-1],
                      g[:,:,:,1:]-g[:,:,:,:-1]) +
            F.l1_loss(p[:,:,1:,:]-p[:,:,:-1,:],
                      g[:,:,1:,:]-g[:,:, :-1,:]))

def compute_dwt_loss(rf, gt, model, wavelet_list, lam_hf=1.0):
    """
    Compute DWT-based high-frequency supervision loss between rf and gt.
    Returns ll_loss and hf_loss.
    """
    coeffs_rf = multi_dwt_mixed(rf, wavelet_list)
    coeffs_gt = multi_dwt_mixed(gt, wavelet_list)

    # 假设 multi_dwt_mixed 最后一项是 (LL, [HF...], name_str)
    ll_rf, hf_rf_list, _ = coeffs_rf[-1]
    ll_gt, hf_gt_list, _ = coeffs_gt[-1]

    # 低频 L1
    ll_loss = F.l1_loss(ll_rf, ll_gt)

    # 高频 L1，确保 zip 里的全是 Tensor
    hf_loss = 0.0
    for p_h, g_h in zip(hf_rf_list, hf_gt_list):
        hf_loss += F.l1_loss(p_h, g_h)
    hf_loss = lam_hf * hf_loss

    return ll_loss, hf_loss


# --------------------
# Compute Total Loss
# --------------------
def compute_total_loss(pred, gt, trimap, epoch, use_gate):
    """
    Warm-up  : 粗结构 BCE+MSE+SSIM @1/8
    Correction: 连续 α + 边缘 + Wavelet-HF (J=1,2) + 1/4 粗尺度 L1 + Soft-IoU（只在前3个 correction epoch）
    Trans/Fine: 加强 HF/Grad, 仅微量 BCE
    """
    # ---------- common ----------
    pred = pred.clamp(0.0, 1.0)
    gt   = gt.clamp(0.0, 1.0).to(pred.dtype).to(pred.device)
    tmap = trimap.to(pred.device)

    # ---------- Warm-up (epoch < warmup) ----------
    if epoch < cfg['warmup_epochs']:
        p8  = F.interpolate(pred, scale_factor=1/8, mode='bilinear', align_corners=False)
        g8  = F.interpolate(gt,   scale_factor=1/8, mode='bilinear', align_corners=False)
        w8  = torch.ones_like(g8)
        w8[g8 > 0.5] = 2.0                               # 前景×2
        b8  = F.binary_cross_entropy_with_logits(p8, g8, weight=w8)
        m8  = F.mse_loss(p8, g8)
        s8  = 1.0 - ssim_loss(p8, g8)
        warm_n = cfg['warmup_epochs']
        # alpha 从 0 → 1
        alpha = float(epoch) / float(max(warm_n - 1, 1))
        # bce_weight 从 1.0 → 2.0
        bce_weight = 1.0 + alpha * (2.0 - 1.0)
        total = bce_weight * b8 + 0.1 * m8 + 0.05 * s8
        return total, {'bce_coarse': b8, 'mse_coarse': m8, 'ssim_coarse': s8}

    # ---------- Correction ----------
    if epoch < cfg['warmup_epochs'] + cfg['correction_epochs']:
        # 基础权重
        w_l1, w_soft  = 1.0, 1.4
        w_grad        = 2.0
        w_w1, w_w2    = 1.0, 1.5
        w_ssim        = 0.2
        w_edge        = 0.3

        # 主 L1 带可学习权重图
        w_map = torch.where(gt > 0.9, torch.ones_like(gt),
                 torch.where(gt < 0.1, 0.3*torch.ones_like(gt),
                             1.5*torch.ones_like(gt)))
        l1    = (F.l1_loss(pred, gt, reduction='none') * w_map).mean()

        lsoft = soft_boundary_loss(pred, gt)
        lgrad = gradient_loss(pred, gt)

        edge_gt = kornia.filters.sobel(gt).abs().sum(1, keepdim=True)  # (B,1,H,W)
        edge_w  = 1.0 + 6.0 * edge_gt                                  # 权重 1~7
        thin_mask = (edge_gt > 0.01) & (edge_gt < 0.2)
        edge_w    = edge_w * (1.0 + 0.3 * thin_mask.float())
        l_edge = (edge_w * (pred - gt).abs()).mean()

        # Wavelet 高频 Loss
        lw1, lw2 = wavelet_hf_loss(pred, gt)
        lssim    = ssim_loss(pred, gt)

        # 汇总 Full-Res Loss
        total = (w_l1 * l1 +
                 w_soft * lsoft +
                 w_grad * lgrad +
                 w_w1 * lw1 +
                 w_w2 * lw2 +
                 w_ssim * lssim +
                 w_edge * l_edge)

        # —— 在 Correction 前 3 个 epoch 中，额外加一个 1/4 下采的粗尺度 L1 + Soft-IoU
        corr_idx = epoch - cfg['warmup_epochs']
        if corr_idx < 3:
            # 下采至 1/4 分辨率
            p4 = F.interpolate(pred, scale_factor=0.25, mode='bilinear', align_corners=False)
            g4 = F.interpolate(gt,   scale_factor=0.25, mode='bilinear', align_corners=False)

            # 粗尺度 L1
            l1_coarse = F.l1_loss(p4, g4)

            # 粗尺度 Soft-IoU
            inter = (p4 * g4).sum((1,2,3))
            union = (p4 + g4 - p4 * g4).sum((1,2,3)) + 1e-6
            iou_coarse = (inter / union).mean()
            loss_iou_coarse = 1.0 - iou_coarse

            # 粗尺度权重，可根据实际调到 0.2~0.3
            lambda_coarse = 0.5
            total += lambda_coarse * l1_coarse + lambda_coarse * loss_iou_coarse

            return total, {
                'l1': w_l1 * l1,
                'soft': w_soft * lsoft,
                'grad': w_grad * lgrad,
                'w1': w_w1 * lw1,
                'w2': w_w2 * lw2,
                'ssim': w_ssim * lssim,
                'edge': w_edge * l_edge,
                'l1_coarse': lambda_coarse * l1_coarse,
                'iou_coarse': lambda_coarse * loss_iou_coarse
            }

        # 若 corr_idx >= 3，则不加粗尺度项，直接返回 Full-Res 损失
        return total, {
            'l1': w_l1 * l1,
            'soft': w_soft * lsoft,
            'grad': w_grad * lgrad,
            'w1': w_w1 * lw1,
            'w2': w_w2 * lw2,
            'ssim': w_ssim * lssim,
            'edge': w_edge * l_edge
        }

    # ---------- Transition + Fine (剩余 epoch) ----------
    # 更注重细节-锐度，仅留极小 BCE 抑制发散
    w_l1, w_soft  = 1.0, 1.0
    w_grad        = 1.0
    w_w1, w_w2    = 1.0, 0.7
    w_ssim        = 0.2
    w_bce         = 0.05  # 极小 BCE

    w_map = torch.where(gt > 0.9, torch.ones_like(gt),
             torch.where(gt < 0.1, 0.3*torch.ones_like(gt),
                         1.5*torch.ones_like(gt)))
    l1    = (F.l1_loss(pred, gt, reduction='none') * w_map).mean()
    lsoft = soft_boundary_loss(pred, gt)
    lgrad = gradient_loss(pred, gt)
    lw1, lw2 = wavelet_hf_loss(pred, gt)
    lssim    = ssim_loss(pred, gt)
    lbce     = F.binary_cross_entropy_with_logits(pred, gt)

    total = (w_l1 * l1 +
             w_soft * lsoft +
             w_grad * lgrad +
             w_w1 * lw1 +
             w_w2 * lw2 +
             w_ssim * lssim +
             w_bce * lbce)

    return total, {
        'l1':   w_l1 * l1,
        'soft': w_soft * lsoft,
        'grad': w_grad * lgrad,
        'w1':   w_w1 * lw1,
        'w2':   w_w2 * lw2,
        'ssim': w_ssim * lssim,
        'bce':  w_bce * lbce
    }



@torch.no_grad()
def update_alpha_auto(model, init_mask, trunk, trimap):
    """根据残差强度 L_res 动态调整 model.module.alpha"""
    res   = torch.abs(init_mask - trunk)
    area  = trimap.numel()                         # 也可用 (trimap>-1).sum()
    L_res = res.sum() / (area + 1e-6)
    k_base = cfg['alpha_k']
    k_now  = k_base * max(0.5, 1 - epoch / cfg['num_epochs'])
    
    alpha_val = torch.clamp(
        k_now * torch.sqrt(L_res),
        min=0.5,   # 新下限
        max=1.0
    )
    model.module.alpha.data.fill_(alpha_val.item())

# --------------------
# 可视化函数
# --------------------
def save_visualization(rgb, init_mask, gt, trimap, outputs, loss_curve, save_path):
    orig = rgb.shape[-2:]

    def to_np(x):
        x_t = x.detach().cpu()
        if x_t.dim() == 3:            # (C,H,W) → (1,C,H,W)
            x_t = x_t.unsqueeze(0)
        #   统一转单通道 & 拉到 full-res
        x_t = x_t.mean(dim=1, keepdim=True)
        x_t = F.interpolate(x_t, size=orig, mode='bilinear',
                            align_corners=False)
        return x_t[0, 0].numpy()

    (main_up, aux1_up, aux2_up,
     lf1, hf1, lf2, hf2, a1_cb, b1_cb, fused,
     trunk_up, trunk_aux_up, g) = outputs

    # ① 统一把 lf2/hf2 加回来
    # ② fused 有 256 通道，用 mean 显示即可
    fused_vis = fused.mean(dim=1, keepdim=True)

    im_list = [
        rgb[0].permute(1, 2, 0).cpu().numpy(),  # 0 RGB
        init_mask[0, 0].cpu().numpy(),          # 1 Init
        gt[0, 0].cpu().numpy(),                 # 2 GT
        trimap[0, 0].cpu().numpy(),                 # 2 GT
        to_np(main_up),                         # 3 Main
        to_np(aux1_up),                         # 4 Aux1
        to_np(aux2_up),                         # 5 Aux2
        to_np(lf1),                             # 6 LF1
        to_np(hf1),                             # 7 HF1
        to_np(lf2),                             # 8 LF2
        to_np(hf2),                             # 9 HF2
        to_np(a1_cb),                           #10 A1_CB
        to_np(b1_cb),                           #11 B1_CB
        to_np(fused_vis),                       #12 Fused
        to_np(trunk_up),                        #13 Trunk
        to_np(trunk_aux_up),
        to_np(g),                               #14 Gate
        to_np(g * (init_mask - trunk_up))
    ]

    titles = ['RGB', 'Init', 'GT', 'Trimap',
              'Main', 'Aux1', 'Aux2',
              'LF1', 'HF1', 'LF2', 'HF2',
              'A1_CB', 'B1_CB',
              'Fused', 'Trunk', 'Trunk_aux_up', 'Gate', 'g*res']

    assert len(im_list) == len(titles), "img/title 数量必须一致！"

    cols = 5
    rows = math.ceil((len(im_list) + 1) / cols)       # +1 给 loss 曲线
    fig, axs = plt.subplots(rows, cols,
                            figsize=(4 * cols, 4 * rows))
    axs = axs.flatten()

    for i, (im, title) in enumerate(zip(im_list, titles)):
        axs[i].imshow(im, cmap='gray' if im.ndim == 2 else None)
        axs[i].set_title(title)
        axs[i].axis('off')

    # 最后一格画 loss
    axs[len(im_list)].plot(loss_curve[-500:])
    axs[len(im_list)].set_title('Loss')
    axs[len(im_list)].axis('off')

    # 把多余的 subplot 关掉
    for j in range(len(im_list) + 1, rows * cols):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# --------------------
# 训练一个 epoch
# --------------------
def train_one_epoch(epoch, loader, optimizer, scaler, stage, mode='low'):
    model.train()
    optimizer.zero_grad()
    use_gate = model.module.use_gate

    for step, batch in enumerate(loader):
        if step==0:
            print(f"Total Step: {len(loader)} \n")
        # batch 中第四项是 trimap
        rgb, init_mask, gt, trimap = [x.to(device) for x in batch]

        corr_mid = cfg['warmup_epochs'] + cfg['correction_epochs'] // 2

        with autocast(device_type=device.type):
            outputs = model(rgb, init_mask)
            main_up, aux1_up, aux2_up, *_, trunk_up, trunk_aux_up, g = outputs

            if stage == 'correction' and cfg['alpha_mode'] == 'auto':
                corr_start = cfg['warmup_epochs']
                if epoch == corr_start:
                    model.module.alpha.data.fill_(0.8)
                else:
                    update_alpha_auto(model, init_mask, trunk_up, trimap)


            # 将 trimap 传入 compute_total_loss
            tot_main, _ = compute_total_loss(main_up, gt, trimap, epoch, use_gate)
            tot1, _     = compute_total_loss(aux1_up, gt, trimap, epoch, use_gate)
            tot2, _     = compute_total_loss(aux2_up, gt, trimap, epoch, use_gate)
            loss_seg    = tot_main
            aux_loss    = 0.4 * (tot1 + tot2)

            # 初始化主干loss以外的损失
            loss_fill, loss_art, loss_feather = 0.0, 0.0, 0.0

            if epoch >= cfg['warmup_epochs']:
                # fill/art两级权重
                if epoch < cfg['warmup_epochs'] + cfg['correction_epochs'] // 2:
                    w_fill, w_art = 2, 1
                else:
                    w_fill, w_art = 2, 2
                # 计算fill/art损失
                res        = init_mask - trunk_up
                loss_fill  = (g * F.relu(res)).abs().mean() * cfg['lambda_fill'] * w_fill
                loss_art   = (g * F.relu(-res)).abs().mean()  * cfg['lambda_art']  * w_art

            # 羽化只在第4阶段开启
            if epoch >= cfg['warmup_epochs']+cfg['correction_epochs']+cfg['transition_epochs']:
                loss_feather = F.l1_loss(
                    kornia.filters.laplacian(main_up, 3),
                    kornia.filters.laplacian(gt,    3)
                ) * cfg['lambda_feather']

            # Gate 显示监督
            if use_gate and epoch >= cfg['warmup_epochs']:
                # 线性调大 gate-only 权重：0.3 → 0.7 随 correction 进度
                corr_pos   = min(1.0, max(0.0,
                                (epoch - cfg['warmup_epochs'])
                                / max(cfg['correction_epochs'] - 1, 1)))
                w_gate_dyn = 0.3 + (0.8 - 0.3) * corr_pos 

                L_gate, L_reg_exp, L_cons = gate_losses(
                    g, init_mask, trunk_up, gt,
                    w_pos = w_gate_dyn,           # 补齐权重
                    w_neg = 0.2 * w_gate_dyn,     # 抠掉弱一些，可自行调
                    lam_reg = cfg['lambda_gate'],
                    w_cons  = 0.05
                )
            else:
                L_gate = L_reg_exp = L_cons = torch.tensor(0.0, device=rgb.device)

            if stage == 'correction':
                trunk_aux_main_loss, _ = compute_total_loss(trunk_aux_up, gt, trimap, epoch, use_gate)
                """l_aux_w1, l_aux_w2 = wavelet_hf_loss(trunk_aux_up, gt)
                l_aux_grad = gradient_loss(trunk_aux_up, gt)
                w_aux_hf   = 0.5
                w_aux_grad = 0.53"""
                w_aux_main = 0.5
                L_aux = w_aux_main * trunk_aux_main_loss # + w_aux_hf * (l_aux_w1 + l_aux_w2) + w_aux_grad * l_aux_grad
            else:
                L_aux = 0.0

            # 最终累加
            loss = (
                loss_seg + aux_loss + L_gate + L_reg_exp + L_cons + loss_feather
                + loss_fill + loss_art + L_aux
            ) / cfg['accum_steps']

        # 反向与更新
        """scaled_loss = scaler.scale(loss)
        is_final_step = (step + 1) % cfg['accum_steps'] == 0

        if not is_final_step:
            # 中间累积：不做 all-reduce
            with model.no_sync():
                scaled_loss.backward()
        else:
            # 最后一步：正常 backward，会触发 all-reduce
            scaled_loss.backward()

            # 反向后做 unscale、clip、step、update、zero_grad
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()"""

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()                      # ⚠️ 直接反向，不再区分是否 no_sync

        is_final_step = (step + 1) % cfg['accum_steps'] == 0
        if is_final_step:
            # ② 仅在“累够梯度”时做一次参数更新
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 打印与可视化
        if (step + 1) % cfg['print_interval'] == 0:
            print(f"[E{epoch}-{mode}] Step{step}: Loss={(loss.item()*cfg['accum_steps']):.4f} \
            Gate alpha={model.module.alpha.item():.3f}")
            print('[debug] g max=', g.max().item(),
          ' delta max=', (model.module.alpha * g * (init_mask - trunk_up)).max().item())

        if (step + 1) % (cfg['print_interval'] * 2) == 0:
            save_visualization(
                rgb, init_mask, gt, trimap, outputs, loss_history,
                os.path.join(cfg['vis_dir'], f"e{epoch}_{mode}_{step}.png")
            )

def run_validation(model, eval_loader, device, epoch, vis_dir, max_images=6):
    """
    在高分裁剪验证集上跑可视化，
    loader 返回 (x, init_mask, gt, trimap)，
    保存前 max_images 张对比图。
    """
    model.eval()
    os.makedirs(vis_dir, exist_ok=True)
    cnt = 0

    with torch.no_grad():
        for batch in eval_loader:
            # unpack the four items
            x, init_mask, gt, _trimap = [t.to(device) for t in batch]
            pred, *_ = model(x, init_mask)

            # make grid
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
            print(f"[val] saving {save_path}")    # <<-- debug print
            vutils.save_image(grid, save_path)

            cnt += 1
            if cnt >= max_images:
                break

    model.train()


# --------------------
# 主训练流程
# --------------------
if __name__=='__main__':
    """local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])"""

    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    """torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',  # 需用 torchrun 启动
        world_size=world_size,
        rank=local_rank
    )

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type=='cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

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
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['num_epochs'],
        eta_min=cfg['lr'] * 0.1
    )
    scaler   = GradScaler()
    """model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True
    )"""
    model = nn.DataParallel(model)

    # 恢复检查点
    cks = glob.glob(os.path.join(cfg['checkpoint_dir'], 'epoch*.pth'))
    if cks:
        cks.sort(key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)))
        start_epoch = int(re.search(r'epoch(\d+)', cks[-1]).group(1)) + 1
        model.load_state_dict(torch.load(cks[-1], map_location=device))
        print(f"Resume from {cks[-1]}, epoch {start_epoch}")
    else:
        start_epoch = 0; print("Start training from scratch")

    ramp_start = cfg['warmup_epochs']
    loss_history = []

    wave1 = DWTForward(J=1, mode='zero', wave='haar').to(device)
    wave2 = DWTForward(J=2, mode='zero', wave='haar').to(device)


    for epoch in range(start_epoch, cfg['num_epochs']):
        stage = get_stage(epoch)
        print(f"Currently in stage: {stage} \n")

        # 只在新阶段起点做一次的 freeze/unfreeze 或 lr 调整
        if stage != prev_stage:
            if stage == 'warmup':
                # Warmup 阶段：关闭 Gate，冻结 trunk_aux
                model.module.use_gate = False
                model.module.use_trunk_aux = False
                set_gate_grad(model, False)
                set_trunk_aux_grad(model, False)

            elif stage == 'correction':
                model.module.use_gate = True
                model.module.use_trunk_aux = True
                set_gate_grad(model, True)
                set_trunk_aux_grad(model, True)
                print("Unfreeze gate & trunk_aux into correction stage")

                gate_params = [p for n, p in model.named_parameters() if 'res_gate' in n]
                other_params = [p for n, p in model.named_parameters() if 'res_gate' not in n]

                # 计算 correction 进行到哪一步了
                corr_start = cfg['warmup_epochs']
                corr_len   = cfg['correction_epochs']
                corr_mid   = corr_start + corr_len // 2

                if epoch < corr_mid:
                    # Correction 前半程：Gate 的 lr 只乘 2.0
                    optimizer = torch.optim.AdamW([
                        {'params': other_params, 'lr': cfg['lr']},       # backbone lr = 5e-5
                        {'params': gate_params,  'lr': cfg['lr'] * 0.5}, # Gate lr = 1e-4
                    ], weight_decay=1e-4)
                else:
                    # Correction 后半程：Gate 的 lr 再次衰减到 0.5 倍（或根据需要调整）
                    optimizer = torch.optim.AdamW([
                        {'params': other_params, 'lr': cfg['lr']},       # backbone lr = 5e-5
                        {'params': gate_params,  'lr': cfg['lr'] * 2.0}, # Gate lr = 2.5e-5
                    ], weight_decay=1e-4)

            elif stage == 'transition':
                # 冻结 backbone
                for m in [
                    model.module.dwt, model.module.lf1, model.module.hf1, model.module.aspp_l1, model.module.aspp_h1,
                    # model.module.lf2, model.module.hf2, model.module.aspp_l2, model.module.aspp_h2
                ]:
                    for p in m.parameters(): p.requires_grad = False
                model.module.use_gate = True
                model.module.use_trunk_aux = True
                set_gate_grad(model, True)
                set_trunk_aux_grad(model, False)
                # 调低 lr
                new_lr = cfg['lr'] * 0.5
                for g in optimizer.param_groups: g['lr'] = new_lr
                print(f"Freeze backbone, lr={new_lr}")
            else:  # fine
                for p in model.parameters(): p.requires_grad = True
                print("Unfreeze all modules")
            prev_stage = stage

        # 数据加载 & 训练 & 验证
        if stage in ('warmup', 'correction'):
            if low_loader is None:
                low_loader, low_val_loader = build_dataloaders(
                    cfg['csv_path'], cfg['low_res'], cfg['batch_size'],
                    cfg['num_workers'], split_ratio=0.8, seed=cfg['seed'],
                    sample_fraction=0.6, do_crop=False
                )
            train_one_epoch(epoch, low_loader, optimizer, scaler, stage, mode='low')
            run_validation(model, low_val_loader, device, epoch, cfg['vis_dir'])

        else:
            if high_loader is None:
                high_loader, high_val_loader = build_dataloaders(
                    cfg['csv_path'], cfg['high_res'], 1,
                    cfg['num_workers'], split_ratio=0.8, seed=cfg['seed'],
                    sample_fraction=0.6, do_crop=True, crop_size=(512,512)
                )
                print(f">>> [val] epoch={epoch}, loader batches = {len(high_val_loader)}")
            train_one_epoch(epoch, high_loader, optimizer, scaler, stage, mode='high trans')
            run_validation(model, high_val_loader, device, epoch, cfg['vis_dir'])


        # 继续 lr 调度 & 保存
        scheduler.step()
        ckpt = os.path.join(cfg['checkpoint_dir'], f'epoch{epoch}.pth')
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")
        gc.collect(); torch.cuda.empty_cache()

    print('✅ Training complete!')


