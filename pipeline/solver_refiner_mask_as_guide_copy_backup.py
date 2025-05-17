#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Refiner training with enhanced curriculum & dual‐channel ErrMap:
  - Phase1→Phase2→Phase3 两段式 Gate 稀疏
  - 强化残差监督权重（含负残差）
  - 周期性 Phase1 回退
  - 温和 LR 调度 (7→0.8×, 11→0.5×)
  - 动态 Err‐dropout 概率随 epoch 增长
  - 记录并可视化 gate μ 曲线（仅最近500步）
"""

import os, sys, gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from torchvision.transforms.functional import gaussian_blur
from pytorch_wavelets import DWTForward
import numpy as np
from kornia.losses import SSIMLoss

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from dataload.build_dataloaders import build_dataloaders
from model.refiner_sanity_mask_attention_blur import RefinerMixedHybrid, multi_dwt_mixed

# 用于记录每个 step 的 gate 平均值
gate_history = []


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

ssim_loss = SSIMLoss(window_size=11, reduction='mean')
wave1     = DWTForward(J=1, mode='zero', wave='haar')
wave2     = DWTForward(J=2, mode='zero', wave='haar')


def save_visualization(rgb, init_m, gt,
                       refined, guided, unguided,
                       gate, err_map, save_path,
                       max_history=500):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    r,i,gt0,rf,gd,ug,gtmp,em = (
        rgb[0], init_m[0], gt[0],
        refined[0], guided[0], unguided[0],
        gate[0], err_map[0]
    )
    def to_np(x):
        a = x.detach().cpu().numpy()
        if a.ndim==3:
            return a.transpose(1,2,0) if a.shape[0]==3 else a[0]
        return a

    imgs = [to_np(t) for t in (r,i,gt0,rf,gd,ug,gtmp, em[0], em[1])]
    cont = np.clip(np.abs(to_np(rf)-to_np(gt0))/0.05,0,1)
    absd = np.clip(np.abs(to_np(rf)-to_np(gt0)),0,1)
    imgs += [cont, absd]

    titles = ["RGB","Init","GT","Refined","Guided","Unguided",
              "Gate","Err_pos","Err_neg","ContErr","AbsErr","Gate μ Curve"]
    cmaps  = [None,"gray","gray","gray","gray","gray",
              "viridis","hot","hot","hot","hot", None]

    fig, axs = plt.subplots(3,4, figsize=(20,15))
    for ax, img, ttl, cmap in zip(axs.flatten()[:11], imgs, titles[:11], cmaps[:11]):
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(ttl); ax.axis("off")

    ax = axs.flatten()[11]
    hist = gate_history[-max_history:]
    ax.plot(hist, '.-', linewidth=1, markersize=2)
    ax.set_title(f"Gate μ (last {len(hist)} steps)")
    ax.set_xlabel("Step"); ax.set_ylabel("μ")
    ax.set_ylim(0,1); ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


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


def compute_phase1_loss(rf, gt, wave1, accum_steps=1):
    """
    Phase 1 losses: weighted L1, soft boundary, wave1, gradient, SSIM.
    """
    w1 = torch.where(gt > 0.9, torch.ones_like(gt),
               torch.where(gt < 0.1, 0.3 * torch.ones_like(gt),
                           1.5 * torch.ones_like(gt)))
    l1 = (F.l1_loss(rf, gt, reduction='none') * w1).mean()
    lsoft = soft_boundary_loss(rf, gt)
    with autocast('cuda', enabled=False):
        _, gh1 = wave1(gt.float())
        _, ph1 = wave1(rf.float())
        lw1 = sum(F.l1_loss(p, g) for p, g in zip(ph1, gh1)) / len(ph1)
    lgrad = gradient_loss(rf, gt) + gradient_loss(F.avg_pool2d(rf, 2), F.avg_pool2d(gt, 2))
    lssim = ssim_loss(rf, gt)
    total = (1.0 * l1 + 1.0 * lsoft + 0.5 * lw1 + 0.5 * lgrad + 0.2 * lssim) / accum_steps
    return total, l1/accum_steps, lsoft/accum_steps, lw1/accum_steps, lgrad/accum_steps, lssim/accum_steps


def compute_phase2_loss(gt, gd, ug, gate_pix, model, accum_steps=1,
                        eps=0.02, lam_pos=3.0, lam_neg=1.0, lam_hf=1.0):
    """
    Phase 2 losses: residual supervision and high-frequency DWT supervision.
    """
    # Residual supervision
    res_pos = gt - gd
    res_neg = gd - gt
    mask_pos = (res_pos > eps).float()
    mask_neg = (res_neg > eps).float()
    pred_pos = gate_pix * (ug - gd)
    pred_neg = gate_pix * (gd - ug)
    loss = lam_pos * F.l1_loss(pred_pos * mask_pos, res_pos * mask_pos)
    loss += lam_neg * F.l1_loss(pred_neg * mask_neg, res_neg * mask_neg)
    # High-frequency DWT
    _, hf_loss = compute_dwt_loss((gd + gate_pix*(ug-gd)).clamp(0,1), gt,
                                  model, model.encoder.wavelet_list, lam_hf)
    loss += hf_loss
    return loss / accum_steps


def compute_phase3_loss(gt, gd, ug, gate, sm, model, accum_steps=1,
                        lam_hf=1.0, sub3=1):
    """
    Phase 3 losses: gate regularization, branch supervision, DWT supervision.
    """
    gate_pix = F.interpolate(gate.float(), gd.shape[-2:], mode='bilinear', align_corners=False)
    loss = 0
    if sub3 == 2:
        mean_sm = sm.mean(dim=1, keepdim=True)
        loss += F.l1_loss(gate_pix, mean_sm)
        H = -(gate_pix*torch.log(gate_pix+1e-6) + (1-gate_pix)*torch.log(1-gate_pix+1e-6))
        loss += 0.02*(H*(mean_sm+1e-2)).mean() + 0.02*gate_pix.mean()
    # Branch supervision
    loss += 0.5*F.l1_loss(gd, gt) + 0.3*F.l1_loss(ug*get_soft_edges(gt), gt*get_soft_edges(gt))
    # DWT supervision
    # Low-frequency
    ll_loss, _ = compute_dwt_loss(gt, gt, model, model.encoder.wavelet_list, lam_hf=0)
    # For simplicity, reuse compute_dwt_loss on fused->gt
    fused = (gd + gate_pix*(ug-gd)).clamp(0,1)
    ll_loss, hf_loss = compute_dwt_loss(fused, gt, model, model.encoder.wavelet_list, lam_hf)
    loss += ll_loss + hf_loss
    return loss / accum_steps

class AvgMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0

class MetricLogger(dict):
    def update(self, metrics: dict, n=1):
        """
        metrics: {'mad': 0.01, 'mse': 1e-4, ...}
        n: batch 大小
        """
        for name, value in metrics.items():
            if name not in self:
                self[name] = AvgMeter()
            self[name].update(value, n)
    def summary(self) -> dict:
        """
        返回一个字典：{'mad': avg_mad, 'mse': avg_mse, ...}
        """
        return {name: meter.avg for name, meter in self.items()}


_sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],
                        dtype=torch.float32).view(1,1,3,3)/8
_sobel_y = _sobel_x.transpose(2,3)

KX, KY = None, None

def _init_kernels(device, dtype):
    global KX, KY
    if KX is None or KX.device != device:
        KX = _sobel_x.to(device, dtype)
        KY = _sobel_y.to(device, dtype)


@torch.no_grad()
def matte_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    trimap: torch.Tensor,
    unknown_mask: torch.Tensor = None,
    esw_const: float = 2.563,
    edge_thresh: float = 5e-3,
) -> dict:
    # 1) 初始化 Sobel 核
    _init_kernels(pred.device, pred.dtype)

    # 2) 计算前景泄漏 MAE (trimap==1 区域)
    fg_mask = (trimap == 1.0).float()
    N_fg    = torch.clamp_min(fg_mask.sum([1,2,3]), 1.0)
    fg_mae  = ((pred - gt).abs() * fg_mask).sum([1,2,3]) / N_fg

    # 3) 计算灰带指标区域
    if unknown_mask is None:
        unknown_mask = ((gt > 0) & (gt < 1)).float()
    else:
        unknown_mask = unknown_mask.float()
    N = torch.clamp_min(unknown_mask.sum([1,2,3]), 1.0)
    diff = pred - gt

    # 4) MAD / MSE
    mad = (diff.abs()   * unknown_mask).sum([1,2,3]) / N
    mse = (diff.square() * unknown_mask).sum([1,2,3]) / N

    # 5) Grad Error (Sobel) over unknown
    gx = F.conv2d(pred, KX, padding=1); gy = F.conv2d(pred, KY, padding=1)
    gpred = torch.sqrt(gx*gx + gy*gy + 1e-12)
    gx = F.conv2d(gt,   KX, padding=1); gy = F.conv2d(gt,   KY, padding=1)
    ggt   = torch.sqrt(gx*gx + gy*gy + 1e-12)
    grad_err = ((gpred - ggt).abs() * unknown_mask).sum([1,2,3]) / N

    # 6) ESW-Error：先算 GT 梯度 & 边缘掩码
    gx = F.conv2d(gt,   KX, padding=1); gy = F.conv2d(gt,   KY, padding=1)
    ggt = torch.sqrt(gx*gx + gy*gy + 1e-12)
    edge_mask = (ggt > edge_thresh).float()

    # 7) 预测梯度
    gx = F.conv2d(pred, KX, padding=1); gy = F.conv2d(pred, KY, padding=1)
    gpred = torch.sqrt(gx*gx + gy*gy + 1e-12)

    # 8) 只在灰带 & GT 真边缘上算 ESW
    valid = (unknown_mask * edge_mask)
    N_esw = torch.clamp_min(valid.sum([1,2,3]), 1.0)
    esw_p = esw_const / (gpred + 1e-6)
    esw_g = esw_const / (ggt   + 1e-6)
    esw_err = ((esw_p - esw_g).abs() * valid).sum([1,2,3]) / N_esw
    esw_err = torch.log1p(esw_err)

    # 9) 返回所有指标
    return {
        'mae':  fg_mae.mean().item(),
        'mad':  mad.mean().item(),
        'mse':  mse.mean().item(),
        'grad': grad_err.mean().item(),
        'esw':  esw_err.mean().item(),
    }

def train_one_epoch(model, loader, opt, scaler, device,
                    epoch, print_int, accum_steps, vis_dir,
                    loss_log_path, metrics_log_path):
    model.train(); opt.zero_grad()
    train_logger = MetricLogger()

    for i, (rgb, init_m, gt, trimap) in enumerate(loader, 1):
        rgb, init_m, gt , trimap = rgb.to(device), init_m.to(device), gt.to(device), trimap.to(device)

        # Prepare inputs
        decay = max(0.2, 1.0 - epoch/10)
        mg = torch.where(init_m>0.99, init_m, init_m*decay)
        if torch.rand(1)<(0.1 if epoch<5 else 0.5): mg.zero_()
        err_pos = F.relu(gt-mg); err_neg = F.relu(mg-gt)
        sm = torch.cat([err_pos, err_neg],1).clamp(0,1)
        # Forward
        with autocast('cuda'):
            refined, gd, ug_dummy, gate = model(rgb, mg, sm)
        with torch.no_grad(), autocast('cuda'):
            _, ug, _, _ = model(rgb, torch.zeros_like(mg), sm)

        gd, ug, gate = gd.float(), ug.float(), gate.float()
        gate_history.append(gate.mean().item())
        gate_pix = F.interpolate(gate, gd.shape[-2:], mode='bilinear', align_corners=False)
        rf = (gd + gate_pix * (ug - gd)).clamp(0, 1)  

        # Compute losses
        loss, l1, lsoft, lw1, lgrad, lssim = compute_phase1_loss(rf, gt, wave1, accum_steps)
        if epoch>3:
            loss += compute_phase2_loss(gt, gd, ug,
                         F.interpolate(gate,gd.shape[-2:],mode='bilinear',align_corners=False),
                         model, accum_steps)
        if epoch>6:
            sub3 = 1 if epoch<11 else 2
            loss += compute_phase3_loss(gt, gd, ug, gate, sm,
                                         model, accum_steps, sub3=sub3)
        if epoch%5==0:
            loss, _, _, _, _, _ = compute_phase1_loss(rf, gt, wave1, accum_steps)

        # calculate metrics
        unknown_mask = (trimap >= 0.5).float()
        batch_metrics = matte_metrics(rf, gt, trimap=trimap, unknown_mask=unknown_mask)
        train_logger.update(batch_metrics, n=rgb.size(0))

        # Backward
        scaler.scale(loss).backward()
        if i%accum_steps==0:
            scaler.step(opt); scaler.update(); opt.zero_grad()
        if i%print_int==0 or i==len(loader)-1:
            avg_metrics = train_logger.summary()
            metrics_str = "  ".join(f"{k}:{v:.4f}" for k,v in avg_metrics.items())
            print(f"Train Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.4f} | Metrics: {metrics_str} ")
            save_visualization(rgb, mg, gt, rf, gd, ug, gate, sm,
                               os.path.join(vis_dir, f"e{epoch:02d}_s{i:04d}.png"))
            # save out the metrics log
            with open(metrics_log_path, "a") as f:
                f.write(json.dumps(avg_metrics) + "\n")
            
            # save out the loss log
            entry = {
                "total": loss.item(),
                "l1":    l1.item(),
                "lsoft": lsoft.item(),
                "lw1":   lw1.item(),
                "lgrad": lgrad.item(),
                "lssim": lssim.item()
            }
            with open(loss_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


def val_one_epoch(model, val_loader, device, vis_dir, epoch, print_int,
                loss_log_path, metrics_log_path):
    """
    Validation and visualization with loss computation.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    val_logger = MetricLogger()

    with torch.no_grad():
        for i, batch in enumerate(val_loader, 1):
            rgb, init_m, gt, trimap = [x.to(device) for x in batch]

            # 构建误差图 (error map)
            err_pos = F.relu(gt - init_m)
            err_neg = F.relu(init_m - gt)
            sm = torch.cat([err_pos, err_neg], dim=1).clamp(0, 1)

            # 模型前向
            with autocast('cuda'):
                refined, guided, ug_dummy, gate  = model(rgb, init_m, sm)
                _, _, unguided, _ = model(rgb, torch.zeros_like(init_m), sm)
            guided, unguided, gate = guided.float(), unguided.float(), gate.float()

            # 融合输出
            gate_pix = F.interpolate(gate.float(), guided.shape[-2:],
                                     mode='bilinear', align_corners=False)
            prediction = (guided + gate_pix * (unguided - guided)).clamp(0, 1)

            # 损失计算 Phase 1
            loss, l1, lsoft, lw1, lgrad, lssim = compute_phase1_loss(prediction, gt, wave1, accum_steps=1)
            # Phase 2
            if epoch > 3:
                loss += compute_phase2_loss(gt, guided, unguided,
                                             gate_pix, model, accum_steps=1)
            # Phase 3
            if epoch > 6:
                sub3 = 1 if epoch < 11 else 2
                loss += compute_phase3_loss(gt, guided, unguided,
                                             gate, sm, model,
                                             accum_steps=1, sub3=sub3)

            total_loss += loss.item()
            count += 1

            # calculate metrics
            unknown_mask = (trimap >= 0.5).float()
            batch_metrics = matte_metrics(prediction, gt, trimap=trimap, unknown_mask=unknown_mask)
            val_logger.update(batch_metrics, n=rgb.size(0))

            # 保存可视化
            if i % print_int == 0 or i==len(val_loader)-1:
                avg_metrics = val_logger.summary()
                metrics_str = "  ".join(f"{k}:{v:.4f}" for k,v in avg_metrics.items())
                print(f"Train Epoch {epoch} [{i}/{len(val_loader)}] Loss: {loss.item():.4f} | Metrics: {metrics_str} ")

                save_path = os.path.join(vis_dir,
                                         f"val_e{epoch:02d}_b{i:04d}.png")
                save_visualization(
                    rgb, init_m, gt,
                    prediction, guided, unguided,
                    gate, sm, save_path
                )

                # save out the metrics log
                with open(metrics_log_path, "a") as f:
                    f.write(json.dumps(avg_metrics) + "\n")

                # save out the loss log
                entry = {
                    "total": loss.item(),
                    "l1":    l1.item(),
                    "lsoft": lsoft.item(),
                    "lw1":   lw1.item(),
                    "lgrad": lgrad.item(),
                    "lssim": lssim.item()
                }
                with open(loss_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")




if __name__=='__main__':
    cfg = {
        'num_epochs':20,'batch_size':2,'print_interval':50,
        'checkpoint_dir':'checkpoints','csv_path':'../data/pair_for_refiner.csv',
        'log_dir': 'log',
        'resize_to':(736,1280),'num_workers':6,'lr':5e-4,'weight_decay':0,
        'accum_steps':4,'seed':42
    }
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)

    # json log file path
    train_loss_json_path = f"{cfg['log_dir']}/train_loss.jsonl"
    train_metrics_json_path = f"{cfg['log_dir']}/train_metrics.jsonl"
    val_loss_json_path = f"{cfg['log_dir']}/val_loss.jsonl"
    val_metrics_json_path = f"{cfg['log_dir']}/val_metrics.jsonl"

    torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.manual_seed_all(cfg['seed'])

    model = RefinerMixedHybrid(base_channels=64, dropout_prob=0.3, wavelet_list=['db1']).to(device)
    scaler = GradScaler()
    opt = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[7,11], gamma=0.8)

    wave1.to(device); wave2.to(device)

    for epoch in range(1, cfg['num_epochs']+1):
        train_loader, val_loader = build_dataloaders(
            cfg['csv_path'], cfg['resize_to'],
            cfg['batch_size'], cfg['num_workers'],
            seed=cfg['seed'], epoch_seed=cfg['seed']+epoch,
            shuffle=True, sample_fraction=100
        )
        train_one_epoch(model, train_loader, opt, scaler,
            device, epoch,
            cfg['print_interval'],
            cfg['accum_steps'],
            'vis_train',
            train_loss_json_path,
            train_metrics_json_path,
        )
        scheduler.step()
        ckpt = os.path.join(cfg['checkpoint_dir'], f"refiner_ep{epoch:02d}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"✓ Saved checkpoint: {ckpt}")

        val_one_epoch(model, val_loader, device, 'vis_val', 
            epoch, cfg['print_interval'], 
            val_loss_json_path,
            val_metrics_json_path,
        )


        gate_history.clear()
        gc.collect(); torch.cuda.empty_cache()