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
from model.refiner_sanity_mask_as_guide import RefinerCascade, custom_dwt, multi_dwt

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


def train_with_error_map(model, loader, opt, scaler, device,
                         epoch, print_int, accum_steps, vis_dir):
    model.train(); opt.zero_grad()
    wave1.to(device); wave2.to(device)

    model.err_dropout_p = min(0.5, epoch * 0.05)
    phase = 1 if epoch <= 3 else 2 if epoch <= 6 else 3
    sub3 = 1 if epoch < 11 else 2

    for i, batch in enumerate(loader, 1):
        rgb, init_m, gt = [x.to(device) for x in batch]

        decay = max(0.2, 1.0 - epoch / 10)
        mg = torch.where(init_m > 0.99, init_m, init_m * decay)
        if torch.rand(1) < (0.1 if epoch < 5 else 0.5): mg.zero_()
        err_pos = F.relu(gt - mg)
        err_neg = F.relu(mg - gt)
        sm = torch.cat([err_pos, err_neg], dim=1).clamp(0, 1)

        with autocast('cuda'):
            refined, gd, ug_dummy, gate = model(rgb, mg, sm)
        with torch.no_grad(), autocast('cuda'):
            _, ug, _, _ = model(rgb, torch.zeros_like(mg), sm)

        gd, ug, gate = gd.float(), ug.float(), gate.float()
        gate_history.append(gate.mean().item())

        gate_pix = F.interpolate(gate, gd.shape[-2:], mode='bilinear', align_corners=False)
        rf = (gd + gate_pix * (ug - gd)).clamp(0, 1)

        # refined losses
        fg = (gt > 0.05).float(); w = fg + 0.3 * (1 - fg)
        l1 = (F.l1_loss(rf, gt, reduction='none') * w).mean()
        gm = ((gt > 0.1) & (gt < 0.9)).float()
        lgray = (F.mse_loss(rf, gt, reduction='none') * gm).mean()
        lshape = F.mse_loss(get_soft_edges(rf), get_soft_edges(gt))
        lsoft = soft_boundary_loss(rf, gt)
        with autocast('cuda', enabled=False):
            _, gh1 = wave1(gt.float()); _, ph1 = wave1(rf.float())
            lw1 = sum(F.l1_loss(p, g) for p, g in zip(ph1, gh1)) / len(ph1)
            _, gh2 = wave2(gt.float()); _, ph2 = wave2(rf.float())
            lw2 = sum(F.l1_loss(p, g) for p, g in zip(ph2, gh2)) / len(ph2)
        ls = (F.l1_loss(rf[:, :, :, :-1], rf[:, :, :, 1:]) +
              F.l1_loss(rf[:, :, :-1, :], rf[:, :, 1:, :]))
        lgrad = gradient_loss(rf, gt) + gradient_loss(
            F.avg_pool2d(rf, 2, 2), F.avg_pool2d(gt, 2, 2))
        lssim = ssim_loss(rf, gt)
        loss_ref = (1 * l1 + 1 * lgray + 0.8 * lsoft + 0.3 * lshape +
                    0.5 * lw1 + 0.2 * lw2 + 0.4 * ls + 1 * lgrad + 0.5 * lssim) / accum_steps

        loss = loss_ref
        if phase >= 2:
            pos_res = gate_pix * (ug - gd)
            loss += 3.0 * F.l1_loss(pos_res, gt - gd) / accum_steps

        if phase >= 3:
            mask_fp = ((mg > 0.1) & (gt < 0.1)).float()
            loss += 1.0 * F.l1_loss((gate_pix * (ug - gd)) * mask_fp,
                                   torch.zeros_like(mask_fp)) / accum_steps
            if sub3 == 2:
                loss_gate = F.l1_loss(gate_pix, sm.mean(dim=1, keepdim=True))
                H = -(gate_pix * torch.log(gate_pix + 1e-6) +
                      (1 - gate_pix) * torch.log(1 - gate_pix + 1e-6))
                loss += 2.0 * loss_gate + 0.05 * (H * (sm.mean(dim=1, keepdim=True) + 1e-2)).mean() + 0.05 * gate_pix.mean()
            loss += 0.5 * F.l1_loss(gd, gt) / accum_steps
            loss += 0.5 * F.l1_loss(ug * get_soft_edges(gt),
                                    gt * get_soft_edges(gt)) / accum_steps
            # 替换为多级 DWT 保证分辨率匹配
            with torch.no_grad():
                coeffs_gt = multi_dwt(gt, model.encoder.levels)
                ll_gt = coeffs_gt[-1][0]
            fused, coeff0 = model.encoder(rgb, torch.zeros_like(mg), sm)
            ll0 = coeff0[-1][0]
            true_ll = ll_gt - ll0
            gate_ll = F.interpolate(gate, size=true_ll.shape[-2:], mode='bilinear', align_corners=False)
            
            # 只解包 fused 和 coeffs
            fused, coeff_g = model.encoder(rgb, mg, sm)  # 解包为 2 个值
            _, coeff_u = model.encoder(rgb, torch.zeros_like(mg), sm)
            ll_g = coeff_g[-1][0]; ll_u = coeff_u[-1][0]
            loss += 1.0 * F.l1_loss(gate_ll * (ll_u - ll_g), true_ll) / accum_steps

        if epoch % 5 == 0:
            loss = loss_ref

        scaler.scale(loss).backward()
        if i % accum_steps == 0 or i == len(loader):
            scaler.step(opt); scaler.update(); opt.zero_grad()

        if i % accum_steps == 0:
            print(f"Ep{epoch:02d} St{i:04d} Ltot={loss.item():.4f} Lref={loss_ref:.4f} Gateμ={gate.mean():.4f} err_do={model.err_dropout_p:.2f}")
        if i % print_int == 0:
            save_visualization(rgb, mg, gt, rf, gd, ug, gate, sm,
                               os.path.join(vis_dir, f"e{epoch:02d}_s{i:04d}.png"))



if __name__=='__main__':
    cfg = {
        'num_epochs':20,'batch_size':2,'print_interval':50,
        'checkpoint_dir':'checkpoints','csv_path':'../data/pair_for_refiner.csv',
        'resize_to':(736,1080),'num_workers':6,'lr':5e-4,'weight_decay':0,
        'accum_steps':4,'seed':42
    }
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.manual_seed_all(cfg['seed'])

    model = RefinerCascade(base_channels=128, err_dropout_p=0.3, levels=3).to(device)
    scaler = GradScaler()
    opt = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[7,11], gamma=0.8)

    wave1.to(device); wave2.to(device)

    for epoch in range(1, cfg['num_epochs']+1):
        train_loader,_ = build_dataloaders(
            cfg['csv_path'], cfg['resize_to'],
            cfg['batch_size'], cfg['num_workers'],
            seed=cfg['seed'], epoch_seed=cfg['seed']+epoch,
            shuffle=True, sample_fraction=100
        )
        train_with_error_map(model, train_loader, opt, scaler,
                             device, epoch,
                             cfg['print_interval'],
                             cfg['accum_steps'],
                             'vis')
        scheduler.step()
        ckpt = os.path.join(cfg['checkpoint_dir'], f"refiner_ep{epoch:02d}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"✓ Saved checkpoint: {ckpt}")
        gate_history.clear()
        gc.collect(); torch.cuda.empty_cache()
