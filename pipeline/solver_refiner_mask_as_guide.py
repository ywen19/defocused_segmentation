#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Refiner training with curriculum: 从 RGB 独立学习到 residual-prior 阶段融合
兼容 PyTorch ≥2.0；GPU + AMP；支持 Phase 切换
"""

import os
import sys
import gc
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

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
from model.refiner_sanity_mask_as_guide import RefinerWithResidualGating, custom_dwt

# ──────────────── 工具函数 ────────────────
def get_soft_edges(mask: torch.Tensor) -> torch.Tensor:
    lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], device=mask.device, dtype=mask.dtype).view(1,1,3,3)
    e = torch.abs(F.conv2d(mask, lap, padding=1))
    return e / (e.max() + 1e-6)

def soft_boundary_loss(p: torch.Tensor, g: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    return F.l1_loss(gaussian_blur(p, [5,5], sigma=sigma), gaussian_blur(g, [5,5], sigma=sigma))

def gradient_loss(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return (F.l1_loss(p[:,:,:,1:] - p[:,:,:,:-1], g[:,:,:,1:] - g[:,:,:,:-1]) +
            F.l1_loss(p[:,:,1:,:] - p[:,:,:-1,:], g[:,:,1:,:] - g[:,:, :-1,:]))

# ──────────────── 可视化 ────────────────
def save_visualization(rgb, init_m, gt,
                       refined, guided, unguided,
                       gate, sm, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rgb_, init_, gt_, ref_, g_, ug_, gate_, sm_ = (
        rgb[0], init_m[0], gt[0], refined[0], guided[0], unguided[0], gate[0], sm[0]
    )
    def to_np(x):
        x = x.detach().cpu().numpy()
        if x.ndim == 3:
            return x.transpose(1,2,0) if x.shape[0] == 3 else x[0]
        return x
    imgs = [to_np(rgb_), to_np(init_), to_np(gt_),
            to_np(ref_), to_np(g_), to_np(ug_), to_np(gate_), to_np(sm_)]
    cont_err  = np.clip(np.abs(to_np(ref_) - to_np(gt_)) / 0.05, 0, 1)
    shape_err = np.abs((to_np(ref_) > 0.5).astype(float) - to_np(gt_))
    abs_err   = np.clip(np.abs(to_np(ref_) - to_np(gt_)), 0, 1)
    imgs += [cont_err, shape_err, abs_err]
    titles = ["RGB","Init Mask","GT Matte",
              "Refined","Guided","Unguided","Gate","Err Map (sm)",
              "Cont Err","Shape Err","Abs Err"]
    cmaps  = [None,"gray","gray",
              "gray","gray","gray","viridis","hot",
              "jet","hot","hot"]
    fig, axs = plt.subplots(3,4, figsize=(20,15))
    for ax, img, ttl, cmap in zip(axs.flatten(), imgs, titles, cmaps):
        if cmap: ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:    ax.imshow(img)
        ax.set_title(ttl); ax.axis("off")
    plt.tight_layout(); plt.savefig(save_path); plt.close(fig)

# ──────────────── 损失 & Wavelet ────────────────
ssim_loss = SSIMLoss(window_size=11, reduction='mean')
wave1     = DWTForward(J=1, mode='zero', wave='haar')
wave2     = DWTForward(J=2, mode='zero', wave='haar')

# ──────────────── 训练循环 ────────────────
def train_with_error_map(model, loader, opt, scaler, device,
                         epoch, print_int, accum_steps, vis_dir):
    model.train(); opt.zero_grad()
    wave1.to(device); wave2.to(device)
    pad_w = nn.ReplicationPad2d((0,1,0,0))
    pad_h = nn.ReplicationPad2d((0,0,0,1))

    # 确定 Phase
    phase = 1 if epoch <= 3 else 2 if epoch <= 6 else 3

    for i, batch in enumerate(loader,1):
        rgb, init_m, gt = [x.to(device) for x in batch]

        # 1) 构造 mg
        decay = max(0.2, 1.0 - epoch/10)
        mg = torch.where(init_m > 0.99, init_m, init_m * decay)
        if torch.rand(1).item() < (0.1 if epoch < 5 else 0.5): mg = torch.zeros_like(mg)

        # 2) 计算 sm
        se = (gt - init_m).abs()
        dx = pad_w(gt[:,:,:,1:] - gt[:,:,:,:-1])
        dy = pad_h(gt[:,:,1:,:] - gt[:,:, :-1,:])
        em = F.avg_pool2d((dx.abs()+dy.abs()).clamp(0,1),3,1,1)
        sm = (se + 0.5*em).clamp(0,1)

        # 3) 前向
        with autocast('cuda'):
            _, guided, _, gate = model(rgb, mg, sm)
        with torch.no_grad(), autocast('cuda'):
            _, _, unguided, _ = model(rgb, torch.zeros_like(mg), sm)

        # To FP32
        guided, unguided, gate = guided.float(), unguided.float(), gate.float()
        gate_pix = F.interpolate(gate, guided.shape[-2:], mode='bilinear', align_corners=False)
        refined  = (guided + gate_pix * (unguided - guided)).clamp(0,1)

        # 4) refined losses
        fg, w_m = (gt>0.05).float(), None
        w_m      = fg + 0.3*(1-fg)
        l1       = (F.l1_loss(refined,gt,reduction='none')*w_m).mean()
        gm       = ((gt>0.1)&(gt<0.9)).float()
        lgray    = (F.mse_loss(refined,gt,reduction='none')*gm).mean()
        lshape   = F.mse_loss(get_soft_edges(refined),get_soft_edges(gt))
        lsoft    = soft_boundary_loss(refined,gt)
        with autocast('cuda',enabled=False):
            _,gh1 = wave1(gt.float()); _,ph1 = wave1(refined.float())
            lw1   = sum(F.l1_loss(p,g) for p,g in zip(ph1,gh1))/len(ph1)
            _,gh2 = wave2(gt.float()); _,ph2 = wave2(refined.float())
            lw2   = sum(F.l1_loss(p,g) for p,g in zip(ph2,gh2))/len(ph2)
        ls       = (F.l1_loss(refined[:,:,:, :-1],refined[:,:,:,1:]) +
                    F.l1_loss(refined[:,:, :-1,:],refined[:,:,1:,:]))
        lgrad    = gradient_loss(refined,gt) + gradient_loss(
                      F.avg_pool2d(refined,2,2),F.avg_pool2d(gt,2,2))
        lssim    = ssim_loss(refined,gt)
        loss_refined = (1.0*l1 +1.0*lgray+0.8*lsoft+0.3*lshape+0.5*lw1+0.2*lw2+0.4*ls+1.0*lgrad+0.5*lssim)/accum_steps

        # 5) combine by Phase
        loss = loss_refined
        if phase >= 2:
            loss += 1.0 * F.l1_loss(gate_pix*(unguided-guided), gt-guided) / accum_steps
        if phase >= 3:
            loss_gate   = F.l1_loss(gate_pix, sm)
            H           = -(gate_pix*torch.log(gate_pix+1e-6)+(1-gate_pix)*torch.log(1-gate_pix+1e-6))
            loss_ent    = (H*(sm+1e-2)).mean()
            loss_sparse = gate_pix.mean()
            loss += 5.0*loss_gate + 0.1*loss_ent + 0.1*loss_sparse
            loss += 0.5*F.l1_loss(guided,gt)/accum_steps
            loss += 0.5*F.l1_loss(unguided*get_soft_edges(gt),gt*get_soft_edges(gt))/accum_steps
            with torch.no_grad(): ll_gt,_,_ = custom_dwt(gt)
            _,ll_init,_,_ = model.encoder(rgb,torch.zeros_like(init_m),sm)
            true_ll = ll_gt - ll_init
            gate_ll = F.interpolate(gate,size=true_ll.shape[-2:],mode='bilinear',align_corners=False)
            _,ll_g,_,_ = model.encoder(rgb,mg,sm)
            _,ll_u,_,_ = model.encoder(rgb,torch.zeros_like(mg),sm)
            loss += 1.0 * F.l1_loss(gate_ll*(ll_u-ll_g),true_ll)/accum_steps

        # backward
        scaler.scale(loss).backward()
        if i%accum_steps==0 or i==len(loader):
            scaler.step(opt); scaler.update(); opt.zero_grad()

        if i%accum_steps==0:
            # 打印 loss 和 gate 统计信息
            print(
                f"Epoch{epoch:02d} Step{i:04d} "
                f"Loss{loss.item():.4f} "
                f"Gate μ={gate.mean().item():.4f} σ={gate.std().item():.4f}"
            )
        if i%print_int==0:
            save_visualization(
                               rgb, init_m, gt,
                               refined, guided, unguided,
                               gate, sm,
                               os.path.join(vis_dir, f"e{epoch:02d}_s{i:04d}.png")
                           )

# ──────────────── 主函数 ────────────────
if __name__ == '__main__':
    cfg = {
        'num_epochs':15,'batch_size':2,'print_interval':20,
        'checkpoint_dir':'checkpoints','csv_path':'../data/pair_for_refiner.csv',
        'resize_to':(736,1080),'num_workers':6,'lr':5e-4,'weight_decay':0,
        'accum_steps':4,'seed':42
    }
    os.makedirs(cfg['checkpoint_dir'],exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg['seed']);
    if device.type=='cuda': torch.cuda.manual_seed_all(cfg['seed'])
    model  = RefinerWithResidualGating(base_channels=64).to(device)
    scaler = GradScaler(); opt = optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    wave1.to(device); wave2.to(device)
    for epoch in range(1,cfg['num_epochs']+1):
        train_loader,_ = build_dataloaders(cfg['csv_path'],cfg['resize_to'],cfg['batch_size'],cfg['num_workers'],
                                          seed=cfg['seed'],epoch_seed=cfg['seed']+epoch,
                                          shuffle=True,sample_fraction=100)
        train_with_error_map(model,train_loader,opt,scaler,device,epoch,cfg['print_interval'],cfg['accum_steps'],'vis')
        ckpt=os.path.join(cfg['checkpoint_dir'],f"refiner_ep{epoch:02d}.pth")
        torch.save(model.state_dict(),ckpt)
        print(f"✓ Saved checkpoint: {ckpt}")
        gc.collect(); torch.cuda.empty_cache()
