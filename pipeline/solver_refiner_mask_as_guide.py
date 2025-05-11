import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from torchvision.transforms.functional import gaussian_blur
from pytorch_wavelets import DWTForward
import numpy as np
from kornia.losses import SSIMLoss

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
from model.refiner_sanity_mask_as_guide import RefinerWithDualBranch

# ------ 工具函数 ------
def get_soft_edges(mask):
    lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                       device=mask.device, dtype=torch.float32).view(1,1,3,3)
    e = torch.abs(F.conv2d(mask, lap, padding=1))
    return e / (e.max()+1e-6)

def extract_soft_boundary(mask, sigma=1.0):
    b = gaussian_blur(mask, [5,5], sigma=sigma)
    e = torch.abs(mask - b)
    return e / (e.max()+1e-6)

def soft_boundary_loss(p, g, sigma=1.0):
    pb = gaussian_blur(p, [5,5], sigma=sigma)
    gb = gaussian_blur(g, [5,5], sigma=sigma)
    return F.l1_loss(pb, gb)

def gradient_loss(p, g):
    dxp = p[:,:,:,1:] - p[:,:,:,:-1]
    dyp = p[:,:,1:,:] - p[:,:,:-1,:]
    dxg = g[:,:,:,1:] - g[:,:,:,:-1]
    dyg = g[:,:,1:,:] - g[:,:,:-1,:]
    return F.l1_loss(dxp,dxg) + F.l1_loss(dyp,dyg)

def to_numpy(x):
    x = x.detach().cpu().numpy()
    if x.ndim == 3:
        # assume (C,H,W)
        if x.shape[0] == 3:
            return x.transpose(1,2,0)
        else:
            return x[0]
    elif x.ndim == 2:
        return x
    else:
        raise ValueError(f"Unsupported shape {x.shape}")

# ------ 可视化 ------
def save_visualization(rgb, init_m, gt, pg, gate, pu, sm, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 只取 batch 中第 0 个
    rgb   = rgb[0]
    init  = init_m[0]
    gt0   = gt[0]
    pg0   = pg[0]
    pu0   = pu[0]
    gate0 = gate[0]
    sm0   = sm[0]

    # Convert to numpy
    rgb_np   = to_numpy(rgb)
    init_np  = to_numpy(init)
    gt_np    = to_numpy(gt0)
    pg_np    = to_numpy(pg0)
    pu_np    = to_numpy(pu0)
    sm_np    = to_numpy(sm0)
    gate_np  = to_numpy(gate0)

    # Error maps
    cont    = np.clip(np.abs(pg_np - gt_np)/0.05, 0, 1)
    shape_e = np.abs((pg_np>0.5).astype(float) - gt_np)
    absd    = np.abs(pg_np - gt_np)

    # Plot
    fig, axs = plt.subplots(2,5,figsize=(25,10))
    titles = ["RGB","Init Mask","GT Matte","Pred Guided","Pred Unguided",
              "Static Err","Gate Map","Cont Err","Shape Err","Raw Abs Diff"]
    imgs   = [rgb_np, init_np, gt_np, pg_np, pu_np,
              sm_np, gate_np, cont, shape_e, absd]
    cmaps  = [None,"gray","gray","gray","gray",
              "hot","viridis","jet","hot","hot"]

    for ax, img, ttl, cmap in zip(axs.flatten(), imgs, titles, cmaps):
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(ttl)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# ------ 损失 & Wavelet ------
ssim_loss = SSIMLoss(window_size=11, reduction='mean')
wave1     = DWTForward(J=1, mode='zero', wave='haar')
wave2     = DWTForward(J=2, mode='zero', wave='haar')

# ------ 训练循环 ------
def train_with_error_map(model, loader, opt, scaler, device,
                         epoch, print_int, accum_steps, vis_dir):
    model.train()
    opt.zero_grad()
    wave1.to(device); wave2.to(device)
    pad_w = nn.ReplicationPad2d((0,1,0,0))
    pad_h = nn.ReplicationPad2d((0,0,0,1))

    for i, batch in enumerate(loader,1):
        rgb, init_m, gt = [x.to(device) for x in batch]

        # Guidance + Dropout 调度
        decay = max(0.2, 1.0 - epoch/10)
        mg    = torch.where(init_m>0.99, init_m, init_m*decay)
        drop_prob = 0.1 if epoch<5 else 0.5
        if torch.rand(1).item() < drop_prob:
            mg = torch.zeros_like(mg)

        # Static error map
        se = (gt - init_m).abs()
        dx = pad_w(gt[:,:,:,1:]-gt[:,:,:,:-1])
        dy = pad_h(gt[:,:,1:,:]-gt[:,:,:-1,:])
        em = F.avg_pool2d((dx.abs()+dy.abs()).clamp(0,1),3,1,1)
        sm = (se + 0.5*em).clamp(0,1)

        # Forward with AMP
        with autocast():
            pg, gate = model(rgb, mg, sm)
            pu, _    = model(rgb, torch.zeros_like(mg), sm)

            pg = pg.clamp(0,1)
            pu = pu.clamp(0,1)

            # 基础 losses on pg
            fg     = (gt>0.05).float()
            w_m    = fg + 0.3*(1-fg)
            l1     = (F.l1_loss(pg,gt,reduction='none')*w_m).mean()
            gm     = ((gt>0.1)&(gt<0.9)).float()
            lgray  = (F.mse_loss(pg,gt,reduction='none')*gm).mean()
            lshape = F.mse_loss(get_soft_edges(pg), get_soft_edges(gt))
            lsoft  = soft_boundary_loss(pg, gt)

        # Wavelet losses
        with torch.cuda.amp.autocast(enabled=False):
            _,gh1 = wave1(gt.float()); _,ph1 = wave1(pg.float())
            lw1   = sum(F.l1_loss(p,g) for p,g in zip(ph1,gh1))/len(ph1)
            _,gh2 = wave2(gt.float()); _,ph2 = wave2(pg.float())
            lw2   = sum(F.l1_loss(p,g) for p,g in zip(ph2,gh2))/len(ph2)

        # smoothness / gradient / SSIM
        ls    = F.l1_loss(pg[:,:,:, :-1],pg[:,:,:,1:]) + \
                F.l1_loss(pg[:,:, :-1,:],pg[:,:,1:,:])
        lgrad = gradient_loss(pg,gt) + gradient_loss(
                    F.avg_pool2d(pg,2,2), F.avg_pool2d(gt,2,2))
        lssim = ssim_loss(pg, gt)

        # combine base
        loss = (0.5*l1 + 1.0*lgray + 0.8*lsoft +
                0.3*lshape + 0.3*lw1 + 0.2*lw2 +
                0.4*ls + 1.0*lgrad + 0.5*lssim) / accum_steps

        # Gate 监督
        gate_target = (sm>0.01).float().to(gate.dtype)
        lbce        = F.binary_cross_entropy(gate, gate_target)
        ll1         = F.l1_loss(gate, sm)
        H           = -(gate*torch.log(gate+1e-6) + (1-gate)*torch.log(1-gate+1e-6))
        lent        = (H*(sm+1e-2)).mean()
        lsparse     = gate.mean()
        loss = loss + (5.0*lbce + 1.0*ll1)/accum_steps + 0.1*lent + 0.1*lsparse

        # Binary saturation
        fb = (gt>0.95).float(); bb = (gt<0.05).float()
        lbin = (F.l1_loss(pg*fb, fb) + F.l1_loss(pg*bb, 0*bb))/(2*accum_steps)
        loss += lbin

        # Structure-guided distill
        pe     = (pg.detach()-gt).abs()
        sb     = extract_soft_boundary(gt)
        mask_s = (1-torch.exp(-se*50))*pe*(1+sb*2)
        sw     = max(2.0, 10.0*(0.95**epoch))
        ldist  = sw * F.l1_loss(pg*mask_s, gt*mask_s) / accum_steps
        loss  += ldist

        # backward & step
        scaler.scale(loss).backward()
        if i%accum_steps==0 or i==len(loader):
            scaler.step(opt); scaler.update(); opt.zero_grad()

        # log & vis
        if i%accum_steps==0:
            print(f"Epoch{epoch} Step{i}: gate mean={gate.mean():.3f}, std={gate.std():.3f}")
        if i%print_int==0:
            path = os.path.join(vis_dir, f"e{epoch}_s{i}.png")
            save_visualization(rgb, init_m, gt, pg, gate, pu, sm, path)

if __name__ == "__main__":
    cfg = {
        "num_epochs":     10,
        "batch_size":     2,
        "print_interval": 20,
        "checkpoint_dir": "checkpoints",
        "csv_path":       "../data/pair_for_refiner.csv",
        "resize_to":      (736,1080),
        "num_workers":    6,
        "lr":             5e-4,
        "weight_decay":   0,
        "accum_steps":    4
    }
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = RefinerWithDualBranch(base_channels=64).to(device)
    scaler = GradScaler()
    opt    = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    for epoch in range(1, cfg["num_epochs"]+1):
        train_loader, _ = build_dataloaders(
            cfg["csv_path"], cfg["resize_to"],
            cfg["batch_size"], cfg["num_workers"],
            seed=42, epoch_seed=42+epoch,
            shuffle=True, sample_fraction=100
        )
        train_with_error_map(
            model, train_loader, opt, scaler, device,
            epoch, cfg["print_interval"], cfg["accum_steps"], "vis3"
        )
        ckpt = os.path.join(cfg["checkpoint_dir"], f"refiner_ep{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")
        gc.collect()
        torch.cuda.empty_cache()
