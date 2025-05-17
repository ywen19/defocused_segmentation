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
from pipeline.utils import *


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