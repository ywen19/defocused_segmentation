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
import itertools 
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


def get_stage(epoch):
    if epoch < 3:
        return 'guided_only'
    elif 3 <= epoch <= 4:
        return 'unguided_only'
    elif 4 <= epoch <= 5:
        return 'fuse_training'
    else:
        return 'hf_detail'

def apply_stage_overrride(stage, gate_p) -> torch.Tensor:
    """
    根据 stage 强制覆盖 gate_p 为常数。

    Args:
        gate_p:    原始上采样后的 gate 权重张量，shape = [B,1,H,W]
        stage:     'guided_only' 或 'unguided_only'；其他值则返回原 gate_p
        guided_value:    在 guided_only 时用的常数
        unguided_value:  在 unguided_only 时用的常数

    Returns:
        一个同 shape 的 Tensor，其中所有元素都被替换成对应阶段的常数。
    """
    if stage == 'guided_only':
        return torch.zeros_like(gate_p)
    elif stage == 'unguided_only':
        return torch.full_like(gate_p, 1.0)
    else:
        return gate_p

def freeze_module(module, freeze: bool = True):
    for p in module.parameters():
        p.requires_grad = not freeze


def apply_stage_control(model, epoch):
    stage = get_stage(epoch)

    # 解冻所有参数作为初始化
    for p in model.parameters():
        p.requires_grad = True

    if stage == 'guided_only':
        print("→ Stage 1: Only Guided branch active")
        freeze_module(model.encoder_g, False)
        freeze_module(model.decoder_g, False)
        freeze_module(model.err_head,   True)
        freeze_module(model.to_mask,    False) 

        # —— 冻结（requires_grad=False）——
        freeze_module(model.encoder_u,  True)
        freeze_module(model.decoder_u,  True)
        freeze_module(model.to_mask_u,  True)
        freeze_module(model.gate_head,  True)
        freeze_module(model.up,         True)  
    

def build_optimizer(model, base_lr, weight_decay):
    params = filter(lambda p: p.requires_grad, model.parameters())
    return optim.Adam(params, lr=base_lr, weight_decay=weight_decay)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch,
                    print_interval, accum_steps, vis_dir,
                    loss_log_path, metrics_log_path):

    model.train()
    loss_logger = MetricLogger()
    metrics_logger = MetricLogger()
    os.makedirs(vis_dir, exist_ok=True)

    stage = get_stage(epoch)
    model.err_dropout_prob = get_err_dropout_prob(epoch)

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        rgb, init_mask, gt_matte = [x.to(device) for x in batch[:3]]

        # 阶段性逻辑
        if stage == 'guided_only':
            with autocast(device_type=device.type):
                outputs = model(rgb, init_mask, err_map=None, stage=stage)

            # 解包
            (refined, guided, unguided, gate, err_map) = outputs
            gate_const = apply_stage_overrride(stage, gate).detach()

            gate_history.append(gate_const.mean().item())

            if stage == 'guided_only':
                if epoch < 3:
                    loss_guided, l1_g, lsoft_g, lw1_g, lgrad_g, lssim_g = compute_phase1_loss(
                        guided, gt_matte, guided, wave1
                    )
            total_loss = loss_guided + 0.4 * guided_structure_loss(gt_matte, guided) / accum_steps
            ll_g, hf_g = compute_dwt_loss(guided.float(), gt_matte.float(), model.wavelet_list)
            total_loss += 0.5 * ll_g / accum_steps
            total_loss += 0.5 * hf_g / accum_steps
            print(f"At epoch {epoch} [{step}/{len(loader)}]  | total_loss: {total_loss.mean().item()} \n")

        # 反向传播
        scaler.scale(total_loss).backward()
        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if (step + 1) % print_interval == 0:
            save_visualization(
                rgb, init_mask, gt_matte,
                refined, guided, unguided,
                gate_const, err_map,
                f"{vis_dir}/e{epoch:02d}_step{step:04d}_fill_sup.jpg",
                hf_response=None,
            )

        # 强制 float32 以防混用错误
        refined_f = refined.float()
        gt_matte_f = gt_matte.float()
        trimap  = (gt_matte_f > 0.9).float() + (gt_matte_f < 0.1).float()
        unknown = 1.0 - trimap

        metric_dict = matte_metrics(refined_f, gt_matte_f, trimap, unknown)
        metrics_logger.update(metric_dict, rgb.size(0))

        # 日志写入
        with open(loss_log_path, 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'step': step, **loss_logger.latest()}) + '\n')
        with open(metrics_log_path, 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'step': step, **metrics_logger.latest()}) + '\n')

        # 定期打印
        if (step + 1) % print_interval == 0:
            summary = loss_logger.summary()
            print(f"[Epoch {epoch:02d} Step {step:04d}] ",
                  " ".join([f"{k}:{v:.4f}" for k, v in summary.items()]))



if __name__ == "__main__":
    cfg = {
        'num_epochs': 3,
        'batch_size': 2,
        'print_interval': 20,
        'checkpoint_dir': 'checkpoints',
        'csv_path': '../data/pair_for_refiner.csv',
        'log_dir': 'log',
        'resize_to': (736, 1280),
        'num_workers': 6,
        'lr': 1e-4,
        'weight_decay': 0,
        'accum_steps': 4,
        'seed': 42
    }
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)

    train_loss_json_path    = f"{cfg['log_dir']}/train_loss.jsonl"
    train_metrics_json_path = f"{cfg['log_dir']}/train_metrics.jsonl"

    torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])

    model = RefinerMixedHybrid(
        base_channels=128,
        dropout_prob=0.3,
        wavelet_list=['db1'],
        err_dropout_prob=0.0
    ).to(device)
    scaler = GradScaler()

    wave1.to(device)
    wave2.to(device)

    for epoch in range(0, cfg['num_epochs']):
        train_loader, val_loader = build_dataloaders(
            cfg['csv_path'],
            cfg['resize_to'],
            cfg['batch_size'],
            cfg['num_workers'],
            seed=cfg['seed'],
            epoch_seed=cfg['seed'] + epoch,
            shuffle=True,
            sample_fraction=50
        )

        apply_stage_control(model, epoch)
        opt = build_optimizer(model, cfg['lr'], cfg['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[7, 11], gamma=0.8)

        if epoch == 1:
            print("===== Gate requires_grad 状态 =====")
            for n, p in model.gate_head.named_parameters():
                print(f"{n:<30} {p.requires_grad}")

        train_one_epoch(
            model, train_loader, opt, scaler, device, epoch,
            cfg['print_interval'], cfg['accum_steps'],
            vis_dir='vis_train',
            loss_log_path=train_loss_json_path,
            metrics_log_path=train_metrics_json_path
        )

        scheduler.step()
        ckpt_path = os.path.join(cfg['checkpoint_dir'], f"refiner_ep{epoch:02d}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✓ Saved checkpoint: {ckpt_path}")

        gate_history.clear()
        gc.collect()
        torch.cuda.empty_cache()

