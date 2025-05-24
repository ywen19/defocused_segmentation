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
    elif 3 <= epoch <= 10:
        return 'unguided_only'
    elif 11 <= epoch <= 15:
        return 'fuse_training'
    else:
        return 'hf_detail'

def apply_stage_override(stage, gate_p) -> torch.Tensor:
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
        return torch.ones_like(gate_p)
    else:
        return gate_p

def get_err_dropout_prob(epoch):
    stage = get_stage(epoch)
    if stage == 'unguided_only':
        return 1.0  # 避免扰动 encoder_u/decoder_u
    elif stage == 'guided_only':
        return 1.0  # guided 阶段 err 本就被 zero 掉，这里也可以设置为 1
    else:
        return 0.0  # 融合阶段再开放 err map

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

    elif stage == 'unguided_only':
        print("→ Stage 2: Only Unguided branch active")
        freeze_module(model.encoder_g, True)
        freeze_module(model.decoder_g, True)
        freeze_module(model.err_head,   True)
        freeze_module(model.to_mask,    True) 

        # —— 冻结（requires_grad=False）——
        freeze_module(model.encoder_u,  False)
        freeze_module(model.decoder_u,  False)
        freeze_module(model.to_mask_u,  False)
        freeze_module(model.gate_head,  True)
        freeze_module(model.up,         True)  
    

def build_optimizer(model, base_lr, weight_decay, stage):
    params = filter(lambda p: p.requires_grad, model.parameters())
    return optim.Adam(params, lr=base_lr, weight_decay=weight_decay)

def unguided_schedule_weights(epoch: int):
    # returns reg_w, pos_gain_w, neg_gain_w, bce_w
    # subphase1: epochs 3-4
    if 3 <= epoch < 5:
        return 1.0, 0.0, 0.0, 2.0  # only BCE
    # subphase2: epochs 5-7
    if 5 <= epoch < 8:
        t = epoch - 5  # 0,1,2
        reg = 1.0 - 0.3 * (t / 2)
        return reg, 3.0, 1.5, 1.0
    # subphase3: epochs 8-10
    if 8 <= epoch < 11:
        t = epoch - 8  # 0,1,2
        reg = 0.5
        neg = 1.5 + 1.5 * (t / 2)
        return reg, 3.0, neg, 0.5
    # other epochs: low reg, balanced lower gains
    return 0.2, 2.5, 2.5, 0.2

def generate_sparse_seed_mask(init_mask, seed_ratio=0.01):
    # init_mask ∈ {0,1}
    B, C, H, W = init_mask.shape
    sparse = torch.zeros_like(init_mask)
    for b in range(B):
        fg_coords = (init_mask[b,0] > 0).nonzero(as_tuple=False)
        if len(fg_coords) == 0:
            continue
        num_samples = max(1, int(len(fg_coords) * seed_ratio))
        idx = torch.randperm(len(fg_coords))[:num_samples]
        seeds = fg_coords[idx]
        for y,x in seeds:
            sparse[b,0,y,x] = 1.0
    return sparse


def train_one_epoch(model, loader, optimizer, scaler, device, epoch,
                    print_interval, accum_steps, vis_dir,
                    loss_log_path, metrics_log_path):

    model.train()
    stage = get_stage(epoch)
    model.err_dropout_prob = get_err_dropout_prob(epoch)
    optimizer.zero_grad()
    loss_logger = MetricLogger()
    metrics_logger = MetricLogger()
    gate_history = []

    os.makedirs(vis_dir, exist_ok=True)

    for step, batch in enumerate(loader):
        rgb, init_mask_gt, gt_matte = [x.to(device) for x in batch[:3]]

        raw_init_mask = init_mask_gt  # ← 原始 full init mask

        # === Stage control for encoder input ===
        if stage == 'unguided_only':
            if epoch <= 4:
                sparse_seed = generate_sparse_seed_mask(raw_init_mask, seed_ratio=0.5)
            else:
                sparse_seed = torch.zeros_like(raw_init_mask)

        # === Forward ===
        with autocast(device_type=device.type):
            refined, mg, mu, gate, err_map, fused_u = model(rgb, sparse_seed, err_map=None, stage=stage)

        gate_const = apply_stage_override(stage, gate).detach()
        gate_history.append(gate_const.mean().item())

        # === Loss ===
        if stage == 'guided_only':
            loss_g, l1_g, lsoft_g, lgrad_g, lssim_g = compute_guided_loss(mg, gt_matte, mg, accum_steps)
            total_loss = loss_g + 0.4 * guided_structure_loss(gt_matte, mg) / accum_steps
            ll_g, hf_g = compute_dwt_loss(mg.float(), gt_matte.float(), model.wavelet_list)
            total_loss += 0.5 * ll_g / accum_steps + 0.5 * hf_g / accum_steps

        elif stage == 'unguided_only':
            reg_w, pos_gain_w, neg_gain_w, bce_w = unguided_schedule_weights(epoch)

            if epoch == 3 or (epoch == 4 and step < len(loader) * 0.3):
                loss_total, edge, shape, flat = loss_structural_mimic_binary(
                    mu, raw_init_mask,
                    edge_weight=2.0,
                    shape_weight=1.5,
                    flat_weight=0.5,
                )
                total_loss = loss_total / accum_steps

                print(f"[Step {step}] EDGE:{edge.item():.4f} SHAPE:{shape.item():.4f} "
                      f"FLAT:{flat.item():.4f} TOTAL:{loss_total.item():.4f}")

            else:
                loss_u, *_ = compute_unguided_loss(mu, gt_matte, raw_init_mask, accum_steps=accum_steps, wavelet_list=model.wavelet_list)
                bce_sil = F.binary_cross_entropy(mu, raw_init_mask.to(mu.dtype)) * bce_w / accum_steps
                reg = init_mask_regularization(mu, raw_init_mask)*reg_w / accum_steps
                pos_gain_loss, neg_gain_loss = residual_gain_loss(mu, raw_init_mask, gt_matte) 
                pos_gain_loss *= pos_gain_w / accum_steps
                neg_gain_loss *= neg_gain_w / accum_steps

                total_loss = loss_u + reg + pos_gain_loss + neg_gain_loss + bce_sil
                print(f"Debug Epoch {epoch} Step [{step}] | loss_u: {loss_u.mean().item():.4f} | "
                      f"reg: {reg.mean().item():.4f} | pos: {pos_gain_loss.mean().item():.4f} | "
                      f"neg: {neg_gain_loss.mean().item():.4f} | BCE: {bce_sil.mean().item():.4f} | "
                      f"total: {total_loss.mean().item():.4f}")

        else:
            raise ValueError(f"Unknown training stage: {stage}")

        # === Optimizer ===
        scaler.scale(total_loss).backward()
        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # === Visualization ===
        if (step + 1) % print_interval == 0:
            save_visualization(
                rgb, raw_init_mask, gt_matte,
                refined, mg, mu,
                gate_const, err_map,
                f"{vis_dir}/e{epoch:02d}_step{step:04d}_withUG.jpg",
                hf_response=None,
                fused_u=fused_u,
                sparse_init=sparse_seed,  # ✅ 传入和模型 forward 同一张 sparse mask
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



def get_latest_checkpoint(checkpoint_dir):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("refiner_ep") and f.endswith(".pth")]
    if not ckpts:
        return None, 0

    # 按 epoch 编号排序
    ckpts = sorted(ckpts, key=lambda x: int(x.replace("refiner_ep", "").replace(".pth", "")))
    latest_ckpt = ckpts[-1]
    latest_epoch = int(latest_ckpt.replace("refiner_ep", "").replace(".pth", ""))
    return os.path.join(checkpoint_dir, latest_ckpt), latest_epoch + 1  # 继续从下一 epoch 开始


if __name__ == "__main__":
    cfg = {
        'num_epochs': 5,
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

    # load if exists
    latest_ckpt_path, start_epoch = get_latest_checkpoint(cfg['checkpoint_dir'])
    if latest_ckpt_path:
        print(f"🔁 Resuming from {latest_ckpt_path} (epoch {start_epoch})")
        model.load_state_dict(torch.load(latest_ckpt_path, map_location=device))
    else:
        print("🚀 Starting fresh from epoch 0")
        start_epoch = 0

    for epoch in range(start_epoch, cfg['num_epochs']):
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
        # model.err_dropout_prob = get_err_dropout_prob(epoch)
        opt = build_optimizer(model, cfg['lr'], cfg['weight_decay'], get_stage(epoch))
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[7, 11], gamma=0.8)

        train_one_epoch(
            model, train_loader, opt, scaler, device, epoch,
            cfg['print_interval'], cfg['accum_steps'],
            vis_dir='vis_train_withUG',
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

