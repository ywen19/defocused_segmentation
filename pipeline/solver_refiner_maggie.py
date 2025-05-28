import os
import sys
import glob
import re
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

import matplotlib.pyplot as plt
import numpy as np
import kornia
from kornia.losses import SSIMLoss

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
from model.refiner_dwt_maggie import XNetDeep
from pipeline.utils import *

# --------------------
# 配置项
# --------------------
cfg = {
    'num_epochs': 20,
    'warmup_epochs': 5,
    'freeze_epochs': 3,
    'batch_size': 2,
    'accum_steps': 2,
    'print_interval': 20,
    'checkpoint_dir': 'checkpoints_xnet',
    'log_dir': 'log_xnet',
    'csv_path': '../data/pair_for_refiner.csv',
    'vis_dir': 'vis_train',
    'low_res': (368, 640),
    'high_res': (736, 1280),
    'num_workers': 4,
    'lr': 3e-5,
    'weight_decay': 1e-5,
    'seed': 42
}

# 日志文件路径
train_loss_json_path = os.path.join(cfg['log_dir'], 'train_loss.jsonl')

# --------------------
# 环境初始化
# --------------------
torch.manual_seed(cfg['seed'])
np.random.seed(cfg['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}, GPUs: {torch.cuda.device_count()}")
if device.type == 'cuda':
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------
# 权重初始化函数
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


# --------------------
# 损失函数定义
# --------------------
ssim_loss = SSIMLoss(window_size=11, reduction='mean')
bce_loss = nn.BCEWithLogitsLoss()


def gradient_loss(p, g):
    return (F.l1_loss(p[..., 1:] - p[..., :-1], g[..., 1:] - g[..., :-1]) +
            F.l1_loss(p[..., 1:, :] - p[..., :-1, :], g[..., 1:, :] - g[..., :-1, :]))


def soft_iou_loss(pred, target, eps=1e-6):
    inter = (pred * target).sum((1, 2, 3))
    union = (pred + target - pred * target).sum((1, 2, 3)) + eps
    return 1.0 - (inter / union).mean()


def tv_loss(x):
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., 1:] - x[..., :, :-1]).abs().mean()


def compute_total_loss(pred, gt):
    si = soft_iou_loss(pred, gt)
    inter = (pred * gt).sum((1, 2, 3))
    sums = (pred + gt).sum((1, 2, 3)) + 1e-6
    di = 1.0 - (2 * inter / sums).mean()
    bc = bce_loss(pred, gt)
    l1 = F.l1_loss(pred, gt)
    ss = 1.0 - ssim_loss(pred, gt)
    gr = gradient_loss(pred, gt)
    tvv = tv_loss(pred)
    ep = kornia.filters.sobel(pred)
    eg = kornia.filters.sobel(gt)
    el = F.l1_loss(ep, eg)
    tot = 2.0 * si + di + bc + l1 + 0.5 * gr + 0.5 * tvv + 0.2 * el
    return tot, si, di, bc, l1, ss, gr, tvv, el


# --------------------
# 可视化函数
# --------------------
def save_visualization(rgb, init_mask, gt, outputs, loss_curve, save_path):
    orig = rgb.shape[-2:]

    def to_np(x):
        x_t = x.detach().cpu()
        # 统一到 4D: (B, C, H, W)
        if x_t.ndim == 4:
            pass
        elif x_t.ndim == 3:
            x_t = x_t.unsqueeze(0)
        elif x_t.ndim == 2:
            x_t = x_t.unsqueeze(0).unsqueeze(0)
        # 平均通道
        x_t = x_t.mean(dim=1, keepdim=True)
        # 上采样到原始尺寸
        x_t = F.interpolate(x_t, size=orig, mode='bilinear', align_corners=False)
        return x_t[0, 0].numpy()

    main, lf1, hf1, lf2, hf2, fused, lx1, hx1, lx2, hx2, aux1, aux2 = outputs
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    im_list = [rgb[0].permute(1, 2, 0).cpu().numpy(), init_mask, gt, main,
               lf1, hf1, lf2, hf2, fused, lx1, hx1, lx2, hx2, aux1, aux2]
    cmaps = [None, 'gray', 'gray', 'gray', 'viridis', 'magma', 'plasma', 'cividis',
             'inferno', 'cubehelix', 'cividis', 'inferno', 'plasma', 'gray', 'gray']
    for ax, im, cmap in zip(axs.flatten(), im_list, cmaps):
        if isinstance(im, torch.Tensor):
            ax.imshow(to_np(im), cmap=cmap)
        else:
            ax.imshow(im, cmap=cmap)
        ax.axis('off')
    axs[-1, -1].plot(loss_curve[-500:]);
    axs[-1, -1].set_title('Loss')
    plt.tight_layout();
    plt.savefig(save_path);
    plt.close(fig)


# --------------------
# 单个 Epoch 训练函数
# --------------------
def train_one_epoch(epoch, loader, optimizer, scaler, mode='low'):
    model.train()
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        rgb, init_mask, gt = [x.to(device) for x in batch[:3]]
        with autocast(device_type=device.type):
            outputs = model(rgb)
            tot, *_ = compute_total_loss(outputs[0], gt)
            loss = tot / cfg['accum_steps']
        # 记录到 jsonl 日志
        with open(train_loss_json_path, 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'step': step, 'loss': loss.item() * cfg['accum_steps']}) + '\n')
        loss_history.append(loss.item() * cfg['accum_steps'])
        if step % cfg['print_interval'] == 0:
            print(f"[E{epoch} - {mode}] Step {step}: Loss={loss.item() * cfg['accum_steps']:.4f}")
        scaler.scale(loss).backward()
        if (step + 1) % cfg['accum_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        if (step + 1) % (cfg['print_interval'] * 2) == 0:
            save_path = os.path.join(cfg['vis_dir'], f"e{epoch:02d}_{mode}_s{step:04d}.png")
            save_visualization(rgb, init_mask, gt, outputs, loss_history, save_path)


# --------------------
# 主训练流程
# --------------------
if __name__ == '__main__':
    os.makedirs(cfg['vis_dir'], exist_ok=True)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)
    # 清空训练日志文件
    with open(train_loss_json_path, 'w') as f:
        pass

    # 构建加载器
    low_loader, _ = build_dataloaders(
        cfg['csv_path'], cfg['low_res'], cfg['batch_size'], cfg['num_workers'],
        True, cfg['seed'], sample_fraction=1
    )
    high_loader, _ = build_dataloaders(
        cfg['csv_path'], cfg['high_res'], cfg['batch_size'], cfg['num_workers'],
        True, cfg['seed'], sample_fraction=1
    )

    # 模型与优化器
    model = XNetDeep().to(device)
    model.apply(init_weights)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = GradScaler()

    # 恢复 checkpoint
    cks = glob.glob(os.path.join(cfg['checkpoint_dir'], 'epoch*.pth'))
    if cks:
        cks.sort(key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)))
        ck = cks[-1]
        start_epoch = int(re.search(r'epoch(\d+)', ck).group(1)) + 1
        model.load_state_dict(torch.load(ck, map_location=device))
        print(f"Resume from {ck}, starting at epoch {start_epoch}")
    else:
        start_epoch = 0
        print('Start training from scratch')

    model = nn.DataParallel(model)

    loss_history = []
    # Main loop
    for epoch in range(start_epoch, cfg['num_epochs']):
        # Phase1: low-res warm-up
        if epoch < cfg['warmup_epochs']:
            print(f"Epoch {epoch}: Low-res warm-up")
            train_one_epoch(epoch, low_loader, optimizer, scaler, mode='low')
        else:
            # Phase2: high-res with freeze/unfreeze
            if epoch == cfg['warmup_epochs']:
                for m in [model.dwt, model.lf_enc1, model.hf_enc1,
                          model.lf_enc2, model.hf_enc2,
                          model.aspp_lf, model.aspp_hf]:
                    for p in m.parameters(): p.requires_grad = False
                # 调整学习率至高分辨率阶段初始值
                new_lr = cfg['lr'] * 1.2
                for g in optimizer.param_groups:
                    g['lr'] = new_lr
                print(f"Freeze backbone for initial high-res training, set lr={new_lr}")
            if epoch == cfg['warmup_epochs'] + cfg['freeze_epochs']:
                for p in model.parameters(): p.requires_grad = True
                print("Unfreeze entire network for fine-tuning")
            # 高分辨率训练
            print(f"Epoch {epoch}: High-res training")
            train_one_epoch(epoch, high_loader, optimizer, scaler, mode='high')

        # 保存 checkpoint
        ckpt_path = os.path.join(cfg['checkpoint_dir'], f'epoch{epoch}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
    print('✅ 训练完成！')
