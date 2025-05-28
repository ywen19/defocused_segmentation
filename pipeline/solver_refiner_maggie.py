import os
import sys
import glob
import re
import json
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt
import math
import numpy as np
import kornia
from kornia.losses import SSIMLoss

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
from model.refiner_dwt_maggie import XNetDeep

# --------------------
# 配置项
# --------------------
cfg = {
    'num_epochs': 20,
    'warmup_epochs': 5,
    'freeze_epochs': 3,
    'batch_size': 2,
    'accum_steps': 2,
    'print_interval': 50,
    'checkpoint_dir': 'checkpoints_xnet',
    'log_dir': 'log_xnet',
    'csv_path': '../data/pair_for_refiner.csv',
    'vis_dir': 'vis_gate',
    'low_res': (368, 640),
    'high_res': (736, 1280),
    'num_workers': 4,
    'lr': 3e-5,
    'weight_decay': 1e-5,
    'seed': 42,
    'lambda_gate': 0.02,  # gate 平滑正则系数（略增以更强抑制）
    'lambda_fill': 0.5,  # 残差补齐权重（更强调填充）
    'lambda_art': 0.3,  # 残差伪影去除权重（次于补齐）
    'lambda_edge': 0.2,  # Sobel 边缘重建权重（加强高频细节）
    'lambda_feather': 0.1,  # Laplacian 羽化平滑权重（平衡边缘自然度）         # Laplacian 羽化平滑权重
}

train_loss_json_path = os.path.join(cfg['log_dir'], 'train_loss.jsonl')


# --------------------
# 初始化与辅助函数
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


def set_gate_grad(model, requires_grad: bool):
    m = model.module if hasattr(model, 'module') else model
    for p in m.res_gate_conv.parameters(): p.requires_grad = requires_grad
    m.alpha.requires_grad = requires_grad


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
    # Total variation loss: encourages smoothness
    # Vertical and horizontal differences
    vert_diff = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    hori_diff = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return vert_diff + hori_diff


def compute_total_loss(pred, gt, weights=None):
    """
    计算多分量损失，总和使用可配置权重。
    不使用 BCE，以避免二值拉扯，适用于 alpha matte 连续值。
    weights: dict 包含各损失项权重，若为 None，则从 cfg 中读取动态权重
    返回 tot_loss 以及各项明细。
    """
    if weights is None:
        # 从 cfg 中读取动态 si/di 权重
        si_w = cfg.get('w_si_current', cfg.get('lambda_si', 2.0))
        di_w = cfg.get('w_di_current', cfg.get('lambda_di', 1.0))
        w = {
            'si': si_w,
            'di': di_w,
            'l1': 1.0,
            'ss': 1.0,
            'gr': 0.5,
            'tv': 0.5,
            'el': 0.2,
        }
    else:
        w = weights
    # Soft IoU 损失
    si = soft_iou_loss(pred, gt)
    # Dice 损失
    inter = (pred * gt).sum((1, 2, 3))
    sums = (pred + gt).sum((1, 2, 3)) + 1e-6
    di = 1.0 - (2 * inter / sums).mean()
    # L1 损失
    l1 = F.l1_loss(pred, gt)
    # SSIM 损失
    ss = 1.0 - ssim_loss(pred, gt)
    # Gradient 损失
    gr = gradient_loss(pred, gt)
    # TV 损失
    tvv = tv_loss(pred)
    # Edge 损失
    ep = kornia.filters.sobel(pred)
    eg = kornia.filters.sobel(gt)
    el = F.l1_loss(ep, eg)
    # 加权总和
    tot = (w['si'] * si +
           w['di'] * di +
           w['l1'] * l1 +
           w['ss'] * ss +
           w['gr'] * gr +
           w['tv'] * tvv +
           w['el'] * el)
    return tot, si, di, l1, ss, gr, tvv, el


# --------------------
# 可视化函数
# --------------------
def save_visualization(rgb, init_mask, gt, outputs, loss_curve, save_path):
    orig = rgb.shape[-2:]

    def to_np(x):
        x_t = x.detach().cpu()
        if x_t.ndim == 4:
            pass
        elif x_t.ndim == 3:
            x_t = x_t.unsqueeze(0)
        elif x_t.ndim == 2:
            x_t = x_t.unsqueeze(0).unsqueeze(0)
        x_t = x_t.mean(dim=1, keepdim=True)
        x_t = F.interpolate(x_t, size=orig, mode='bilinear', align_corners=False)
        return x_t[0, 0].numpy()

    main, lf1, hf1, lf2, hf2, fused, lx1, hx1, lx2, hx2, aux1, aux2, trunk, g = outputs
    im_list = [
        rgb[0].permute(1, 2, 0).cpu().numpy(),  # RGB
        init_mask[0, 0].cpu().numpy(),  # Init mask (gray)
        gt[0, 0].cpu().numpy(),  # GT mask (gray)
        to_np(main), to_np(lf1), to_np(hf1), to_np(lf2), to_np(hf2),
        to_np(fused), to_np(lx1), to_np(hx1), to_np(lx2), to_np(hx2),
        to_np(aux1), to_np(aux2), to_np(trunk), to_np(g)
    ]
    titles = ['RGB', 'Init', 'GT', 'Main', 'LF1', 'HF1', 'LF2', 'HF2', 'Fused', 'LX1', 'HX1', 'LX2', 'HX2', 'Aux1',
              'Aux2', 'Trunk', 'Gate']
    # 指定 Init 和 GT 为灰度，其余 feature 也用灰度
    cmaps = [None, 'gray', 'gray'] + ['gray'] * (len(im_list) - 3)

    cols = 5
    rows = math.ceil((len(im_list) + 1) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten()

    for i, (im, title) in enumerate(zip(im_list, titles)):
        axs[i].imshow(im, cmap=cmaps[i])
        axs[i].set_title(title)
        axs[i].axis('off')

    # loss 曲线
    axs[len(im_list)].plot(loss_curve[-500:])
    axs[len(im_list)].set_title('Loss')
    axs[len(im_list)].axis('off')

    # 隐藏多余子图
    for j in range(len(im_list) + 1, rows * cols):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# --------------------
# 训练每周期
# --------------------
def train_one_epoch(epoch, loader, optimizer, scaler, mode='low'):
    model.train();
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        rgb, init_mask, gt = [x.to(device) for x in batch[:3]]
        with autocast(device_type=device.type):
            outputs = model(rgb, init_mask)
            main_up, *_, aux1, aux2, trunk_up, g = outputs
            # 主/辅助
            tot, *_ = compute_total_loss(main_up, gt)
            tot1, *_ = compute_total_loss(aux1, gt)
            tot2, *_ = compute_total_loss(aux2, gt)
            loss_seg = tot;
            aux_loss = 0.4 * (tot1 + tot2)
            # gate正则
            reg_gate = cfg['lambda_gate'] * (g.mean() if model.module.use_gate else 0.0)
            # 高频/羽化/残差
            if model.module.use_gate and epoch >= cfg['warmup_epochs'] + cfg['freeze_epochs']:
                loss_edge = F.l1_loss(kornia.filters.sobel(main_up), kornia.filters.sobel(gt))
                loss_feather = F.l1_loss(kornia.filters.laplacian(main_up, 3), kornia.filters.laplacian(gt, 3))
                reg_edge = cfg['lambda_edge'] * loss_edge
                reg_feather = cfg['lambda_feather'] * loss_feather
                ramp = cfg['num_epochs'] - (cfg['warmup_epochs'] + cfg['freeze_epochs']);
                half = ramp // 2
                w_fill, w_art = (3.0, 1.0) if epoch < cfg['warmup_epochs'] + cfg['freeze_epochs'] + half else (1.0, 1.0)
                res = init_mask - trunk_up
                loss_fill = (g * F.relu(res)).abs().mean() * w_fill * cfg['lambda_fill']
                loss_art = (g * F.relu(-res)).abs().mean() * w_art * cfg['lambda_art']
            else:
                reg_edge = reg_feather = loss_fill = loss_art = 0.0
            loss = (loss_seg + aux_loss + reg_gate + reg_edge + reg_feather + loss_fill + loss_art) / cfg['accum_steps']
        scaler.scale(loss).backward()
        if (step + 1) % cfg['accum_steps'] == 0:
            scaler.unscale_(optimizer);
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer);
            scaler.update();
            optimizer.zero_grad()
        if step % cfg['print_interval'] == 0:
            print(f"[E{epoch}-{mode}] Step{step}:Loss={loss.item() * cfg['accum_steps']:.4f}")
        if (step + 1) % (cfg['print_interval'] * 2) == 0:
            save_visualization(rgb, init_mask, gt, outputs, loss_history,
                               os.path.join(cfg['vis_dir'], f"e{epoch}_{mode}_{step}.png"))


# --------------------
# 主流程
# --------------------
if __name__ == '__main__':
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(cfg['vis_dir'], exist_ok=True)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)
    open(train_loss_json_path, 'w').close()

    low_loader, _ = build_dataloaders(
        cfg['csv_path'], cfg['low_res'], cfg['batch_size'],
        cfg['num_workers'], True, cfg['seed'], sample_fraction=80
    )
    high_loader, _ = build_dataloaders(
        cfg['csv_path'], cfg['high_res'], cfg['batch_size'],
        cfg['num_workers'], True, cfg['seed'], sample_fraction=80
    )

    model = XNetDeep().to(device)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = GradScaler()

    model = nn.DataParallel(model)

    ramp_start = cfg['warmup_epochs'] + cfg['freeze_epochs']
    cks = glob.glob(os.path.join(cfg['checkpoint_dir'], 'epoch*.pth'))
    if cks:
        cks.sort(key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)))
        start_epoch = int(re.search(r'epoch(\d+)', cks[-1]).group(1)) + 1
        model.load_state_dict(torch.load(cks[-1], map_location=device));
        print(f"Resume from {cks[-1]}, epoch{start_epoch}")
    else:
        start_epoch = 0; print('Start training from scratch')
    loss_history = []

    for epoch in range(start_epoch, cfg['num_epochs']):
        # Gate 开关与参数冻结/解冻
        if epoch < ramp_start:
            model.module.use_gate = False
            set_gate_grad(model, False)
        else:
            model.module.use_gate = True
            set_gate_grad(model, True)
            # alpha 调度：最后 3 个 epoch 直接拉满
            if epoch >= cfg['num_epochs'] - 3:
                alpha = 1.0
            else:
                total_steps = cfg['num_epochs'] - 1 - ramp_start
                rel_step = max(0, epoch - ramp_start)
                alpha = 0.1 + (1.0 - 0.1) * (rel_step / total_steps)
                alpha = min(alpha, 1.0)
            with torch.no_grad():
                model.module.alpha.data.fill_(alpha)

        print(f"Epoch {epoch}: gate {'ON' if model.module.use_gate else 'OFF'}, alpha={model.module.alpha.item():.3f}")

        # 训练阶段选择
        if epoch < cfg['warmup_epochs']:
            # Warmup 阶段
            train_one_epoch(epoch, low_loader, optimizer, scaler, mode='low')
        else:
            # 高分辨率阶段：包含 Freeze & Unfreeze
            if epoch == cfg['warmup_epochs']:
                # Freeze backbone
                for m in [model.module.dwt, model.module.lf1_enc, model.module.hf1_enc,
                          model.module.lf2_enc, model.module.hf2_enc,
                          model.module.aspp_lf, model.module.aspp_hf]:
                    for p in m.parameters():
                        p.requires_grad = False
                # 提升学习率
                new_lr = cfg['lr'] * 1.2
                for g in optimizer.param_groups:
                    g['lr'] = new_lr
                print(f"Freeze backbone, lr={new_lr}")
            if epoch == ramp_start:
                # Unfreeze all
                for p in model.parameters():
                    p.requires_grad = True
                print("Unfreeze all modules")
            # High-res 训练
            train_one_epoch(epoch, high_loader, optimizer, scaler, mode='high')

        # 保存 checkpoint
        ckpt = os.path.join(cfg['checkpoint_dir'], f'epoch{epoch}.pth')
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")
        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()

print('✅ Training complete!')
