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

global gate_history
gate_history = []

# log tracking
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
    def __init__(self):
        super().__init__()
        self._latest = {}

    def update(self, metrics: dict, n=1):
        self._latest = metrics  # ✅ 保存最新一条 step 级数据
        for name, value in metrics.items():
            if name not in self:
                self[name] = AvgMeter()
            self[name].update(value, n)

    def summary(self) -> dict:
        return {name: meter.avg for name, meter in self.items()}

    def latest(self) -> dict:
        return self._latest  # ✅ 方便 step 写入 JSONL

# loss calculation
ssim_loss = SSIMLoss(window_size=11, reduction='mean')
wave1     = DWTForward(J=1, mode='zero', wave='haar')
wave2     = DWTForward(J=2, mode='zero', wave='haar')

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

def guided_structure_loss(gt, guided, threshold=0.15):
    residual = (gt - guided).abs()
    soft_mask = torch.clamp((residual - threshold) / (1 - threshold), 0, 1)
    return (residual * soft_mask).mean()

def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(1,2,3))         # 交集
    union = (pred + target - pred * target).sum(dim=(1,2,3)) + eps  # 并集

    iou = inter / union
    return 1.0 - iou.mean()

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (tensor * mask).sum() / (mask.sum() + 1e-6)

def compute_residual_supervision(gt, guided, unguided, eps=0.02, lam_pos=3.0, lam_neg=1.0):
    res_pos = gt - guided
    res_neg = guided - gt
    mask_pos = (res_pos > eps).float()
    mask_neg = (res_neg > eps).float()
    pred_pos = unguided - guided
    pred_neg = guided - unguided
    loss = lam_pos * F.l1_loss(pred_pos * mask_pos, res_pos * mask_pos)
    loss += lam_neg * F.l1_loss(pred_neg * mask_neg, res_neg * mask_neg)
    return loss

def divergence_from_init(mu: torch.Tensor, init_mask: torch.Tensor, weight=1.0) -> torch.Tensor:
    diff = (mu - init_mask).abs()
    err_init = diff.detach()  # 不让它反传
    focus_mask = torch.clamp((err_init - 0.05) / 0.3, 0, 1)

    return -weight * masked_mean(diff, focus_mask)

def residual_gain_loss(mu, init_mask, gt, eps=1e-6):
    # 1) 绝对误差
    diff_mu   = (mu       - gt).abs()
    diff_init = (init_mask - gt).abs()

    # 2) 改进量 gain = init_error - mu_error
    gain = diff_init - diff_mu  # 正值表示 mu 更接近 gt

    # 3) 区分正负残差区域
    pos_mask = (gt - mu) > 0     # gt>mu  ⇒ 欠估
    neg_mask = (mu - gt) > 0     # mu>gt  ⇒ 过估

    # 4) 分别在各自区域上做 mean
    pos_gain = (gain * pos_mask.float()).sum() / (pos_mask.sum() + eps)
    neg_gain = (gain * neg_mask.float()).sum() / (neg_mask.sum() + eps)

    # 5) 转成 loss：希望 gain 越大越好 ⇒ loss = -gain
    pos_loss = -pos_gain
    neg_loss = -neg_gain

    return pos_loss, neg_loss


def init_mask_regularization(mu: torch.Tensor,
                             init_mask: torch.Tensor,
                             threshold: float = 0.1,
                             ramp: float = 0.05,
                             sharpness: float = 10.0,
                             eps: float = 1e-6) -> torch.Tensor:
    """
    Encourage μ to stay close to init_mask in regions where init_mask is most confident.

    Args:
        mu:         [B,1,H,W]  current unguided output
        init_mask:  [B,1,H,W]  initial weak mask
        threshold:  float in [0,0.5], below which we ignore (low confidence)
        ramp:       float >0, width over which we ramp up to full confidence
        sharpness:  float >0, higher -> steeper focus on high-confidence areas
        eps:        numeric stability
        
    Returns:
        scalar loss
    """
    # 1) raw confidence = |init_mask − 0.5|
    conf = torch.abs(init_mask - 0.5)  # [B,1,H,W]

    # 2) linear ramp from threshold → threshold+ramp:
    #    normalized_conf ∈ [0,1] over that interval, 0 outside
    norm_conf = torch.clamp((conf - threshold) / ramp, 0.0, 1.0)

    # 3) sharpen: emphasize only the very highest confidence pixels
    #    e.g. raise to a power or use a smooth exponential
    focus = norm_conf.pow(sharpness)  # now highly peaked on >~0.5+threshold+ramp

    # 4) deviation
    diff = (mu - init_mask).abs()

    return (diff * focus).sum() / (focus.sum() + eps)


def compute_soft_edge_mask(mask, kernel_size=7, threshold=0.05):
    """
    输入: mask ∈ [B,1,H,W]
    输出: 边缘 mask ∈ [B,1,H,W]，边缘区域为1，其余为0
    """
    blurred = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    diff = torch.abs(mask - blurred)
    edge_mask = (diff > threshold).float()
    return edge_mask

def flatness_penalty(pred, mask=None, mode="l1"):
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    if mask is not None:
        mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        dx = dx * mask_dx
        dy = dy * mask_dy

    return (dx.mean() + dy.mean()) / 2


######## gate loss ############
def gate_entropy_loss(gate, eps=1e-8):
    ent = -gate * torch.log(gate + eps) - (1 - gate) * torch.log(1 - gate + eps)
    return ent.mean()

def gate_center_loss(gate, center=0.5):
    return ((gate - center) ** 2).mean()

def get_gate_soft_residual_loss(gt_matte, guided, gate, threshold=0.15):
    residual = (gt_matte - guided).abs().detach()
    residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-5)

    if residual_norm.shape[-2:] != gate.shape[-2:]:
        residual_norm = F.interpolate(residual_norm, size=gate.shape[-2:], mode='bilinear', align_corners=False)

    # Soft mask 替代 hard threshold
    soft_mask = torch.clamp((residual_norm - threshold) / (1 - threshold), 0, 1)

    if soft_mask.sum() > 0:
        gate_align_loss = F.smooth_l1_loss(gate * soft_mask, residual_norm * soft_mask, reduction='sum')
        gate_align_loss = gate_align_loss / (soft_mask.sum() + 1e-6)
    else:
        gate_align_loss = torch.tensor(0.0, device=gate.device)

    return gate_align_loss

def dynamic_weight_from_gate_mu(gate, base_weight=1.0, min_weight=0.1, center=0.5, width=0.1):
    """
    根据 gate 的 μ 值自动缩放 loss 权重（靠近 center 时权重高，偏离时变低）
    """
    gate_mu = gate.mean().item()
    delta = abs(gate_mu - center)

    if delta < width:
        # gate μ 在稳定范围内，不衰减
        return base_weight
    else:
        # 随偏离程度按高斯方式衰减权重
        scale = np.exp(-((delta - width) ** 2) / (2 * width ** 2))  # 可以换其他衰减函数
        return max(min_weight, base_weight * scale)

def compute_dwt_loss(rf, gt, wavelet_list, lam_hf=1.0):
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
    hf_loss = lam_hf * F.l1_loss(hf_rf_list, hf_gt_list)
    hf_loss = lam_hf * hf_loss

    return ll_loss, hf_loss

def compute_err_map_loss(
    err_pred_map: torch.Tensor,
    pred_matte:    torch.Tensor,
    gt_matte:      torch.Tensor,
    loss_type:     str = "mse"
) -> torch.Tensor:
    """
    Supervise the dual-channel error map with the true over-/under-estimation.
    
    Args:
        err_pred_map (B, 2, H, W): predicted error map channels:
            channel 0 = positive error (pred_matte > gt_matte),
            channel 1 = negative error (gt_matte > pred_matte)
        pred_matte   (B, 1, H, W): final predicted grayscale matte ∈ [0,1]
        gt_matte     (B, 1, H, W): ground-truth grayscale matte ∈ [0,1]
        loss_type    one of {"mse", "l1"} for L2 or L1 supervision.
    
    Returns:
        scalar loss tensor
    """
    # compute positive / negative error targets (grayscale preserves soft edges)
    pos_err = F.relu(pred_matte - gt_matte)   # overshoot
    neg_err = F.relu(gt_matte - pred_matte)   # undershoot

    # stack into (B,2,H,W)
    err_target = torch.cat([pos_err, neg_err], dim=1)

    if loss_type == "mse":
        return F.mse_loss(err_pred_map, err_target)
    elif loss_type == "l1":
        return F.l1_loss(err_pred_map, err_target)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def compute_guided_loss(
    rf, gt, guided, accum_steps=4
):
    # === 1. 区域权重 ===
    base_w = torch.ones_like(gt)
    base_w[gt < 0.1] = 0.3
    base_w[(gt >= 0.1) & (gt <= 0.9)] = 1.5
    base_w[gt > 0.9] = 1.0

    fn_mask = (gt > 0.9) & (guided < 0.3)
    fp_mask = (gt < 0.1) & (guided > 0.7)
    boost_mask = (fn_mask | fp_mask).float()
    final_weight = base_w + 1.5 * boost_mask

    # === 2. Refined 主 loss ===
    l1    = (F.l1_loss(rf, gt, reduction='none') * final_weight).mean()
    lsoft = soft_boundary_loss(rf, gt)
    lgrad = gradient_loss(rf, gt) + gradient_loss(
        F.avg_pool2d(rf, 2), F.avg_pool2d(gt, 2)
    )
    lssim = ssim_loss(rf, gt)

    # === 4. 合并 ===
    total = (
        1.5 * l1
      + 1.2 * lsoft
      + 1.0 * lgrad
      + 0.3 * lssim
    ) / accum_steps

    return total, l1/accum_steps, lsoft/accum_steps, lgrad/accum_steps, lssim/accum_steps

def loss_structural_mimic_binary(
    mu,
    init_mask,
    edge_weight=2.0,
    shape_weight=1.5,
    flat_weight=0.5,
):
    """
    用于 unguided 结构模仿阶段（Epoch 3–4）。
    输入为：预测 μ 和二值 init_mask。
    返回：总损失和各组成项。
    """
    mu = mu.float()
    init_mask = init_mask.float()

    # 1. 边缘对齐损失
    edge_mu = get_soft_edges(mu)
    edge_init = get_soft_edges(init_mask)
    edge_term = F.l1_loss(edge_mu, edge_init) * edge_weight

    # 2. 图像级形状拟合（结构拟合主项）
    shape_term = F.l1_loss(mu, init_mask) * shape_weight

    # 3. 平滑正则项（防止 hallucination 伪结构）
    flat_term = flatness_penalty(mu) * flat_weight

    total = edge_term + shape_term + flat_term
    return total, edge_term, shape_term, flat_term

def compute_unguided_loss(mu, gt, init_mask, accum_steps=4, wavelet_list=None):
    """
    Unguided-only 阶段的主损失函数：结构、边缘、高频、残差修偏
    """
    # 1. L1 + Soft IoU + SSIM + Gradient Loss
    l1 = F.l1_loss(mu, gt)
    soft_iou = 1 - (mu * gt).sum() / (mu + gt - mu * gt).sum().clamp(min=1e-5)
    lsoft = soft_boundary_loss(mu, gt)
    lgrad = gradient_loss(mu, gt)
    lssim = ssim_loss(mu, gt)

    # 2. 结构损失（大结构区域）可选加权加强
    struct = guided_structure_loss(gt, mu)

    # 3. Wavelet HF + LL Loss
    if wavelet_list is not None:
        ll, hf = compute_dwt_loss(mu.float(), gt.float(), wavelet_list)
    else:
        ll = hf = torch.tensor(0.0, device=mu.device)

    # 总和
    total = (
        1.5 * l1 +
        1.2 * lsoft +
        1.0 * lgrad +
        0.6 * lssim +
        1.0 * struct +
        0.5 * ll +
        1.2 * hf
    ) / accum_steps

    return total, l1, soft_iou, lsoft, lgrad, lssim, struct, ll, hf


def compute_phase2_loss(gt, gd, ug, gate_pix, model, accum_steps=4,
                        eps=0.02, lam_pos=30.0, lam_neg=12.0, lam_hf=4.0, lam_gate=3.0):
    """
    Phase 2 losses: residual supervision, high-frequency DWT supervision,
    and structure-aware gate supervision.
    """
    target_size = gd.shape[-2:]
    if gate_pix.shape[-2:] != target_size:
        gate_pix = F.interpolate(gate_pix, size=target_size, mode='bilinear', align_corners=False)
    if ug.shape[-2:] != target_size:
        ug = F.interpolate(ug, size=target_size, mode='bilinear', align_corners=False)

    res_pos = gt - gd
    res_neg = gd - gt
    mask_pos = (res_pos > eps).float()
    mask_neg = (res_neg > eps).float()

    structure_mask = ((gt - gd).abs() > 0.25).float() * 2.0
    mask_pos = mask_pos * structure_mask
    mask_neg = mask_neg * structure_mask

    pred_pos = gate_pix * (ug - gd)
    pred_neg = gate_pix * (gd - ug)

    loss = lam_pos * F.l1_loss(pred_pos * mask_pos, res_pos * mask_pos, reduction='sum')
    loss += lam_neg * F.l1_loss(pred_neg * mask_neg, res_neg * mask_neg, reduction='sum')

    valid_pixels = mask_pos.sum() + mask_neg.sum() + 1e-6
    loss = loss / valid_pixels

    gate_structure_loss = F.l1_loss(gate_pix * structure_mask, structure_mask, reduction='sum')
    gate_structure_loss = gate_structure_loss / (structure_mask.sum() + 1e-6)
    loss += lam_gate * gate_structure_loss

    fused = (gd + gate_pix * (ug - gd)).clamp(0, 1)
    _, hf_loss = compute_dwt_loss(fused, gt, model, model.encoder.wavelet_list, lam_hf)
    loss += hf_loss

    return loss / accum_steps


def compute_phase3_loss(gt, gd, ug, gate_pix, sm, model, accum_steps=4,
                        lam_hf=1.0, sub3=1):
    """
    Phase 3 losses: gate regularization, branch supervision, DWT supervision.
    """
    loss = 0
    if sub3 == 2:
        mean_sm = sm.mean(dim=1, keepdim=True)
        loss += 2.0 * F.l1_loss(gate_pix, mean_sm)
        H = -(gate_pix * torch.log(gate_pix + 1e-6) + (1 - gate_pix) * torch.log(1 - gate_pix + 1e-6))
        loss += 0.05 * (H * (mean_sm + 1e-2)).mean() + 0.05 * gate_pix.mean()

    # Branch supervision
    loss += 0.5 * F.l1_loss(gd, gt) + 0.3 * F.l1_loss(ug * get_soft_edges(gt), gt * get_soft_edges(gt))

    # ✅ Ensure shape match before fusion
    target_size = gd.shape[-2:]
    if ug.shape[-2:] != target_size:
        ug = F.interpolate(ug, size=target_size, mode='bilinear', align_corners=False)
    if gate_pix.shape[-2:] != target_size:
        gate_pix = F.interpolate(gate_pix, size=target_size, mode='bilinear', align_corners=False)

    # DWT supervision
    fused = (gd + gate_pix * (ug - gd)).clamp(0, 1)
    ll_loss, hf_loss = compute_dwt_loss(fused, gt, model, model.encoder.wavelet_list, lam_hf)
    loss += ll_loss + hf_loss

    return loss / accum_steps


def get_phase3_weight(epoch: int,
                      start: int = 4,
                      ramp: int = 3,
                      max_w: float = 0.5) -> float:
    if epoch < start:
        return 0.0
    elif epoch < start + ramp:
        return max_w * (epoch - start) / ramp
    else:
        return max_w

def get_phase2_weight(epoch: int,
                    start: int = 0,
                    ramp: int = 2,
                    max_w: float = 1.0) -> float:
    if epoch < start:
        return 0.0
    elif epoch < start + ramp:
        return max_w * (epoch - start) / ramp
    else:
        return max_w

def get_err_dropout_prob(epoch: int,
                         start_full: int = 2,
                         ramp: int = 3,
                         max_p: float = 1.0) -> float:
    """
    ErrMap Dropout 阶段化：
      - epoch < start_full: 丢弃率 = 1.0（完全丢弃）
      - start_full ≤ epoch < start_full + ramp: 线性从 1 → 0
      - epoch ≥ start_full + ramp: 0（不丢弃，始终用 err_map_pred）
    """
    if epoch < start_full:
        return 1.0
    elif epoch < start_full + ramp:
        # 从 1 线性过渡到 0
        return 1.0 - (epoch - start_full) / ramp
    else:
        return 0.0


# visualization
def save_visualization(
    rgb, init_m, gt, refined, guided, unguided,
    gate, err_map, save_path,
    hf_response=None,
    fused_u=None,
    sparse_init=None,
    max_history=500
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def to_np(x):
        a = x.detach().cpu().numpy()
        # 如果第 0 维是 batch 或通道 1，就降维
        while a.ndim >= 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 3 and a.shape[0] == 3:
            return np.transpose(a, (1,2,0))
        return a

    # 只取第 0 张样本
    r, i, gt0 = rgb[0], init_m[0], gt[0]
    rf, gd, ug, gtmp = refined[0], guided[0], unguided[0], gate[0]
    em = err_map[0]
    sparse_init = sparse_init[0]
    if em.ndim == 4:
        em = em[0]
    em_pos, em_neg = to_np(em[0]), to_np(em[1])

    # 基本几张图
    imgs = [to_np(x) for x in (r, i, gt0, rf, gd, ug, gtmp)]
    imgs += [em_pos, em_neg]
    abs_err = np.clip(np.abs(to_np(rf) - to_np(gt0)), 0, 1)
    imgs.append(abs_err)

    titles = ["RGB","Init","GT","Refined","Guided","Unguided","Gate",
              "ErrMap Pred","AbsErr"]
    cmaps  = [None,"gray","gray","gray","gray","gray","viridis",
              "hot","hot"]

    # Residual |GT-G|
    res = (gt0 - ug).abs()
    res = to_np((res - res.min())/(res.max()-res.min()+1e-5))
    imgs.append(res); titles.append("|GT−G|"); cmaps.append("hot")
    print(titles)

    # HF Response
    if hf_response is not None:
        h = to_np(hf_response[0])
        if h.ndim == 3: h = h.mean(0)
        h = np.clip(h,0,1)
        imgs.append(h); titles.append("HF Resp"); cmaps.append("hot")
    
    if fused_u is not None:
        enc_u_vis = to_np(fused_u[0])
        if enc_u_vis.ndim == 3 and enc_u_vis.shape[0] > 1:
            enc_u_vis = enc_u_vis.mean(0)
        imgs.append(enc_u_vis)
        titles.append("EncoderU Fused")
        cmaps.append("hot")

    if sparse_init is not None:
        vis = to_np(sparse_init[0])
        if vis.ndim == 3: vis = vis.mean(0)
        vis = np.clip(vis,0,1)
        imgs.append(vis); titles.append("Sparse Init"); cmaps.append("gray")

    # 画图
    n = len(imgs)
    cols = 4
    rows = (n+cols-1)//cols
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols,3*rows))
    axs = axs.flatten()
    for ax, img, ttl, cmap in zip(axs, imgs, titles, cmaps):
        ax.imshow(img, cmap=cmap or None, vmin=0, vmax=1)
        ax.set_title(ttl); ax.axis("off")

    """# gate μ 曲线
    ax = axs[n-1]
    hist = gate_history[-max_history:]
    ax.plot(hist, '.-', linewidth=1, markersize=2)
    ax.set_title(f"Gate μ (last {len(hist)} steps)")
    ax.set_xlabel("Step")
    ax.set_ylabel("μ")
    ax.set_ylim(0, 1)
    ax.grid(True)"""

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# metrics calculation
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
    edge_thresh: float = 5e-5,
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