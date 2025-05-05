import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


def compute_mse(pred, target):
    """Mean Squared Error"""
    return F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])  # (B,)


def compute_sad(pred, target, reduction='mean'):
    """
    Computes Sum of Absolute Differences (SAD) normalized per pixel.
    """
    B, _, H, W = pred.shape
    per_sample_sad = torch.abs(pred - target).view(B, -1).sum(dim=1) / (H * W)  # normalize here
    if reduction == 'none':
        return per_sample_sad
    elif reduction == 'sum':
        return per_sample_sad.sum()
    elif reduction == 'mean':
        return per_sample_sad.mean()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")



def compute_grad(pred, target, reduction='mean'):
    """
    Computes gradient error between pred and target.
    Args:
        pred, target: (B, 1, H, W)
        reduction: 'mean' | 'sum' | 'none'
    """
    def gradient(x):
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])  # vertical gradient → shape (B, 1, H-1, W)
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])  # horizontal gradient → shape (B, 1, H, W-1)
        return dh, dw

    dh_pred, dw_pred = gradient(pred)
    dh_gt, dw_gt = gradient(target)

    # Align spatial dims in case of rounding inconsistencies (should be equal but safer this way)
    h_dh = min(dh_pred.shape[2], dh_gt.shape[2])
    w_dh = min(dh_pred.shape[3], dh_gt.shape[3])
    h_dw = min(dw_pred.shape[2], dw_gt.shape[2])
    w_dw = min(dw_pred.shape[3], dw_gt.shape[3])

    dh_diff = torch.abs(dh_pred[:, :, :h_dh, :w_dh] - dh_gt[:, :, :h_dh, :w_dh])
    dw_diff = torch.abs(dw_pred[:, :, :h_dw, :w_dw] - dw_gt[:, :, :h_dw, :w_dw])

    if reduction == 'sum':
        return dh_diff.sum() + dw_diff.sum()
    elif reduction == 'mean':
        return (dh_diff.mean() + dw_diff.mean()) / 2
    elif reduction == 'none':
        dh_loss = dh_diff.view(pred.size(0), -1).sum(dim=1)
        dw_loss = dw_diff.view(pred.size(0), -1).sum(dim=1)
        return dh_loss + dw_loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")



def compute_conn(pred, target, step=0.1):
    """
    Connectivity Error based on: "Learning Matting in the Wild" by Liu et al.
    Reference: https://openaccess.thecvf.com/content_cvpr_2019/html/Liu_Learning_Matting_in_The_Wild_CVPR_2019_paper.html
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    batch_size = pred_np.shape[0]

    conn_errors = []
    thresholds = np.arange(0, 1 + step, step)

    for b in range(batch_size):
        pred_img = pred_np[b, 0]
        target_img = target_np[b, 0]
        fg_pred = pred_img >= 0.5
        fg_target = target_img >= 0.5

        # Distance transform for connectivity error
        mask = (fg_pred & fg_target).astype(np.uint8)
        dt = distance_transform_edt(mask == 0)

        pred_diff = np.abs(pred_img - target_img)
        conn_errors.append(np.sum(pred_diff * dt) / pred_img.size)

    return torch.tensor(conn_errors, device=pred.device)  # (B,)


# ---------- Losses ----------
def matte_l1_loss(pred, target):
    return F.l1_loss(pred, target)


def mask_guided_loss(pred, init_mask):
    return F.l1_loss(pred, init_mask)


def edge_aware_loss(pred, target):
    """
    Computes edge-aware L1 loss using LoG-style subtraction of Gaussian blur.
    """
    def get_edge(x, sigma=1.0):
        blurred = gaussian_blur(x, sigma=sigma)
        return x - blurred

    return F.l1_loss(get_edge(pred), get_edge(target))


def gaussian_blur(x, sigma=1.0):
    """
    AMP-safe Gaussian blur using scipy.ndimage.
    x: Tensor of shape (B, 1, H, W)
    """
    x_np = x.detach().cpu().float().numpy()
    blurred = np.zeros_like(x_np)

    for i in range(x_np.shape[0]):
        blurred[i, 0] = gaussian_filter(x_np[i, 0], sigma=sigma)

    return torch.from_numpy(blurred).to(x.device, dtype=x.dtype)
