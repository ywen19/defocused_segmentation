import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import numpy as np

def connected_components(bin_map):
    """
    Basic 4-connectivity connected components labeling for binary numpy array.
    Returns labeled map and number of labels.
    """
    H, W = bin_map.shape
    labeled = np.zeros_like(bin_map, dtype=np.int32)
    label_id = 0
    for i in range(H):
        for j in range(W):
            if bin_map[i, j] and labeled[i, j] == 0:
                label_id += 1
                # BFS flood fill
                stack = [(i, j)]
                labeled[i, j] = label_id
                while stack:
                    x, y = stack.pop()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and bin_map[nx, ny] and labeled[nx, ny] == 0:
                            labeled[nx, ny] = label_id
                            stack.append((nx, ny))
    return labeled, label_id

from utils import (
    extract_unknown_mask,
    compute_cf_loss,
    compute_grad_loss_soft_only,
    compute_conn_mismatch_loss,
    compute_sad_with_trimap,
    compute_ctr_loss,
    compute_mse_with_trimap,
    compute_grad_metric,
    compute_conn_with_unknown_mask
)


def show_mask_region(title, mask_tensor):
    mask_np = mask_tensor.squeeze().detach().cpu().numpy()
    plt.imshow(mask_np, cmap='jet')
    plt.title(title)
    plt.axis("off")


# === Loss 可视化 ===

def visualize_cf_loss_region(gt_alpha, pred_alpha, trimap=None):
    lower, upper = 0.05, 0.95
    soft_mask = ((gt_alpha > lower) & (gt_alpha < upper)).float()
    if trimap is not None:
        unknown = extract_unknown_mask(trimap)
        soft_mask *= unknown
    loss_map = ((pred_alpha - gt_alpha) ** 2) * soft_mask

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    show_mask_region("GT Alpha", gt_alpha)
    plt.subplot(2, 2, 2)
    show_mask_region("Pred Alpha", pred_alpha)
    plt.subplot(2, 2, 3)
    show_mask_region("CF Mask", soft_mask)
    plt.subplot(2, 2, 4)
    show_mask_region("CF Loss Map", loss_map)
    plt.tight_layout()
    plt.show()


def visualize_grad_loss_region(gt_alpha, pred_alpha, coarse_mask=None):
    def get_gradient(x):
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return gx, gy

    eps = 1e-6
    lower, upper = 0.05, 0.95
    soft_mask = ((gt_alpha > lower) & (gt_alpha < upper)).float()
    if coarse_mask is not None:
        soft_mask *= ((coarse_mask > lower) & (coarse_mask < upper)).float()

    pred_gx, pred_gy = get_gradient(pred_alpha)
    gt_gx, gt_gy     = get_gradient(gt_alpha)

    edge_strength = torch.sqrt(gt_gx ** 2 + gt_gy ** 2 + eps)
    edge_weight = edge_strength / (edge_strength.max() + eps)
    weighted_mask = soft_mask * edge_weight

    diff_map = (torch.abs(pred_gx - gt_gx) + torch.abs(pred_gy - gt_gy)) / 2.0
    masked_map = diff_map * weighted_mask

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    show_mask_region("GT Alpha", gt_alpha)
    plt.subplot(2, 2, 2)
    show_mask_region("Pred Alpha", pred_alpha)
    plt.subplot(2, 2, 3)
    show_mask_region("Grad Weighted Mask", weighted_mask)
    plt.subplot(2, 2, 4)
    show_mask_region("Grad Loss Map", masked_map)
    plt.tight_layout()
    plt.show()


def visualize_conn_loss_region(gt_alpha, pred_alpha, threshold=0.5):
    gt_bin = (gt_alpha > threshold).float()
    pred_bin = (pred_alpha > threshold).float()
    mismatch = torch.abs(gt_bin - pred_bin)
    bce_map = F.binary_cross_entropy(pred_alpha, gt_bin, reduction='none') * mismatch

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    show_mask_region("GT Alpha", gt_alpha)
    plt.subplot(2, 2, 2)
    show_mask_region("Pred Alpha", pred_alpha)
    plt.subplot(2, 2, 3)
    show_mask_region("Conn Mismatch", mismatch)
    plt.subplot(2, 2, 4)
    show_mask_region("Conn Loss Map", bce_map)
    plt.tight_layout()
    plt.show()


def visualize_sad_loss_region(gt_alpha, pred_alpha):
    # 使用 soft 区域替代 trimap
    soft_mask = ((gt_alpha > 0.1) & (gt_alpha < 0.9)).float()
    abs_diff = torch.abs(pred_alpha - gt_alpha)
    loss_map = abs_diff * soft_mask

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    show_mask_region("GT Alpha", gt_alpha)
    plt.subplot(2, 2, 2)
    show_mask_region("Pred Alpha", pred_alpha)
    plt.subplot(2, 2, 3)
    show_mask_region("SAD Soft Mask", soft_mask)
    plt.subplot(2, 2, 4)
    show_mask_region("SAD Loss Map", loss_map)
    plt.tight_layout()
    plt.show()


def visualize_ctr_loss_region(gt_alpha, pred_alpha, threshold=0.5):
    gt_bin = (gt_alpha > threshold).float()
    pred_bin = (pred_alpha > threshold).float()
    diff_mask = torch.abs(gt_bin - pred_bin)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    show_mask_region("GT Alpha", gt_alpha)
    plt.subplot(2, 2, 2)
    show_mask_region("Pred Alpha", pred_alpha)
    plt.subplot(2, 2, 3)
    show_mask_region("CTR Region", diff_mask)
    plt.tight_layout()
    plt.show()


# === Metric 可视化 ===

def visualize_mse_metric_region(gt_alpha, pred_alpha, trimap=None):
    # trimap 可空，使用 gt_alpha 估计
    mask_src = trimap if trimap is not None else gt_alpha
    unknown = extract_unknown_mask(mask_src)
    mse_map = (pred_alpha - gt_alpha) ** 2 * unknown

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    show_mask_region("GT Alpha", gt_alpha)
    plt.subplot(2, 2, 2)
    show_mask_region("Pred Alpha", pred_alpha)
    plt.subplot(2, 2, 3)
    show_mask_region("Unknown Mask", unknown)
    plt.subplot(2, 2, 4)
    show_mask_region("MSE Map", mse_map)
    plt.tight_layout()
    plt.show()


def visualize_sad_metric_region(gt_alpha, pred_alpha, trimap=None):
    mask_src = trimap if trimap is not None else gt_alpha
    unknown = extract_unknown_mask(mask_src, strategy='auto')
    sad_map = torch.abs(pred_alpha - gt_alpha) * unknown

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    show_mask_region("GT Alpha", gt_alpha)
    plt.subplot(2, 2, 2)
    show_mask_region("Pred Alpha", pred_alpha)
    plt.subplot(2, 2, 3)
    show_mask_region("Unknown Mask", unknown)
    plt.subplot(2, 2, 4)
    show_mask_region("SAD Map", sad_map)
    plt.tight_layout()
    plt.show()

def visualize_grad_metric_region(gt_alpha, pred_alpha, trimap=None, eps=1e-6):
    mask_src = trimap if trimap is not None else gt_alpha
    unknown = extract_unknown_mask(mask_src, strategy='auto')
    def get_gradient_xy(x):
        kx = torch.tensor([[1, 0, -1],[2,0,-2],[1,0,-1]],dtype=torch.float32).view(1,1,3,3).to(x.device)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32).view(1,1,3,3).to(x.device)
        gx = F.conv2d(x,kx,padding=1)
        gy = F.conv2d(x,ky,padding=1)
        return gx, gy
    pred_gx, pred_gy = get_gradient_xy(pred_alpha)
    gt_gx, gt_gy     = get_gradient_xy(gt_alpha)
    diff_map = (torch.abs(pred_gx - gt_gx) + torch.abs(pred_gy - gt_gy)) / 2.0
    masked_grad_map = diff_map * unknown
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    show_mask_region("GT Alpha", gt_alpha)
    plt.subplot(2,2,2)
    show_mask_region("Pred Alpha", pred_alpha)
    plt.subplot(2,2,3)
    show_mask_region("Unknown Mask", unknown)
    plt.subplot(2,2,4)
    show_mask_region("Grad Error Map", masked_grad_map)
    plt.tight_layout()
    plt.show()

def visualize_conn_metric_region(gt_alpha, pred_alpha, trimap=None, threshold=0.5):
    mask_src = trimap if trimap is not None else gt_alpha
    unknown = extract_unknown_mask(mask_src)
    # 最大连通前景
    gt_bin = (gt_alpha[0,0].detach().cpu().numpy() > threshold).astype(np.uint8)
    labeled, num = connected_components(gt_bin)
    max_region, max_area = 0,0
    for j in range(1,num+1):
        area = (labeled==j).sum()
        if area>max_area:
            max_area, max_region = area, j
    gt_main = (labeled==max_region).astype(np.float32)
    gt_main_t = torch.from_numpy(gt_main).to(gt_alpha.device).unsqueeze(0).unsqueeze(0)
    eval_mask = (gt_main_t * unknown).clamp(0,1)
    bce_map = F.binary_cross_entropy(pred_alpha, gt_main_t, reduction='none') * eval_mask
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    show_mask_region("GT Alpha", gt_alpha)
    plt.subplot(2,2,2)
    show_mask_region("Pred Alpha", pred_alpha)
    plt.subplot(2,2,3)
    show_mask_region("Conn Eval Mask", eval_mask)
    plt.subplot(2,2,4)
    show_mask_region("Conn Error Map", bce_map)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 示例路径，可不提供 trimap
    gt_path = "../data/video_defocused_processed/train/alpha/0000/frames/0140.png"
    pred_path = "../data/video_defocused_processed/train/fgr/0000/mask/0140.png"
    trimap_path = None
    coarse_path = None

    to_tensor = T.ToTensor()
    gt_alpha = to_tensor(Image.open(gt_path).convert("L")).unsqueeze(0)
    pred_alpha = to_tensor(Image.open(pred_path).convert("L")).unsqueeze(0)
    trimap = to_tensor(Image.open(trimap_path).convert("L")).unsqueeze(0) if trimap_path else None
    coarse = to_tensor(Image.open(coarse_path).convert("L")).unsqueeze(0) if coarse_path else None

    # Loss 可视
    visualize_cf_loss_region(gt_alpha, pred_alpha, trimap)
    visualize_grad_loss_region(gt_alpha, pred_alpha, coarse)
    visualize_conn_loss_region(gt_alpha, pred_alpha)
    visualize_sad_loss_region(gt_alpha, pred_alpha)
    visualize_ctr_loss_region(gt_alpha, pred_alpha)

    # Metrics 可视
    visualize_mse_metric_region(gt_alpha, pred_alpha, trimap)
    visualize_sad_metric_region(gt_alpha, pred_alpha, trimap)
    visualize_grad_metric_region(gt_alpha, pred_alpha, trimap)
    visualize_conn_metric_region(gt_alpha, pred_alpha, trimap)
