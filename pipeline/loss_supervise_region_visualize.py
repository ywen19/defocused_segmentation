import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


def get_gradient(tensor):
    """
    使用 Sobel 样式卷积计算梯度，保持输入输出尺寸一致。
    """
    kernel_x = torch.tensor([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    kernel_x = kernel_x.to(tensor.device)
    kernel_y = kernel_y.to(tensor.device)

    grad_x = F.conv2d(tensor, kernel_x, padding=1)
    grad_y = F.conv2d(tensor, kernel_y, padding=1)
    return grad_x.abs(), grad_y.abs()


def visualize_grad_loss_region(gt_alpha, coarse_mask=None, lower=0.05, upper=0.95):
    """
    可视化 gradient loss 在由 GT alpha 定义的 soft matte 区域的作用。
    """
    gt_alpha = gt_alpha.clone()
    if coarse_mask is None:
        coarse_mask = gt_alpha  # fallback

    # 仅使用 GT 决定 soft 区域（这样对 edge 模糊有更好体现）
    soft_mask = ((gt_alpha > lower) & (gt_alpha < upper)).float()

    # 梯度图必须在 full gt_alpha 上计算，再用 soft mask 提取目标区域
    grad_x_all, grad_y_all = get_gradient(gt_alpha)
    grad_x = grad_x_all * soft_mask
    grad_y = grad_y_all * soft_mask

    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(gt_alpha.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title("GT Alpha")

    axes[1].imshow(coarse_mask.squeeze().cpu().numpy(), cmap='gray')
    axes[1].set_title("Coarse Mask")

    axes[2].imshow(grad_x.squeeze().cpu().numpy(), cmap='viridis')
    axes[2].set_title("Grad X (GT Soft Region)")

    axes[3].imshow(grad_y.squeeze().cpu().numpy(), cmap='viridis')
    axes[3].set_title("Grad Y (GT Soft Region)")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_cf_loss_region(gt_alpha, pred_alpha, lower=0.05, upper=0.95):
    """
    可视化 CF Loss 的监督区域和误差分布，仅在 soft matte 区域内。
    """
    def create_soft_mask(gt_alpha, lower=0.05, upper=0.95):
        return ((gt_alpha > lower) & (gt_alpha < upper)).float()

    def compute_cf_loss(pred, target, mask, eps=1e-6):
        # 计算平方差
        loss_map = (pred - target) ** 2
        masked_loss = loss_map * mask
        loss_mean = masked_loss.sum() / (mask.sum() + eps)
        return masked_loss.squeeze(0).squeeze(0).detach(), loss_mean.item()

    # 创建 soft matte 区域掩码
    soft_mask = create_soft_mask(gt_alpha, lower, upper)

    # 计算 CF Loss 误差图（仅在 soft 区域）
    loss_map, loss_value = compute_cf_loss(pred_alpha, gt_alpha, soft_mask)

    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(gt_alpha.squeeze().numpy(), cmap='gray')
    axes[0].set_title("GT Alpha")

    axes[1].imshow(pred_alpha.squeeze().numpy(), cmap='gray')
    axes[1].set_title("Pred Alpha")

    axes[2].imshow(soft_mask.squeeze().numpy(), cmap='hot')
    axes[2].set_title("Soft Matte Region")

    axes[3].imshow(loss_map.numpy(), cmap='inferno')
    axes[3].set_title(f"CF Loss (Soft Region)\nMean: {loss_value:.4f}")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_ctr_loss_region(gt_alpha, coarse_mask=None, lower=0.05, upper=0.95):
    """
    可视化 CTR Loss 监督的 soft 区域（GT 决定 + optional coarse mask refine）
    """
    # soft 区域：GT 决定
    soft_mask = ((gt_alpha > lower) & (gt_alpha < upper)).float()
    if coarse_mask is not None:
        soft_mask *= ((coarse_mask > lower) & (coarse_mask < upper)).float()

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gt_alpha.squeeze().numpy(), cmap='gray')
    axes[0].set_title("GT Alpha")

    axes[1].imshow(coarse_mask.squeeze().numpy(), cmap='gray')
    axes[1].set_title("Coarse Mask")

    axes[2].imshow(soft_mask.squeeze().numpy(), cmap='hot')
    axes[2].set_title("CTR Loss Region (Soft, Refined)")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_ctr_loss_region(gt_alpha, pred_alpha=None, coarse_mask=None, threshold=0.5):
    """
    可视化 CTR Loss 监督区域（用于粗粒度的区域修正，不使用 soft matte 掩码）。
    """
    # 二值化 GT 和 coarse mask（或 prediction）
    gt_binary = (gt_alpha > threshold).float()
    
    if pred_alpha is None and coarse_mask is None:
        raise ValueError("必须提供 pred_alpha 或 coarse_mask 之一用于比较")

    pred_binary = (pred_alpha if pred_alpha is not None else coarse_mask) > threshold
    pred_binary = pred_binary.float()

    # 差异区域（预测与 GT 不一致的位置）
    diff = torch.abs(pred_binary - gt_binary)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(gt_alpha.squeeze().numpy(), cmap='gray')
    axes[0].set_title("GT Alpha")

    if coarse_mask is not None:
        axes[1].imshow(coarse_mask.squeeze().numpy(), cmap='gray')
        axes[1].set_title("Coarse Mask")
    else:
        axes[1].imshow(pred_alpha.squeeze().numpy(), cmap='gray')
        axes[1].set_title("Prediction")

    axes[2].imshow(diff.squeeze().numpy(), cmap='hot')
    axes[2].set_title("CTR Loss Region (Binary Diff)")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def get_conn_map(alpha, threshold=0.5):
    """
    获取 alpha matte 的连通性 map（二值+连通区域标记）
    """
    alpha_np = alpha.squeeze().cpu().numpy()
    binary = (alpha_np > threshold).astype(np.uint8)
    labeled = label(binary, connectivity=1)
    return binary, labeled

def visualize_conn_loss_region_simple(gt_alpha, pred_alpha, threshold=0.5):
    """
    简化版的 Conn Loss 可视化：比较 GT 和预测的连通区域差异（不使用 skimage）。
    """
    # 二值化处理
    gt_bin = (gt_alpha > threshold).float()
    pred_bin = (pred_alpha > threshold).float()

    # 差异区域：GT 是前景但 Pred 不是，或者 Pred 是前景但 GT 不是
    mismatch = torch.abs(gt_bin - pred_bin)

    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gt_alpha.squeeze().numpy(), cmap='gray')
    axes[0].set_title("GT Alpha")

    axes[1].imshow(pred_alpha.squeeze().numpy(), cmap='gray')
    axes[1].set_title("Pred Alpha")

    axes[2].imshow(gt_bin.squeeze().numpy(), cmap='gray')
    axes[2].set_title("GT Binary Mask")

    axes[3].imshow(mismatch.squeeze().numpy(), cmap='hot')
    axes[3].set_title("Mismatch (Conn Loss Region)")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_sad_loss_region(gt_alpha, pred_alpha):
    """
    可视化 SAD Loss 区域：直接展示预测与 GT 的逐像素绝对差。
    """
    # 计算 SAD 误差图
    abs_diff = torch.abs(pred_alpha - gt_alpha)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(gt_alpha.squeeze().numpy(), cmap='gray')
    axes[0].set_title("GT Alpha")

    axes[1].imshow(pred_alpha.squeeze().numpy(), cmap='gray')
    axes[1].set_title("Pred Alpha")

    axes[2].imshow(abs_diff.squeeze().numpy(), cmap='hot')
    axes[2].set_title("SAD Loss Map (|Pred - GT|)")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # === 路径替换为你自己的 ===
    gt_path = "../data/video_defocused_processed/train/alpha/0006/frames/0132.png"
    coarse_path = "../data/video_defocused_processed/train/fgr/0006/mask/0132.png"
    # gt_path = "../data/video_defocused_processed/train/alpha/0028/frames/0377.png"
    # coarse_path = "../data/video_defocused_processed/train/fgr/0028/mask/0377.png"

    to_tensor = T.ToTensor()
    gt_alpha = to_tensor(Image.open(gt_path).convert("L")).unsqueeze(0)  # (1, 1, H, W)
    coarse_mask = to_tensor(Image.open(coarse_path).convert("L")).unsqueeze(0)

    # visualize_grad_loss_region(gt_alpha, coarse_mask)
    visualize_cf_loss_region(gt_alpha, coarse_mask)
    # visualize_ctr_loss_region(gt_alpha, coarse_mask)
    # visualize_conn_loss_region_simple(gt_alpha, coarse_mask)
    # visualize_sad_loss_region(gt_alpha, coarse_mask)

