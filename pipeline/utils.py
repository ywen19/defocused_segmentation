import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter

# ----------------------- Metrics -----------------------

def compute_mae(pred, target):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target))

def compute_sad(pred, target):
    """Sum of Absolute Differences, scaled for interpretability"""
    return torch.sum(torch.abs(pred - target)) / 1000.0  # scaled

def compute_grad(pred, target):
    """Gradient Loss using Laplacian operator"""
    def _laplacian(x):
        kernel = torch.tensor(
            [[[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]]],
            dtype=x.dtype, device=x.device
        ).unsqueeze(0)  # [1,1,3,3]
        return F.conv2d(x, kernel, padding=1)

    grad_pred = _laplacian(pred)
    grad_target = _laplacian(target)
    return torch.mean(torch.abs(grad_pred - grad_target))


# ----------------------- Losses -----------------------

def matte_l1_loss(pred, target):
    """Standard L1 loss"""
    return F.l1_loss(pred, target)

def edge_aware_loss(pred, target, sigma=1.0):
    """Edge-aware loss using difference-of-Gaussian"""
    def get_edge(x):
        return x - gaussian_blur(x, sigma)
    return F.l1_loss(get_edge(pred), get_edge(target))

def gaussian_blur(x, sigma):
    """Apply Gaussian blur to each sample in batch (1 channel assumed)"""
    x_np = x.detach().cpu().numpy()
    blurred = np.zeros_like(x_np)
    for i in range(x_np.shape[0]):
        blurred[i, 0] = gaussian_filter(x_np[i, 0], sigma=sigma)
    return torch.tensor(blurred, device=x.device, dtype=x.dtype)

def mask_guided_loss(pred, mask):
    """Loss that encourages output to not deviate much from initial binary mask"""
    return F.l1_loss(pred * mask, mask)

def save_or_show_matte(matte_tensor, save_path=None, show=False, cmap="gray", title="Predicted Matte"):
    """
    可视化或保存模型预测的 alpha matte。

    Args:
        matte_tensor (Tensor): shape [H, W] or [1, H, W]，范围 [0, 1]。
        save_path (str): 可选，保存路径（.png）。
        show (bool): 是否使用 matplotlib 显示。
        cmap (str): 显示颜色映射。
        title (str): 图像标题。
    """
    if matte_tensor.dim() == 3:
        matte_tensor = matte_tensor.squeeze(0)
    
    matte_image = to_pil_image(matte_tensor)

    if save_path:
        matte_image.save(save_path)

    if show:
        plt.imshow(matte_image, cmap=cmap)
        plt.title(title)
        plt.axis("off")
        plt.show()
