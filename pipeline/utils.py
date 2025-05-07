import torch
import torch.nn.functional as F


def extract_unknown_mask(trimap, strategy='auto', threshold=20):
    """
    提取 unknown 区域的掩码（适用于不同格式的 trimap 或 alpha matte）

    :param trimap: [B, 1, H, W] or [B, H, W] tensor, dtype = uint8 or float
                   - 可传入真实 trimap（三值），也可传入 alpha matte（自动估计 unknown）
    :param strategy: 'auto' | 'fixed' | 'range'
    :param threshold: 用于 'range' 模式，表示 [128 - t, 128 + t] 或 [0.5 - t, 0.5 + t]
    :return: unknown 区域掩码（float 类型，[B, 1, H, W]）
    """
    if trimap.ndim == 3:
        trimap = trimap.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

    is_float = trimap.dtype in [torch.float32, torch.float64]

    if strategy == 'fixed':
        unknown_mask = (trimap == 128).float()

    elif strategy == 'range':
        if is_float:
            lower = 0.5 - threshold
            upper = 0.5 + threshold
        else:
            lower = 128 - threshold
            upper = 128 + threshold
        unknown_mask = ((trimap >= lower) & (trimap <= upper)).float()

    elif strategy == 'auto':
        unique_vals = torch.unique(trimap)
        if unique_vals.numel() == 3:
            # 常见三值trimap
            sorted_vals = unique_vals.sort()[0]
            unknown_val = sorted_vals[1]
            unknown_mask = (trimap == unknown_val).float()
        else:
            # fallback: 估计自 alpha matte
            if is_float:
                unknown_mask = ((trimap > 0.1) & (trimap < 0.9)).float()
            else:
                unknown_mask = ((trimap > 25) & (trimap < 230)).float()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return unknown_mask


################################################ metrics ################################################

def compute_mse_with_trimap(pred, gt, trimap):
    unknown_mask = extract_unknown_mask(trimap)  # 使用我们之前写的函数
    diff = (pred - gt) ** 2 * unknown_mask
    mse_per_image = diff.sum(dim=[1, 2, 3]) / (unknown_mask.sum(dim=[1,2,3]) + 1e-6)
    return mse_per_image.mean()


def compute_sad_with_trimap(pred, gt, trimap, normalize=True):
    unknown_mask = extract_unknown_mask(trimap, strategy='auto')  # 更通用
    diff = torch.abs(pred - gt) * unknown_mask
    sad_per_image = diff.sum(dim=[1, 2, 3])

    if normalize:
        area = unknown_mask.sum(dim=[1, 2, 3])
        sad_per_image = sad_per_image / (area + 1e-6) * 1000.0

    return sad_per_image.mean()


def compute_conn_with_unknown_mask(pred, gt, trimap, threshold=0.5):
    """
    Trimap-aware Connectivity Loss（Conn）:
    - 使用 GT 中最大连通前景区域作为监督目标
    - 只在 trimap 的 unknown 区域内计算差异（结构敏感）
    
    Args:
        pred: [B, 1, H, W] 预测 alpha matte
        gt: [B, 1, H, W] ground-truth alpha
        trimap: [B, 1, H, W] or [B, H, W]，三值或 soft trimap
        threshold: 连通性二值化阈值，默认 0.5
    Returns:
        mean Conn loss over batch
    """
    B, C, H, W = pred.shape
    assert C == 1

    unknown_mask = extract_unknown_mask(trimap)  # 统一提取 unknown 区域
    losses = []

    for i in range(B):
        gt_bin = (gt[i, 0].detach().cpu().numpy() > threshold).astype(np.uint8)

        # 获取最大连通区域
        labeled, num = connected_components(gt_bin, return_num=True)
        if num == 0:
            losses.append(torch.tensor(0.0, device=pred.device))
            continue

        max_region = 0
        max_area = 0
        for j in range(1, num + 1):
            area = np.sum(labeled == j)
            if area > max_area:
                max_area = area
                max_region = j

        gt_main_mask = (labeled == max_region).astype(np.float32)  # [H, W]
        gt_main_mask_tensor = torch.from_numpy(gt_main_mask).to(pred.device).unsqueeze(0).unsqueeze(0)

        # 限定在 unknown 区域上计算
        eval_mask = (gt_main_mask_tensor * unknown_mask[i:i+1]).clamp(0, 1)

        pred_crop = pred[i:i+1]
        target_crop = gt_main_mask_tensor

        bce = F.binary_cross_entropy(pred_crop * eval_mask, target_crop * eval_mask, reduction='sum')
        norm = eval_mask.sum()
        if norm < 1:
            norm = torch.tensor(1.0, device=pred.device)
        losses.append(bce / norm)

    return torch.stack(losses).mean()


def compute_grad_metric(pred_alpha, gt_alpha, trimap, eps=1e-6):
    """
    Gradient Metric（benchmark 版评估指标）:
    - 在 trimap 的 unknown 区域上计算
    - 分别计算 x、y 方向梯度的平均绝对差
      grad_error = (sum(|∇x_pred - ∇x_gt|) + sum(|∇y_pred - ∇y_gt|))
                   / (2 * num_unknown_pixels)
    Returns:
        mean gradient error over batch（标量）
    """
    def get_gradient_xy(x):
        kernel_x = torch.tensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        kernel_y = torch.tensor([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        gx = F.conv2d(x, kernel_x, padding=1)
        gy = F.conv2d(x, kernel_y, padding=1)
        return gx, gy

    # 1. 计算预测和 GT 的 x、y 梯度
    pred_gx, pred_gy = get_gradient_xy(pred_alpha)
    gt_gx,   gt_gy   = get_gradient_xy(gt_alpha)

    # 2. 提取 unknown 区域掩码
    unknown_mask = extract_unknown_mask(trimap, strategy='auto')  # [B,1,H,W]

    # 3. 计算各方向的绝对差，并限制在 unknown 区域
    diff_x = torch.abs(pred_gx - gt_gx) * unknown_mask
    diff_y = torch.abs(pred_gy - gt_gy) * unknown_mask

    # 4. 对每张图分别求和并除以 (2 * unknown_area)
    area = unknown_mask.sum(dim=[1,2,3]) + eps
    per_img_error = (diff_x.sum(dim=[1,2,3]) + diff_y.sum(dim=[1,2,3])) / (2 * area)

    # 5. 批量取平均
    return per_img_error.mean()



################################################ loss ################################################

def compute_grad_loss_soft_only(pred_alpha, gt_alpha, coarse_mask=None, lower=0.05, upper=0.95):
    """
    Blur-aware Gradient Loss（不使用 trimap）:
    - 仅在 GT 的 soft matte 区域上监督
    - 可选进一步与 coarse mask 的 soft 区域相交
    - 使用 GT 的边缘梯度作为权重，引导关注虚化边缘
    """

    def get_gradient(x):
        kernel_x = torch.tensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        kernel_y = torch.tensor([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        grad_x = F.conv2d(x, kernel_x, padding=1)
        grad_y = F.conv2d(x, kernel_y, padding=1)
        return grad_x, grad_y

    eps = 1e-6

    # === 梯度计算 ===
    pred_grad_x, pred_grad_y = get_gradient(pred_alpha)
    gt_grad_x, gt_grad_y = get_gradient(gt_alpha)

    # === soft matte 区域（GT中alpha ∈ (0.05, 0.95)） ===
    soft_mask = ((gt_alpha > lower) & (gt_alpha < upper)).float()

    # 可选 coarse mask 限制
    if coarse_mask is not None:
        soft_mask *= ((coarse_mask > lower) & (coarse_mask < upper)).float()

    # === blur-aware 部分 ===
    edge_strength = torch.sqrt(gt_grad_x ** 2 + gt_grad_y ** 2 + eps)
    edge_weight = edge_strength / (edge_strength.max() + eps)

    # 组合权重
    weighted_mask = soft_mask * edge_weight

    # === 计算带权梯度差 ===
    diff_x = torch.abs(pred_grad_x - gt_grad_x) * weighted_mask
    diff_y = torch.abs(pred_grad_y - gt_grad_y) * weighted_mask

    loss_x = diff_x.sum() / (weighted_mask.sum() + eps)
    loss_y = diff_y.sum() / (weighted_mask.sum() + eps)

    return (loss_x + loss_y) / 2



def compute_conn_mismatch_loss(pred_alpha, gt_alpha, threshold=0.5):
    """
    Conn Loss: 在 GT 和预测二值化结果 mismatch 的区域上，计算 BCE。
    该实现与 visualize_conn_loss_region_simple 中的监督方式一致。
    """
    gt_bin = (gt_alpha > threshold).float()
    pred_bin = (pred_alpha > threshold).float()
    mismatch = torch.abs(gt_bin - pred_bin)  # (B, 1, H, W)

    bce = F.binary_cross_entropy(pred_alpha, gt_bin, reduction='none')
    masked_bce = bce * mismatch

    return masked_bce.sum() / (mismatch.sum() + 1e-6)


def compute_cf_loss(pred_alpha, gt_alpha, trimap=None, use_trimap=False,
                    lower=0.05, upper=0.95, eps=1e-6):
    """
    Closed-form Loss:
    只在 GT 的 soft matte 区域（可选限定在 trimap unknown 区域）上监督 pred_alpha。
    default do not restrict to trimap.

    Args:
        pred_alpha: [B, 1, H, W]，模型预测的 alpha matte
        gt_alpha: [B, 1, H, W]，ground-truth matte
        trimap: [B, 1, H, W] 或 [B, H, W]，可选，表示三值 trimap
        use_trimap: bool，是否启用 trimap unknown 区域限制
        lower, upper: float，soft matte 阈值范围
        eps: 防除零小常数
    Returns:
        平均 masked MSE loss
    """
    # 定义 soft matte 区域（GT 中 alpha ∈ (lower, upper)）
    soft_mask = ((gt_alpha > lower) & (gt_alpha < upper)).float()

    # 如果启用 trimap 限制，进一步取交集
    if use_trimap and trimap is not None:
        unknown_mask = extract_unknown_mask(trimap, strategy='auto')  # ✅ 使用已有函数
        soft_mask *= unknown_mask

    # MSE loss 计算
    loss_map = (pred_alpha - gt_alpha) ** 2
    masked_loss = loss_map * soft_mask
    return masked_loss.sum() / (soft_mask.sum() + eps)



def compute_ctr_loss(anchor, positive, negative, gt_alpha, pred_alpha=None, coarse_mask=None, threshold=0.5, margin=0.5):
    """
    计算基于 GT 和预测差异区域的对比损失（CTR Loss）。
    
    Args:
        anchor: Tensor (B, C, H, W)
        positive: Tensor (B, C, H, W)
        negative: Tensor (B, C, H, W)
        gt_alpha: Tensor (B, 1, H, W) - Ground truth alpha matte
        pred_alpha: Tensor (B, 1, H, W), optional - 模型预测 alpha
        coarse_mask: Tensor (B, 1, H, W), optional - 粗糙分割
        threshold: float - 二值化阈值
        margin: float - 对比损失边界
        
    Returns:
        ctr_loss: scalar tensor
    """
    assert (pred_alpha is not None) or (coarse_mask is not None), "必须提供 pred_alpha 或 coarse_mask"

    # 1. 获取监督区域掩码（GT 与预测/coarse 的差异区域）
    gt_bin = (gt_alpha > threshold).float()
    pred_or_coarse = pred_alpha if pred_alpha is not None else coarse_mask
    pred_bin = (pred_or_coarse > threshold).float()
    diff_mask = torch.abs(gt_bin - pred_bin)  # (B, 1, H, W)

    # 2. 计算每个位置的 cos 距离
    pos_sim = F.cosine_similarity(anchor, positive, dim=1, eps=1e-6)  # (B, H, W)
    neg_sim = F.cosine_similarity(anchor, negative, dim=1, eps=1e-6)  # (B, H, W)

    # 3. 应用掩码（仅在区域差异处计算 loss）
    loss = F.relu(neg_sim - pos_sim + margin)
    loss = loss * diff_mask.squeeze(1)  # (B, H, W)

    return loss.sum() / (diff_mask.sum() + 1e-6)


def compute_sad_loss(pred_alpha, gt_alpha, lower=0.1, upper=0.9):
    """
    计算不依赖 trimap 的 SAD Loss
    - 使用 `gt_alpha` 的 soft 区域进行监督
    - 计算 `abs(pred_alpha - gt_alpha)` 仅在 soft 区域内进行
    """
    # 提取 soft 区域 (alpha ∈ (0.1, 0.9))
    soft_mask = ((gt_alpha > lower) & (gt_alpha < upper)).float()

    # 计算绝对误差
    abs_diff = torch.abs(pred_alpha - gt_alpha)

    # 计算 loss map（仅在 soft 区域内）
    loss_map = abs_diff * soft_mask

    # 返回最终的 loss（平均）
    return loss_map.sum() / (soft_mask.sum() + 1e-6)