import os
import sys
import gc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

from pytorch_wavelets import DWTForward

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
from model.refiner_sanity_dualbranch_V2 import RefinerWithDualBranch

wavelet = DWTForward(J=1, mode='zero', wave='haar')

def compute_edge_loss(pred, gt):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)

def save_visualization(rgb, m_init, m_gt, m_pred, static_error_map, pred_error_map, corr_fp, corr_fn, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    def to_numpy(x): return x.detach().cpu().squeeze().numpy()
    fig, axs = plt.subplots(1, 6, figsize=(32, 4))
    axs[0].imshow(to_numpy(rgb).transpose(1, 2, 0)); axs[0].set_title("RGB")
    axs[1].imshow(to_numpy(m_init), cmap='gray'); axs[1].set_title("Init Mask")
    axs[2].imshow(to_numpy(m_gt), cmap='gray'); axs[2].set_title("GT Matte")
    axs[3].imshow(to_numpy(m_pred), cmap='gray'); axs[3].set_title("Pred Matte")
    axs[4].imshow(to_numpy(static_error_map), cmap='hot'); axs[4].set_title("Static Error Map")
    axs[5].imshow(to_numpy(pred_error_map), cmap='hot'); axs[5].set_title("Pred Error Map")
    for ax in axs: ax.axis('off')
    plt.tight_layout(); fig.savefig(save_path); plt.close(fig)

def train_with_error_map(model, loader, optimizer, scaler, device, epoch,
                         print_interval=20, accum_steps=4, vis_dir="vis2",
                         teacher_model=None):
    model.train()
    optimizer.zero_grad()

    pad_w = torch.nn.ReplicationPad2d((0, 1, 0, 0))
    pad_h = torch.nn.ReplicationPad2d((0, 0, 0, 1))

    laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    wavelet.to(device)

    for i, batch in enumerate(loader):
        rgb, init_mask, gt = [x.to(device) for x in batch]

        decay_weight = max(0.2, 1.0 - epoch / 10.0)
        mask_guidance = torch.where(init_mask > 0.99, init_mask, init_mask * decay_weight)

        static_error = (gt - init_mask).abs()
        dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]; dx = pad_w(dx)
        dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]; dy = pad_h(dy)
        edge_map = (dx.abs() + dy.abs()).clamp(0, 1)
        edge_map = F.avg_pool2d(edge_map, 3, stride=1, padding=1)
        static_error_map = (static_error + 0.5 * edge_map).clamp(0, 1)

        with autocast():
            m_pred, corr_fp, corr_fn = model(rgb, mask_guidance, static_error_map)

            if epoch <= 3:
                fp_mask = ((init_mask > 0.5) & (gt < 0.5)).float()
                fn_mask = ((init_mask < 0.5) & (gt > 0.5)).float()
            else:
                with torch.no_grad():
                    fp_mask = ((m_pred.detach() > 0.5) & (gt < 0.5)).float()
                    fn_mask = ((m_pred.detach() < 0.5) & (gt > 0.5)).float()

            m_pred, corr_fp, corr_fn = model(rgb, mask_guidance, static_error_map, fp_mask, fn_mask)
            m_pred = m_pred.clamp(0, 1)

            fg_mask = (gt > 0.05).float()
            bg_mask = 1.0 - fg_mask
            weight = 1.0 * fg_mask + 0.3 * bg_mask
            loss_l1 = (F.l1_loss(m_pred, gt, reduction='none') * weight).mean()

            loss_grad = compute_edge_loss(m_pred, gt)
            lap_gt = F.conv2d(gt, laplace_kernel, padding=1)
            lap_pred = F.conv2d(m_pred, laplace_kernel, padding=1)
            loss_laplace = F.l1_loss(lap_pred, lap_gt)

            gray_mask = ((gt > 0.1) & (gt < 0.9)).float()
            loss_gray = (F.mse_loss(m_pred, gt, reduction='none') * gray_mask).mean()

        with torch.cuda.amp.autocast(enabled=False):
            _, gt_H = wavelet(gt)
            _, pred_H = wavelet(m_pred)
            loss_wavelet = sum(F.l1_loss(ph, gh) for ph, gh in zip(pred_H, gt_H)) / len(gt_H)

        loss_smooth = (
            F.l1_loss(m_pred[:, :, :, :-1], m_pred[:, :, :, 1:]) +
            F.l1_loss(m_pred[:, :, :-1, :], m_pred[:, :, 1:, :])
        )

        loss_fp = F.l1_loss(corr_fp, (init_mask - gt).clamp(min=0))
        loss_fn = F.l1_loss(corr_fn, (gt - init_mask).clamp(min=0))

        loss = (
            1.0 * loss_l1 +
            0.5 * loss_laplace +
            0.5 * loss_grad +
            0.5 * loss_gray +
            0.5 * loss_wavelet +
            0.1 * loss_smooth +
            0.3 * loss_fp +
            0.3 * loss_fn
        ) / accum_steps

        pred_error = (m_pred.detach() - gt).abs()
        structure_mask = (1 - torch.exp(-static_error * 50)) * pred_error
        structure_weight = max(2.0, 10.0 * (0.95 ** epoch))
        loss_structure = F.l1_loss(m_pred * structure_mask, gt * structure_mask)
        loss += (structure_weight * loss_structure) / accum_steps

        if epoch >= 5 and teacher_model is not None:
            with torch.no_grad():
                teacher_pred, _, _ = teacher_model(rgb, mask_guidance, static_error_map)
            inverse_mask = 1.0 - structure_mask  # 重点学习非结构区域
            loss_distill = (F.mse_loss(m_pred, teacher_pred, reduction='none') * inverse_mask).mean()
            loss += 0.3 * loss_distill / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if (i + 1) % print_interval == 0:
            print(f"[Epoch {epoch:02d} Step {i+1:04d}] Loss: {loss.item():.4f} | Pred mean: {m_pred.mean().item():.4f}, min/max: {m_pred.min().item():.4f}/{m_pred.max().item():.4f}")
            save_path = os.path.join(vis_dir, f"epoch{epoch}_step{i+1}.png")
            save_visualization(rgb[0], init_mask[0], gt[0], m_pred[0], static_error_map[0], (m_pred[0] - gt[0]).abs(), corr_fp[0], corr_fn[0], save_path)

def main():
    config = {
        "num_epochs": 10,
        "batch_size": 2,
        "print_interval": 20,
        "checkpoint_dir": "checkpoints",
        "csv_path": "../data/pair_for_refiner.csv",
        "resize_to": (376, 1280),
        "num_workers": 6,
        "lr": 5e-4,
        "weight_decay": 0,
        "accum_steps": 4,
    }

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RefinerWithDualBranch(base_channels=64).to(device)
    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{config['num_epochs']} ---")

        teacher_model = None
        if epoch >= 3:
            teacher_model = RefinerWithDualBranch(base_channels=64).to(device)
            teacher_ckpt = os.path.join(config["checkpoint_dir"], f"refiner_wavelet_epoch{epoch - 1}.pth")
            if os.path.exists(teacher_ckpt):
                teacher_model.load_state_dict(torch.load(teacher_ckpt), strict=False)
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False
            else:
                print(f"⚠️ Warning: teacher checkpoint not found for epoch {epoch - 1}, skipping distillation.")

        train_loader, val_loader = build_dataloaders(
            csv_path=config["csv_path"],
            resize_to=config["resize_to"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            seed=42,
            epoch_seed=42 + epoch,
            shuffle=True,
            sample_fraction=50,
        )

        train_with_error_map(model, train_loader, optimizer, scaler, device, epoch,
                             print_interval=config["print_interval"],
                             accum_steps=config["accum_steps"],
                             teacher_model=teacher_model)

        ckpt_path = os.path.join(config["checkpoint_dir"], f"refiner_wavelet_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

        del train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
