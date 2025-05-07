import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# 项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
from model.refiner_sanity import RefinerWithErrorMap

# ---------- 边缘 loss ----------
def compute_edge_loss(pred, gt):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)

# ---------- 可视化 ----------
def save_visualization(rgb, m_init, m_gt, m_pred, static_error_map, pred_error_map, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    def to_numpy(x): return x.detach().cpu().squeeze().numpy()

    fig, axs = plt.subplots(1, 6, figsize=(24, 4))
    axs[0].imshow(to_numpy(rgb).transpose(1, 2, 0)); axs[0].set_title("RGB")
    axs[1].imshow(to_numpy(m_init), cmap='gray'); axs[1].set_title("Init Mask")
    axs[2].imshow(to_numpy(m_gt), cmap='gray'); axs[2].set_title("GT Matte")
    axs[3].imshow(to_numpy(m_pred), cmap='gray'); axs[3].set_title("Pred Matte")
    axs[4].imshow(to_numpy(static_error_map), cmap='hot'); axs[4].set_title("Static Error Map")
    axs[5].imshow(to_numpy(pred_error_map), cmap='hot'); axs[5].set_title("Pred Error Map")
    for ax in axs: ax.axis('off')
    plt.tight_layout(); fig.savefig(save_path); plt.close(fig)

# ---------- 训练主函数 ----------
def train_with_error_map(model, loader, optimizer, scaler, device, epoch,
                         print_interval=20, accum_steps=4, vis_dir="vis"):
    model.train()
    optimizer.zero_grad()

    pad_w = torch.nn.ReplicationPad2d((0, 1, 0, 0))
    pad_h = torch.nn.ReplicationPad2d((0, 0, 0, 1))

    for i, batch in enumerate(loader):
        rgb, init_mask, gt = [x.to(device) for x in batch]

        # Static error map from GT and init
        static_error = (gt - init_mask).abs()

        dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        dx = pad_w(dx)
        dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        dy = pad_h(dy)
        edge_map = (dx.abs() + dy.abs()).clamp(0, 1)
        edge_map = F.avg_pool2d(edge_map, 3, stride=1, padding=1)
        static_error_map = (static_error + 0.5 * edge_map).clamp(0, 1)

        with autocast():
            m_pred = model(rgb, init_mask, static_error_map)

            # Main loss
            fg_mask = (gt > 0.05).float()
            bg_mask = 1.0 - fg_mask
            weight = 1.0 * fg_mask + 0.3 * bg_mask
            loss_main = F.l1_loss(m_pred, gt, reduction='none')
            loss_main = (loss_main * weight).mean()

            # Dynamic pred error
            pred_error = (m_pred.detach() - gt).abs()  # no grad
            combined_mask = (static_error > 0.05).float() * pred_error  # highlight not-yet-fixed areas
            loss_structure = F.l1_loss(m_pred * combined_mask, gt * combined_mask)

            # Edge loss
            loss_edge = compute_edge_loss(m_pred, gt)

            structure_weight = 10.0 if epoch <= 3 else 5.0
            loss = (loss_main + structure_weight * loss_structure + 0.3 * loss_edge) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if (i + 1) % print_interval == 0:
            print(f"[Epoch {epoch:02d} Step {i+1:04d}] "
                  f"Loss: {loss.item():.4f} | Pred mean: {m_pred.mean().item():.4f}, "
                  f"min/max: {m_pred.min().item():.4f}/{m_pred.max().item():.4f}")

            save_path = os.path.join(vis_dir, f"epoch{epoch}_step{i+1}.png")
            save_visualization(rgb[0], init_mask[0], gt[0], m_pred[0],
                               static_error_map[0], pred_error[0], save_path)

# ---------- 启动入口 ----------
def main():
    config = {
        "num_epochs": 3,
        "batch_size": 1,
        "print_interval": 20,
        "checkpoint_dir": "checkpoints",
        "csv_path": "../data/pair_for_refiner.csv",
        "resize_to": (736, 1280),
        "num_workers": 6,
        "lr": 5e-4,
        "weight_decay": 0,
        "accum_steps": 4,
    }

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = build_dataloaders(
        csv_path=config["csv_path"],
        resize_to=config["resize_to"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        sample_fraction=1.0,
    )

    model = RefinerWithErrorMap(base_channels=64).to(device)
    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{config['num_epochs']} ---")
        train_with_error_map(model, train_loader, optimizer, scaler, device, epoch,
                             print_interval=config["print_interval"],
                             accum_steps=config["accum_steps"])

        ckpt_path = os.path.join(config["checkpoint_dir"], f"refiner_error_map_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()
