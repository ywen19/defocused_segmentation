import os
import sys
import json
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms.functional import to_tensor, resize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
from model.refiner import MatteRefiner
from utils import compute_mse, compute_sad, compute_grad, compute_conn
from utils import matte_l1_loss, mask_guided_loss, edge_aware_loss


def write_log(log_path, data):
    with open(log_path, "a") as f:
        f.write(json.dumps(data) + "\n")


def train_one_epoch(model, loader, optimizer, scaler, device, epoch,
                    print_interval=20, log_path="train_metrics.jsonl", accum_steps=2):
    model.train()
    total_loss = 0
    mse_total, sad_total, grad_total, conn_total = 0.0, 0.0, 0.0, 0.0
    sample_count = 0

    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        rgb, init_mask, gt = [x.to(device) for x in batch]
        bs = rgb.size(0)

        with autocast():
            pred, attn = model(rgb, init_mask)
            assert pred.shape == gt.shape, f"Pred shape {pred.shape} != GT shape {gt.shape}"

            loss_l1 = matte_l1_loss(pred, gt)
            loss_edge = edge_aware_loss(pred, gt)
            loss_mask = mask_guided_loss(pred, init_mask) if epoch <= 2 else 0.0
            loss = loss_l1 + 0.5 * loss_edge + 0.3 * loss_mask
            loss = loss / accum_steps  # scale loss

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * bs * accum_steps
        mse_total += compute_mse(pred, gt).mean().item() * bs
        sad_total += compute_sad(pred, gt, reduction='mean').item()
        grad_total += compute_grad(pred, gt, reduction='mean').item() * bs
        conn_total += compute_conn(pred, gt).mean().item() * bs
        sample_count += bs

        if (i + 1) % print_interval == 0:
            print(f"[Train {i+1:04d}/{len(loader)}] "
                  f"Loss: {total_loss/sample_count:.4f} | "
                  f"MSE: {mse_total/sample_count:.6f} | "
                  f"SAD: {sad_total/sample_count:.2f} | "
                  f"GRAD: {grad_total/sample_count:.4f} | "
                  f"CONN: {conn_total/sample_count:.4f}")
            write_log(log_path, {
                "epoch": epoch,
                "step": i + 1,
                "loss": total_loss / sample_count,
                "mse": mse_total / sample_count,
                "sad": sad_total / sample_count,
                "grad": grad_total / sample_count,
                "conn": conn_total / sample_count,
            })


@torch.no_grad()
def val_one_epoch(model, loader, device, epoch,
                  print_interval=10, log_path="val_metrics.jsonl"):
    model.eval()
    total_loss, mse_total, sad_total, grad_total, conn_total = 0.0, 0.0, 0.0, 0.0, 0.0
    sample_count = 0

    for i, batch in enumerate(loader):
        rgb, init_mask, gt = [x.to(device) for x in batch]
        bs = rgb.size(0)

        with autocast():
            pred, attn = model(rgb, init_mask)
            pred = torch.clamp(pred, 0, 1)
            loss = matte_l1_loss(pred, gt)

        total_loss += loss.item() * bs
        mse_total += compute_mse(pred, gt).mean().item() * bs
        sad_total += compute_sad(pred, gt, reduction='mean').item()
        grad_total += compute_grad(pred, gt, reduction='mean').item() * bs
        conn_total += compute_conn(pred, gt).mean().item() * bs
        sample_count += bs

        if (i + 1) % print_interval == 0:
            print(f"[Val {i+1:04d}/{len(loader)}] "
                  f"Loss: {total_loss/sample_count:.4f} | "
                  f"MSE: {mse_total/sample_count:.6f} | "
                  f"SAD: {sad_total/sample_count:.2f} | "
                  f"GRAD: {grad_total/sample_count:.4f} | "
                  f"CONN: {conn_total/sample_count:.4f}")
            write_log(log_path, {
                "epoch": epoch,
                "step": i + 1,
                "loss": total_loss / sample_count,
                "mse": mse_total / sample_count,
                "sad": sad_total / sample_count,
                "grad": grad_total / sample_count,
                "conn": conn_total / sample_count,
            })

    return (
        total_loss / sample_count,
        mse_total / sample_count,
        sad_total / sample_count,
        grad_total / sample_count,
        conn_total / sample_count,
    )


def main():
    config = {
        "num_epochs": 1,
        "batch_size": 4,
        "print_interval": 20,
        "checkpoint_dir": "checkpoints",
        "csv_path": "../data/pair_for_refiner.csv",
        "resize_to": (1088, 1920),
        "num_workers": 6,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "base_channel": 48,
        "sample_fraction": 0.05,
    }

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        csv_path=config["csv_path"],
        resize_to=config["resize_to"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        sample_fraction=config["sample_fraction"],
    )

    model = MatteRefiner(base_channels=config['base_channel']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = GradScaler()

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{config['num_epochs']} ---")
        train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, accum_steps=2)
        val_loss, val_mse, val_sad, val_grad, val_conn = val_one_epoch(model, val_loader, device, epoch)

        print(f"Validation Metrics: "
              f"Loss={val_loss:.4f} | MSE={val_mse:.6f} | "
              f"SAD={val_sad:.2f} | GRAD={val_grad:.4f} | CONN={val_conn:.4f}")

        ckpt_path = os.path.join(config["checkpoint_dir"], f"refiner_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
