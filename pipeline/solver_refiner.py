"""
Training, validation and inference pipeline for our proposed MatteRefiner.
"""

import os
import sys
import json
from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms.functional import to_tensor, resize
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from dataload.build_dataloaders import build_dataloaders
from model.refiner import MatteRefiner
from utils import compute_mae, compute_sad, compute_grad
from utils import matte_l1_loss, mask_guided_loss, edge_aware_loss


def write_log(log_path, data):
    with open(log_path, "a") as f:
        f.write(json.dumps(data) + "\n")


def train_one_epoch(model, loader, optimizer, scaler, device, epoch,
                    print_interval=20, log_path="train_metrics.jsonl"):
    model.train()
    total_loss = 0
    mae_total, sad_total, grad_total = 0, 0, 0
    count = 0

    for i, batch in enumerate(loader):
        rgb, init_mask, gt = [x.to(device) for x in batch]

        optimizer.zero_grad()
        with autocast():
            pred, attn = model(rgb, init_mask)
            loss_l1 = matte_l1_loss(pred, gt)
            loss_edge = edge_aware_loss(pred, gt)
            if epoch <= 2:
                loss_mask = mask_guided_loss(pred, init_mask)
            else:
                loss_mask = 0.0
            loss = loss_l1 + 0.5 * loss_edge + 0.3 * loss_mask

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        mae_total += compute_mae(pred, gt).item()
        sad_total += compute_sad(pred, gt).item()
        grad_total += compute_grad(pred, gt).item()
        count += 1

        if (i + 1) % print_interval == 0:
            avg_loss = total_loss / count
            avg_mae = mae_total / count
            avg_sad = sad_total / count
            avg_grad = grad_total / count
            print(f"[Train {i+1:04d}/{len(loader)}] "
                  f"Loss: {avg_loss:.4f} | MAE: {avg_mae:.4f} | SAD: {avg_sad:.2f} | GRAD: {avg_grad:.4f}")

            write_log(log_path, {
                "epoch": epoch,
                "step": i + 1,
                "loss": avg_loss,
                "mae": avg_mae,
                "sad": avg_sad,
                "grad": avg_grad,
            })


def val_one_epoch(model, loader, device, epoch,
                  print_interval=20, log_path="val_metrics.jsonl"):
    model.eval()
    mae_total, sad_total, grad_total = 0, 0, 0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            rgb, init_mask, gt = [x.to(device) for x in batch]

            with autocast():
                pred, attn = model(rgb, init_mask)
                pred = torch.clamp(pred, 0, 1)

            mae_total += compute_mae(pred, gt).item()
            sad_total += compute_sad(pred, gt).item()
            grad_total += compute_grad(pred, gt).item()
            count += 1

            if (i + 1) % print_interval == 0:
                avg_mae = mae_total / count
                avg_sad = sad_total / count
                avg_grad = grad_total / count
                print(f"[Val {i+1:04d}/{len(loader)}] "
                      f"MAE: {avg_mae:.4f} | SAD: {avg_sad:.2f} | GRAD: {avg_grad:.4f}")

                write_log(log_path, {
                    "epoch": epoch,
                    "step": i + 1,
                    "mae": avg_mae,
                    "sad": avg_sad,
                    "grad": avg_grad,
                })

    return mae_total / count, sad_total / count, grad_total / count


@torch.no_grad()
def infer_single(model, rgb_path, mask_path, device, resize_to=(720, 1280)):
    model.eval()
    rgb = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    rgb = resize(rgb, resize_to)
    mask = resize(mask, resize_to)

    rgb_tensor = to_tensor(rgb).unsqueeze(0).to(device)
    mask_tensor = to_tensor(mask).unsqueeze(0).to(device)

    pred, _ = model(rgb_tensor, mask_tensor)
    return pred.squeeze(0).cpu()


def main():
    # ---------- Config ----------
    config = {
        "num_epochs": 3,
        "batch_size": 4,
        "print_interval": 20,
        "checkpoint_dir": "checkpoints",
        "csv_path": "../data/pair_for_refiner.csv",  # TODO: Set correct path
        "resize_to": (720, 1280),
        "num_workers": 4,
        "lr": 1e-4,
        "weight_decay": 1e-4,
    }

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Data ----------
    train_loader, val_loader = build_dataloaders(
        csv_path=config["csv_path"],
        resize_to=config["resize_to"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    # ---------- Model ----------
    model = MatteRefiner().to(device)

    # ---------- Optimizer & Scaler ----------
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = GradScaler()

    # ---------- Training Loop ----------
    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{config['num_epochs']} ---")
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            print_interval=config["print_interval"],
            log_path="train_metrics.jsonl"
        )

        val_mae, val_sad, val_grad = val_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            print_interval=config["print_interval"],
            log_path="val_metrics.jsonl"
        )

        print(f"Validation Metrics: MAE={val_mae:.4f} | SAD={val_sad:.2f} | GRAD={val_grad:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(config["checkpoint_dir"], f"refiner_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
