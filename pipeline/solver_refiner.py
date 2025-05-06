import os
import sys
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
from model.refiner import MatteRefiner
from utils import (
    compute_mse, compute_sad, compute_grad, compute_conn,
    contrastive_loss, closed_form_matting_loss,
    gradient_preserving_loss
)

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
            pred, attn, trans_feat, enc_feat, contrastive_feat = model(rgb, init_mask)
            pred_sigmoid = torch.sigmoid(pred)

            loss_cf = closed_form_matting_loss(pred_sigmoid, rgb, gt_alpha=gt)
            loss_grad = gradient_preserving_loss(pred_sigmoid, gt)
            loss_conn = compute_conn(pred_sigmoid, gt).mean()
            loss_sad = compute_sad(pred_sigmoid, gt, reduction='mean')

            with torch.no_grad():
                fg_mask = (gt > 0.95).float()
                bg_mask = (gt < 0.05).float()
                fg_mask = F.interpolate(fg_mask, size=trans_feat.shape[2:], mode='nearest')
                bg_mask = F.interpolate(bg_mask, size=trans_feat.shape[2:], mode='nearest')
                pos_feat = (trans_feat * fg_mask).sum(dim=[2, 3]) / (fg_mask.sum(dim=[2, 3]) + 1e-6)
                neg_feat = (trans_feat * bg_mask).sum(dim=[2, 3]) / (bg_mask.sum(dim=[2, 3]) + 1e-6)

            pred_feat = F.normalize(contrastive_feat, dim=1)

            # Project pos/neg features using the same head
            pos_feat = model.contrastive_head(pos_feat.unsqueeze(-1).unsqueeze(-1))
            neg_feat = model.contrastive_head(neg_feat.unsqueeze(-1).unsqueeze(-1))

            pos_feat = F.normalize(pos_feat, dim=1)
            neg_feat = F.normalize(neg_feat, dim=1)

            loss_ctr = contrastive_loss(pred_feat, pos_feat, neg_feat)

            loss = (1.0 * loss_cf + 0.5 * loss_grad + 2.0 * loss_ctr + 0.5 * loss_conn + 1.0 * loss_sad)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * bs * accum_steps
        mse_total += compute_mse(pred_sigmoid, gt).mean().item() * bs
        sad_total += loss_sad.item() * bs
        grad_total += compute_grad(pred_sigmoid, gt).item() * bs
        conn_total += compute_conn(pred_sigmoid, gt).mean().item() * bs
        sample_count += bs

        if (i + 1) % print_interval == 0:
            print(f"[Train {i+1:04d}/{len(loader)}] "
                  f"Loss: {total_loss/sample_count:.4f} | "
                  f"MSE: {mse_total/sample_count:.6f} | "
                  f"SAD: {sad_total/sample_count:.2f} | "
                  f"GRAD: {grad_total/sample_count:.4f} | "
                  f"CONN: {conn_total/sample_count:.4f} | "
                  f"CF: {loss_cf.item():.4f} | GRAD_L: {loss_grad.item():.4f} | "
                  f"CTR: {loss_ctr.item():.4f} | CONN_Loss: {loss_conn.item():.4f} | SAD_Loss: {loss_sad.item():.4f}")
            write_log(log_path, {
                "epoch": epoch,
                "step": i + 1,
                "loss": total_loss / sample_count,
                "mse": mse_total / sample_count,
                "sad": sad_total / sample_count,
                "grad": grad_total / sample_count,
                "conn": conn_total / sample_count,
                "loss_cf": loss_cf.item(),
                "loss_grad": loss_grad.item(),
                "loss_ctr": loss_ctr.item(),
                "loss_conn": loss_conn.item(),
                "loss_sad": loss_sad.item()
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
            pred, attn, _, _, _ = model(rgb, init_mask)
            pred_sigmoid = torch.sigmoid(pred)
            loss_cf = closed_form_matting_loss(pred_sigmoid, rgb, gt_alpha=gt)
            loss_grad = gradient_preserving_loss(pred_sigmoid, gt)
            loss = 1.0 * loss_cf + 0.5 * loss_grad

        total_loss += loss.item() * bs
        mse_total += compute_mse(pred_sigmoid, gt).mean().item() * bs
        sad_total += compute_sad(pred_sigmoid, gt, reduction='mean').item() * bs
        grad_total += compute_grad(pred_sigmoid, gt).item() * bs
        conn_total += compute_conn(pred_sigmoid, gt).mean().item() * bs
        sample_count += bs

        if (i + 1) % print_interval == 0:
            print(f"[Val {i+1:04d}/{len(loader)}] "
                  f"Loss: {total_loss/sample_count:.4f} | "
                  f"MSE: {mse_total/sample_count:.6f} | "
                  f"SAD: {sad_total/sample_count:.2f} | "
                  f"GRAD: {grad_total/sample_count:.4f} | "
                  f"CONN: {conn_total/sample_count:.4f} | "
                  f"CF: {loss_cf.item():.4f} | GRAD_L: {loss_grad.item():.4f}")
            write_log(log_path, {
                "epoch": epoch,
                "step": i + 1,
                "loss": total_loss / sample_count,
                "mse": mse_total / sample_count,
                "sad": sad_total / sample_count,
                "grad": grad_total / sample_count,
                "conn": conn_total / sample_count,
                "loss_cf": loss_cf.item(),
                "loss_grad": loss_grad.item()
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
        "batch_size": 2,
        "print_interval": 20,
        "checkpoint_dir": "checkpoints",
        "csv_path": "../data/pair_for_refiner.csv",
        "resize_to": (736, 1280),
        "num_workers": 6,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "base_channel": 48,
        "sample_fraction": 1.0,
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
        train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, accum_steps=4)
        val_loss, val_mse, val_sad, val_grad, val_conn = val_one_epoch(model, val_loader, device, epoch)

        print(f"Validation Metrics: "
              f"Loss={val_loss:.4f} | MSE={val_mse:.6f} | "
              f"SAD={val_sad:.2f} | GRAD={val_grad:.4f} | CONN={val_conn:.4f}")

        ckpt_path = os.path.join(config["checkpoint_dir"], f"refiner_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()
