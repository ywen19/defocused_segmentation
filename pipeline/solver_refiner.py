import os
import sys
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dataload.build_dataloaders import build_dataloaders
from model.refiner import MatteRefiner
from utils import (
    compute_cf_loss,
    compute_grad_loss_soft_only,
    compute_conn_with_unknown_mask,
    compute_ctr_loss,
    compute_embedding_ctr_loss,
    compute_sad_loss,
    compute_grad_metric,
    compute_conn_mismatch_loss,
    compute_mse_with_trimap,
    compute_sad_with_trimap,
    extract_unknown_mask,
    generate_trimap_from_gt,
)

def write_log(log_path, data):
    with open(log_path, "a") as f:
        f.write(json.dumps(data) + "\n")

def compute_metrics(pred, gt):
    with torch.no_grad():
        trimap = generate_trimap_from_gt(gt, dilation=20)
        unknown_mask = extract_unknown_mask(trimap, strategy='auto', threshold=0.1)
        grad = compute_grad_metric(pred, gt, trimap)
        conn = compute_conn_mismatch_loss(pred, gt)
        mse = compute_mse_with_trimap(pred, gt, trimap)
        sad = compute_sad_with_trimap(pred, gt, trimap)
    return grad.item(), conn.item(), mse.item(), sad.item()

def train_one_epoch(model, loader, optimizer, scaler, device, epoch,
                    print_interval=20, log_path="train_metrics.jsonl", accum_steps=2):
    model.train()
    total_loss = 0.0
    sample_count = 0

    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        rgb, init_mask, gt = [x.to(device) for x in batch]
        bs = rgb.size(0)

        trimap = generate_trimap_from_gt(gt, dilation=20)

        with torch.amp.autocast(device_type='cuda'):
            pred_alpha, feat_enc, feat_trans, ctr_enc, ctr_trans = model(rgb, init_mask)

            loss_cf = compute_cf_loss(pred_alpha, gt, trimap=trimap, use_trimap=True)
            loss_grad = compute_grad_loss_soft_only(pred_alpha, gt, coarse_mask=init_mask, trimap=trimap, use_trimap=True)
            loss_sad = compute_sad_loss(pred_alpha, gt, trimap=trimap, use_trimap=True)

            loss_ctr_spatial = compute_ctr_loss(
                anchor_feat=feat_trans,
                positive_feat=feat_enc,
                pred_mask=pred_alpha,
                gt_mask=gt,
                margin=0.05,
                use_triplet=True
            )

            loss_ctr_embed = compute_embedding_ctr_loss(
                anchor_embed=ctr_trans,
                positive_embed=ctr_enc,
                margin=0.05,
                use_triplet=True
            )

        with torch.cuda.amp.autocast(enabled=False):
            loss_conn = compute_conn_with_unknown_mask(pred_alpha, gt, trimap=trimap)

        with torch.amp.autocast(device_type='cuda'):
            loss = (0.5 * loss_cf + 1.0 * loss_grad + 0.5 * loss_conn +
                    0.5 * loss_sad + 1.0 * loss_ctr_spatial + 0.5 * loss_ctr_embed) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * bs * accum_steps
        sample_count += bs

        if (i + 1) % print_interval == 0:
            grad_m, conn_m, mse_m, sad_m = compute_metrics(pred_alpha, gt)
            print(f"[Train {i+1:04d}/{len(loader)}] Loss: {total_loss/sample_count:.4f} "
                  f"| CF: {loss_cf.item():.4f} | GRAD: {loss_grad.item():.4f} "
                  f"| CONN: {loss_conn.item():.4f} | SAD: {loss_sad.item():.4f} "
                  f"| CTR: {loss_ctr_spatial.item():.4f} | E-CTR: {loss_ctr_embed.item():.4f} "
                  f"| M-GRAD: {grad_m:.4f} | M-CONN: {conn_m:.4f} "
                  f"| M-MSE: {mse_m:.4f} | M-SAD: {sad_m:.4f}")

            write_log(log_path, {
                "epoch": epoch,
                "step": i + 1,
                "loss": total_loss / sample_count,
                "loss_cf": loss_cf.item(),
                "loss_grad": loss_grad.item(),
                "loss_conn": loss_conn.item(),
                "loss_sad": loss_sad.item(),
                "loss_ctr_spatial": loss_ctr_spatial.item(),
                "loss_ctr_embed": loss_ctr_embed.item(),
                "metric_grad": grad_m,
                "metric_conn": conn_m,
                "metric_mse": mse_m,
                "metric_sad": sad_m,
            })

@torch.no_grad()
def val_one_epoch(model, loader, device, epoch,
                  print_interval=10, log_path="val_metrics.jsonl"):
    model.eval()
    total_loss = 0.0
    sample_count = 0

    for i, batch in enumerate(loader):
        rgb, init_mask, gt = [x.to(device) for x in batch]
        bs = rgb.size(0)
        trimap = generate_trimap_from_gt(gt, dilation=20)

        with torch.amp.autocast(device_type='cuda'):
            pred_alpha, feat_enc, feat_trans, ctr_enc, ctr_trans = model(rgb, init_mask)

            loss_cf = compute_cf_loss(pred_alpha, gt, trimap=trimap, use_trimap=True)
            loss_grad = compute_grad_loss_soft_only(pred_alpha, gt, coarse_mask=init_mask, trimap=trimap, use_trimap=True)
            loss_sad = compute_sad_loss(pred_alpha, gt, trimap=trimap, use_trimap=True)
            loss_ctr_spatial = compute_ctr_loss(feat_trans, feat_enc, pred_alpha, gt, margin=0.05, use_triplet=True)
            loss_ctr_embed = compute_embedding_ctr_loss(ctr_trans, ctr_enc, margin=0.05, use_triplet=True)

        with torch.cuda.amp.autocast(enabled=False):
            loss_conn = compute_conn_with_unknown_mask(pred_alpha, gt, trimap=trimap)

        loss = (0.5 * loss_cf + 1.0 * loss_grad + 0.5 * loss_conn +
                0.5 * loss_sad + 1.0 * loss_ctr_spatial + 0.5 * loss_ctr_embed)

        total_loss += loss.item() * bs
        sample_count += bs

        if (i + 1) % print_interval == 0:
            grad_m, conn_m, mse_m, sad_m = compute_metrics(pred_alpha, gt)
            print(f"[Val {i+1:04d}/{len(loader)}] Loss: {total_loss/sample_count:.4f} "
                  f"| CF: {loss_cf.item():.4f} | GRAD: {loss_grad.item():.4f} "
                  f"| CONN: {loss_conn.item():.4f} | SAD: {loss_sad.item():.4f} "
                  f"| CTR: {loss_ctr_spatial.item():.4f} | E-CTR: {loss_ctr_embed.item():.4f} "
                  f"| M-GRAD: {grad_m:.4f} | M-CONN: {conn_m:.4f} | M-MSE: {mse_m:.4f} | M-SAD: {sad_m:.4f}")

            write_log(log_path, {
                "epoch": epoch,
                "step": i + 1,
                "loss": total_loss / sample_count,
                "loss_cf": loss_cf.item(),
                "loss_grad": loss_grad.item(),
                "loss_conn": loss_conn.item(),
                "loss_sad": loss_sad.item(),
                "loss_ctr_spatial": loss_ctr_spatial.item(),
                "loss_ctr_embed": loss_ctr_embed.item(),
                "metric_grad": grad_m,
                "metric_conn": conn_m,
                "metric_mse": mse_m,
                "metric_sad": sad_m,
            })

    return total_loss / sample_count

def main():
    config = {
        "num_epochs": 1,
        "batch_size": 1,
        "print_interval": 20,
        "checkpoint_dir": "checkpoints",
        "csv_path": "../data/pair_for_refiner.csv",
        "resize_to": (736, 1280),
        "num_workers": 6,
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "base_channel": 64,
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
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{config['num_epochs']} ---")
        train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, accum_steps=4)
        val_loss = val_one_epoch(model, val_loader, device, epoch)

        ckpt_path = os.path.join(config["checkpoint_dir"], f"refiner_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

        scheduler.step()

if __name__ == "__main__":
    main()
