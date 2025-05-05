"""
Training, validation and inference pipeline for our proposed MatteRefiner.
"""

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import sobel, to_pil_image
from tqdm import tqdm
import os
import json

# ---------------- Metrics ---------------- #
def compute_mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def compute_sad(pred, target):
    return torch.sum(torch.abs(pred - target)).item()

def compute_grad(pred, target):
    grad_pred = sobel(pred)
    grad_target = sobel(target)
    return torch.mean(torch.abs(grad_pred - grad_target)).item()

# ---------------- Logger ---------------- #
def log_to_file(filepath, entry):
    with open(filepath, 'a') as f:
        f.write(json.dumps(entry) + '\n')

# ---------------- Train ---------------- #
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, log_path='train_logs.jsonl', print_interval=20):
    model.train()
    total_loss = total_mae = total_sad = total_grad = 0.0
    for i, (rgb, init_mask, gt) in enumerate(tqdm(loader, desc=f"[Train Epoch {epoch}]"), 1):
        rgb, init_mask, gt = rgb.to(device), init_mask.to(device), gt.to(device)

        with torch.cuda.amp.autocast():
            pred, _ = model(rgb, init_mask)
            loss = F.l1_loss(pred, gt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        mae = compute_mae(pred, gt)
        sad = compute_sad(pred, gt)
        grad = compute_grad(pred, gt)

        total_loss += loss.item()
        total_mae += mae
        total_sad += sad
        total_grad += grad

        if i % print_interval == 0 or i == len(loader):
            print(f"  At [{i}/{len(loader)}]: loss={total_loss/i:.4f}, mae={total_mae/i:.4f}, sad={total_sad/i:.2f}, grad={total_grad/i:.4f}")

        log_to_file(log_path, {
            "epoch": epoch,
            "batch": i,
            "loss": loss.item(),
            "mae": mae,
            "sad": sad,
            "grad": grad,
            "mode": "train"
        })

    return total_loss / len(loader)

# ---------------- Validation ---------------- #
def val_one_epoch(model, loader, device, epoch, log_path='val_logs.jsonl', print_interval=20):
    model.eval()
    total_mae = total_sad = total_grad = total_loss = 0.0
    with torch.no_grad():
        for i, (rgb, init_mask, gt) in enumerate(tqdm(loader, desc=f"[Val Epoch {epoch}]"), 1):
            rgb, init_mask, gt = rgb.to(device), init_mask.to(device), gt.to(device)

            with torch.cuda.amp.autocast():
                pred, _ = model(rgb, init_mask)
                loss = F.l1_loss(pred, gt)

            mae = compute_mae(pred, gt)
            sad = compute_sad(pred, gt)
            grad = compute_grad(pred, gt)

            total_loss += loss.item()
            total_mae += mae
            total_sad += sad
            total_grad += grad

            if i % print_interval == 0 or i == len(loader):
                print(f"  At [{i}/{len(loader)}]: loss={total_loss/i:.4f}, mae={total_mae/i:.4f}, sad={total_sad/i:.2f}, grad={total_grad/i:.4f}")

            log_to_file(log_path, {
                "epoch": epoch,
                "batch": i,
                "loss": loss.item(),
                "mae": mae,
                "sad": sad,
                "grad": grad,
                "mode": "val"
            })

    N = len(loader)
    return total_mae / N, total_sad / N, total_grad / N

# ---------------- Full Train Loop ---------------- #
def train(model, train_loader, val_loader, device, epochs=50, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_mae, val_sad, val_grad = val_one_epoch(model, val_loader, device, epoch)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | SAD: {val_sad:.2f} | Grad: {val_grad:.4f}")

# ---------------- Inference ---------------- #
@torch.no_grad()
def test(model, dataloader, device, save_dir="inference_results", save_fn=None):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for i, (rgb, init_mask, _) in enumerate(tqdm(dataloader, desc="[Inference]")):
        rgb, init_mask = rgb.to(device), init_mask.to(device)

        with torch.cuda.amp.autocast():
            pred, _ = model(rgb, init_mask)

        matte = torch.clamp(pred, 0, 1)

        for b in range(matte.shape[0]):
            matte_b = matte[b, 0].cpu()
            matte_b = to_pil_image(matte_b)

            if save_fn:
                save_fn(matte_b, i * matte.shape[0] + b)
            else:
                matte_b.save(os.path.join(save_dir, f"matte_{i * matte.shape[0] + b:04d}.png"))
