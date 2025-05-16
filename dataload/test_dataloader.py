import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import matplotlib.pyplot as plt
from build_dataloaders import build_dataloaders

def visualize_batch(csv_path, resize_to=(1080, 1920), batch_size=4, num_workers=2):
    train_loader, _ = build_dataloaders(
        csv_path=csv_path,
        resize_to=resize_to,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    batch = next(iter(train_loader))
    rgb, mask, gt, trimap = batch
    tp_uint8 = (trimap * 255).byte()

    n = min(4, rgb.size(0))
    fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(12, 4 * n))

    for i in range(n):
        axes[i, 0].imshow(rgb[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("RGB Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask[i, 0].numpy(), cmap='gray')
        axes[i, 1].set_title("Initial Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(gt[i, 0].numpy(), cmap='gray')
        axes[i, 2].set_title("GT Matte")
        axes[i, 2].axis("off")

        im = tp_uint8[i, 0].cpu().numpy()
        axes[i, 3].imshow(im, cmap='gray',
                          vmin=0, vmax=255,
                          interpolation='nearest')
        axes[i, 3].set_title("Trimap")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()

# Run
if __name__ == "__main__":
    visualize_batch("../data/pair_for_refiner.csv", resize_to=(1080, 1920))
