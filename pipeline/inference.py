import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, resize
from torchvision.transforms.functional import to_pil_image
import os
import sys

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from model.refiner import MatteRefiner

# === Config ===
ckpt_path = "checkpoints/refiner_epoch1.pth"   # your model
input_rgb_path = "../data/video_defocused_processed/train/fgr/0000/frames/0138.png"
input_mask_path = "../data/video_defocused_processed/train/fgr/0000/mask/0138.png"
resize_to = (1088, 1920)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
model = MatteRefiner(base_channels=48)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval().to(device)

# === Load and prepare input ===
def load_input(rgb_path, mask_path, resize_to):
    rgb = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    rgb = resize(rgb, resize_to)
    mask = resize(mask, resize_to)

    rgb = to_tensor(rgb).unsqueeze(0).to(device)  # (1, 3, H, W)
    mask = to_tensor(mask).unsqueeze(0).to(device)  # (1, 1, H, W)
    return rgb, mask

rgb, init_mask = load_input(input_rgb_path, input_mask_path, resize_to)

# === Inference ===
with torch.no_grad():
    pred_alpha, *_ = model(rgb, init_mask)
    pred_alpha = torch.clamp(pred_alpha, 0, 1)[0, 0].cpu()  # (H, W)

# === Save output ===
output_path = "output_alpha.png"
pred_alpha_img = to_pil_image(pred_alpha)  # converts (H, W) in [0,1] to PIL image
pred_alpha_img.save(output_path)
