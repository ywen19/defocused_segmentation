import os
import sys
import cv2
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.abspath("Video_Depth_Anything"))

from video_depth_anything.video_depth import VideoDepthAnything

# --- Load VDA model ---
def load_vda_model(ckpt_path="./Video_Depth_Anything/checkpoints/video_depth_anything_vitl.pth", encoder="vitl"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    model = VideoDepthAnything(**model_configs[encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
    return model.to(device).eval(), device

# --- Compute focal depth & spread ---
def compute_focus_and_spread(alpha_gray, depth_map):
    h, w = alpha_gray.shape
    center_mask = np.zeros_like(alpha_gray, dtype=bool)
    center_mask[h//4:h*3//4, w//4:w*3//4] = True
    mask = (alpha_gray > 127) & center_mask
    if np.sum(mask) == 0:
        return 0.5, 0.05
    vals = depth_map[mask]
    focus = float(np.median(vals)) - 0.02
    spread = np.percentile(vals, 95) - np.percentile(vals, 5)
    return focus, spread

# --- Use VDA to get depth ---
def get_video_depths(frames, model, input_size=518, device='cuda', fp32=False):
    # Convert list of frames to NumPy array
    frame_array = np.stack(frames, axis=0)  # shape: (N, H, W, 3)
    depths, _ = model.infer_video_depth(frame_array, target_fps=-1, input_size=input_size, device=device, fp32=fp32)
    # Normalize each depth frame
    norm_depths = [(d - d.min()) / (d.max() - d.min() + 1e-8) for d in depths]
    return norm_depths

# --- Smooth across time ---
def temporally_smooth_depth(depth_maps, t, radius=2):
    total = len(depth_maps)
    weights, frames = [], []
    for offset in range(-radius, radius + 1):
        idx = np.clip(t + offset, 0, total - 1)
        w = np.exp(- (offset**2) / (2 * radius**2))
        weights.append(w)
        frames.append(depth_maps[idx])
    smoothed = np.sum([w*f for w,f in zip(weights, frames)], axis=0) / np.sum(weights)
    return gaussian_filter(smoothed, sigma=1)

# --- Apply defocus blur ---
def apply_depth_blur(image, depth_map, focus, focus_range=0.1, max_sigma=6.0):
    is_gray = image.ndim == 2
    if is_gray: image = image[..., np.newaxis]

    blur_levels = np.zeros_like(depth_map)
    lower, upper = focus - focus_range, focus + focus_range
    blur_levels[depth_map < lower] = 1.0
    blur_levels[depth_map > upper] = 0.0

    mask_mid = (depth_map >= lower) & (depth_map <= upper)
    t = np.clip((depth_map[mask_mid] - lower) / (upper - lower), 0, 1)
    blur_levels[mask_mid] = 1.0 - np.exp(-5 * t)
    blur_levels = gaussian_filter(blur_levels, sigma=4)

    if np.mean(blur_levels) < 0.15:
        blur_levels = blur_levels * 0.3 + 0.7 * np.power(blur_levels, 1.5)

    sigma_map = np.clip(blur_levels * max_sigma, 0, max_sigma * 0.95)
    num_levels = 8
    sigma_values = np.linspace(0, max_sigma, num_levels)

    blurred_stack = []
    for sigma in sigma_values:
        if sigma == 0:
            blurred = image.copy()
        else:
            blurred = np.stack([
                gaussian_filter(image[..., c], sigma=sigma)
                for c in range(image.shape[2])
            ], axis=-1)
        blurred_stack.append(blurred)
    blurred_stack = np.stack(blurred_stack, axis=0)

    idx_float = sigma_map / max_sigma * (num_levels - 1)
    idx_low = np.floor(idx_float).astype(int)
    idx_high = np.clip(idx_low + 1, 0, num_levels - 1)
    w_high = idx_float - idx_low
    w_low = 1.0 - w_high

    yy, xx = np.indices(depth_map.shape)
    output = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        low = blurred_stack[idx_low, yy, xx, c]
        high = blurred_stack[idx_high, yy, xx, c]
        output[..., c] = w_low * low + w_high * high

    return np.clip(output, 0, 255).astype(np.uint8).squeeze() if is_gray else np.clip(output, 0, 255).astype(np.uint8)

# --- Full process ---
def process_video_blur(input_rgb, input_alpha, output_rgb, output_alpha, output_depth, vda_ckpt="./Video_Depth_Anything/checkpoints/video_depth_anything_vitl.pth"):
    max_sigma = 6.0
    temporal_radius = 2

    cap_rgb = cv2.VideoCapture(input_rgb)
    cap_alpha = cv2.VideoCapture(input_alpha)
    W, H = int(cap_rgb.get(3)), int(cap_rgb.get(4))
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    total = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_rgb = cv2.VideoWriter(output_rgb, fourcc, fps, (W, H))
    out_alpha = cv2.VideoWriter(output_alpha, fourcc, fps, (W, H))
    out_depth = cv2.VideoWriter(output_depth, fourcc, fps, (W, H))

    model, device = load_vda_model(vda_ckpt)

    # Load all frames first
    rgb_frames, alpha_frames = [], []
    while True:
        ret_rgb, rgb = cap_rgb.read()
        ret_alpha, alpha = cap_alpha.read()
        if not ret_rgb or not ret_alpha: break
        rgb_frames.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        alpha_frames.append(alpha)

    cap_rgb.release(); cap_alpha.release()

    depth_maps = get_video_depths(rgb_frames, model, input_size=518, device=device)

    # Focus stats
    focus_list, spread_list = [], []
    N = min(int(total * 0.05), 10)
    for i in range(N):
        alpha_gray = cv2.cvtColor(alpha_frames[i], cv2.COLOR_BGR2GRAY)
        f, s = compute_focus_and_spread(alpha_gray, depth_maps[i])
        focus_list.append(f)
        spread_list.append(s)

    focus = np.mean(focus_list)
    focus_range = max(np.median(spread_list) * 0.8, 0.03)
    print(f"\n\u2705 Focus: {focus:.3f} | Range: {focus_range:.3f}")

    for t in range(total):
        rgb = cv2.cvtColor(rgb_frames[t], cv2.COLOR_RGB2BGR)
        alpha = alpha_frames[t]
        alpha_gray = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
        avg_depth = temporally_smooth_depth(depth_maps, t, temporal_radius)

        out_depth.write(cv2.cvtColor((avg_depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        out_rgb.write(apply_depth_blur(rgb, avg_depth, focus, focus_range, max_sigma))
        blurred_alpha = apply_depth_blur(alpha_gray, avg_depth, focus, focus_range, max_sigma)
        out_alpha.write(cv2.cvtColor(blurred_alpha, cv2.COLOR_GRAY2BGR))

        print(f"\U0001f7e2 Frame {t+1}/{total}", end='\r')

    out_rgb.release(); out_alpha.release(); out_depth.release()
    print("\n\n✅ Done. All videos saved.")

# Run
if __name__ == "__main__":
    process_video_blur(
        "./data/video_composed/train/fgr/0174.mp4",
        "./data/video_composed/train/pha/0174.mp4",
        "0174_blurred_rgb.mp4",
        "0174_blurred_alpha.mp4",
        "0174_depth.mp4",
        vda_ckpt="./Video_Depth_Anything/checkpoints/video_depth_anything_vitl.pth"
    )



"""
if __name__ == "__main__":
    test_points = ['0167', '0080', '0239', '0241', '0266', '0295', '0353', '0405', '0120', 
    '0171', '0324', '0426', '0456', '0470', '0000', '0174']
    for test_point in test_points:
        process_video_blur(
            f"./data/video_defocused/train/fgr/{test_point}.mp4",
            f"./data/video_defocused/train/pha/{test_point}.mp4",
            f"{test_point}_blurred_rgb.mp4",
            f"{test_point}_blurred_alpha_3ch.mp4",
            f"{test_point}_estimated_depth.mp4"
        )"""
