import os
import cv2
import torch
import numpy as np
from torchvision.transforms import Compose
from scipy.ndimage import gaussian_filter
from midas.midas.dpt_depth import DPTDepthModel
from midas.midas.transforms import Resize, NormalizeImage, PrepareForNet

def load_midas_model():
    model = DPTDepthModel(
        "./midas/midas/weights/dpt_large_384.pt",
        pretrained=True,
        backbone="vitl16_384",
        non_negative=True
    )
    model.eval().cuda()
    transform = Compose([
        Resize(384, 384, keep_aspect_ratio=True),
        NormalizeImage(mean=[0.5] * 3, std=[0.5] * 3),
        PrepareForNet()
    ])
    return model, transform

def compute_focus_and_spread(alpha_gray, depth_map):
    h, w = alpha_gray.shape
    center_mask = np.zeros_like(alpha_gray, dtype=bool)
    center_mask[h//4:h*3//4, w//4:w*3//4] = True
    mask = (alpha_gray > 220) & center_mask
    if np.sum(mask) == 0:
        return 0.5, 0.05
    depth_vals = depth_map[mask]
    focus = float(np.median(depth_vals)) - 0.02
    spread = np.percentile(depth_vals, 95) - np.percentile(depth_vals, 5)
    return focus, spread

def get_depth_map(img, model, transform):
    sample = transform({"image": img / 255.0})
    input_tensor = sample["image"]
    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.from_numpy(input_tensor)
    input_tensor = input_tensor.unsqueeze(0).cuda()
    with torch.no_grad():
        prediction = model(input_tensor)[0]
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.resize(depth_map, (img.shape[1], img.shape[0]))
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return depth_map

def temporally_smooth_depth(depth_maps, t, radius=2):
    total_frames = len(depth_maps)
    weights = []
    frames = []
    for offset in range(-radius, radius + 1):
        idx = np.clip(t + offset, 0, total_frames - 1)
        weight = np.exp(- (offset**2) / (2 * (radius**2)))
        weights.append(weight)
        frames.append(depth_maps[idx])
    weighted_sum = np.sum([w * f for w, f in zip(weights, frames)], axis=0)
    normalized = weighted_sum / np.sum(weights)
    return gaussian_filter(normalized, sigma=1)

def apply_depth_blur(image, depth_map, focus, focus_range=0.1, max_sigma=6.0):
    is_gray = image.ndim == 2
    if is_gray:
        image = image[..., np.newaxis]

    H, W = depth_map.shape
    output = np.zeros_like(image, dtype=np.float32)

    blur_levels = np.zeros_like(depth_map, dtype=np.float32)
    lower = focus - focus_range
    upper = focus + focus_range

    blur_levels[depth_map < lower] = 1.0
    blur_levels[depth_map > upper] = 0.0

    mask_middle = (depth_map >= lower) & (depth_map <= upper)
    t = (depth_map[mask_middle] - lower) / (upper - lower)
    t = np.clip(t, 0, 1)
    blur_levels[mask_middle] = 1.0 - np.exp(-5 * t)

    blur_levels = gaussian_filter(blur_levels, sigma=4)

    # Boost if average blur is weak
    mean_blur = np.mean(blur_levels)
    if mean_blur < 0.15:
        blur_levels = blur_levels * 0.3 + 0.7 * np.power(blur_levels, 1.5)

    sigma_map = blur_levels * max_sigma
    sigma_map = np.clip(sigma_map, 0, max_sigma * 0.95)
    num_levels = 8
    sigma_values = np.linspace(0, max_sigma, num_levels)

    blurred_images = []
    for sigma in sigma_values:
        if sigma == 0:
            blurred = image.copy()
        else:
            if is_gray:
                blurred = gaussian_filter(image[..., 0], sigma=sigma)[..., np.newaxis]
            else:
                blurred = np.stack([
                    gaussian_filter(image[..., c], sigma=sigma)
                    for c in range(image.shape[2])
                ], axis=-1)
        blurred_images.append(blurred)
    blurred_images = np.stack(blurred_images, axis=0)

    sigma_map_clipped = np.clip(sigma_map, 0, max_sigma - 1e-6)
    idx_float = sigma_map_clipped / max_sigma * (num_levels - 1)
    idx_low = np.floor(idx_float).astype(np.int32)
    idx_high = np.clip(idx_low + 1, 0, num_levels - 1)
    weight_high = idx_float - idx_low
    weight_low = 1.0 - weight_high

    yy, xx = np.indices((H, W))
    for i in range(image.shape[2] if not is_gray else 1):
        stack = blurred_images[:, :, :, i]
        low = stack[idx_low, yy, xx]
        high = stack[idx_high, yy, xx]
        blended = weight_low * low + weight_high * high
        output[..., i] = blended

    return np.clip(output, 0, 255).astype(np.uint8).squeeze() if is_gray else \
        np.clip(output, 0, 255).astype(np.uint8)

def process_video_blur(input_rgb, input_alpha, output_rgb, output_alpha, output_depth):
    max_sigma = 6.0
    temporal_radius = 2

    cap_rgb = cv2.VideoCapture(input_rgb)
    cap_alpha = cv2.VideoCapture(input_alpha)

    frame_width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_rgb = cv2.VideoWriter(output_rgb, fourcc, fps, (frame_width, frame_height))
    out_alpha = cv2.VideoWriter(output_alpha, fourcc, fps, (frame_width, frame_height))
    out_depth = cv2.VideoWriter(output_depth, fourcc, fps, (frame_width, frame_height))

    model, transform = load_midas_model()

    rgb_frames, alpha_frames, depth_maps = [], [], []
    while True:
        ret_rgb, rgb = cap_rgb.read()
        ret_alpha, alpha = cap_alpha.read()
        if not ret_rgb or not ret_alpha:
            break
        rgb_input = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth_map = get_depth_map(rgb_input, model, transform)
        rgb_frames.append(rgb)
        alpha_frames.append(alpha)
        depth_maps.append(depth_map)

    cap_rgb.release()
    cap_alpha.release()

    focus_list = []
    spread_list = []
    N = min(int(total_frames * 0.05), 10)
    for i in range(N):
        alpha_gray = cv2.cvtColor(alpha_frames[i], cv2.COLOR_BGR2GRAY)
        focus, spread = compute_focus_and_spread(alpha_gray, depth_maps[i])
        focus_list.append(focus)
        spread_list.append(spread)

    fixed_focus = float(np.mean(focus_list))
    depth_spread = np.median(spread_list)
    focus_range = max(depth_spread * 0.8, 0.03)
    print(f"\n\u2705 Focus: {fixed_focus:.3f} | Spread: {depth_spread:.3f} | Range: {focus_range:.3f}")

    for t in range(total_frames):
        rgb = rgb_frames[t]
        alpha = alpha_frames[t]
        alpha_gray = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
        avg_depth = temporally_smooth_depth(depth_maps, t, temporal_radius)

        depth_vis = (avg_depth * 255).astype(np.uint8)
        depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        out_depth.write(depth_vis_bgr)

        blurred_rgb = apply_depth_blur(rgb, avg_depth, fixed_focus, focus_range, max_sigma)
        blurred_alpha = apply_depth_blur(alpha_gray, avg_depth, fixed_focus, focus_range, max_sigma)
        blurred_alpha_3ch = cv2.cvtColor(blurred_alpha, cv2.COLOR_GRAY2BGR)

        out_rgb.write(blurred_rgb)
        out_alpha.write(blurred_alpha_3ch)

        print(f"\U0001f7e2 Frame {t+1}/{total_frames}", end='\r')

    out_rgb.release()
    out_alpha.release()
    out_depth.release()
    print("\n\n✅ Done. All videos saved.")

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
        )
