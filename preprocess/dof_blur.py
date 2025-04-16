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


def compute_focus_from_alpha(alpha_gray, depth_map):
    mask = alpha_gray > 127
    if np.sum(mask) == 0:
        return 0.5
    return float(np.mean(depth_map[mask]))


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


def apply_depth_blur(image, depth_map, focus, focus_range=0.02, max_sigma=10.0):
    is_gray = image.ndim == 2
    if is_gray:
        image = image[..., np.newaxis]

    H, W = depth_map.shape
    output = np.zeros_like(image, dtype=np.float32)

    blur_levels = np.zeros_like(depth_map)
    lower = focus - focus_range
    upper = focus + focus_range

    # Reversed blur logic: Far = more blur
    blur_levels[depth_map < lower] = 1.0
    blur_levels[depth_map > upper] = 0.0

    mask_middle = (depth_map >= lower) & (depth_map <= upper)
    t = (depth_map[mask_middle] - lower) / (upper - lower)
    t = np.clip(t, 0, 1)
    smoothstep = (1 - t) ** 2 * (3 - 2 * (1 - t))
    blur_levels[mask_middle] = smoothstep

    sigma_map = blur_levels * max_sigma
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
    blurred_images = np.stack(blurred_images, axis=0)  # shape: (L, H, W, C)

    sigma_map_clipped = np.clip(sigma_map, 0, max_sigma - 1e-6)
    idx_float = sigma_map_clipped / max_sigma * (num_levels - 1)
    idx_low = np.floor(idx_float).astype(np.int32)
    idx_high = np.clip(idx_low + 1, 0, num_levels - 1)
    weight_high = idx_float - idx_low
    weight_low = 1.0 - weight_high

    # Generate per-pixel index grids
    yy, xx = np.indices((H, W))

    for i in range(image.shape[2] if not is_gray else 1):
        stack = blurred_images[:, :, :, i]  # shape: (L, H, W)

        low = stack[idx_low, yy, xx]
        high = stack[idx_high, yy, xx]
        blended = weight_low * low + weight_high * high
        output[..., i] = blended

    return np.clip(output, 0, 255).astype(np.uint8).squeeze() if is_gray else \
        np.clip(output, 0, 255).astype(np.uint8)


def process_video_blur(input_rgb, input_alpha, output_rgb, output_alpha):
    focus_range = 0.02
    max_sigma = 10.0

    cap_rgb = cv2.VideoCapture(input_rgb)
    cap_alpha = cv2.VideoCapture(input_alpha)

    frame_width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(output_rgb) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_alpha) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_rgb = cv2.VideoWriter(output_rgb, fourcc, fps, (frame_width, frame_height))
    out_alpha = cv2.VideoWriter(output_alpha, fourcc, fps, (frame_width, frame_height))

    if not out_rgb.isOpened() or not out_alpha.isOpened():
        print("❌ Failed to open video writers")
        print(f"Output paths: {output_rgb}, {output_alpha}")
        return

    model, transform = load_midas_model()

    ret_rgb, first_frame = cap_rgb.read()
    ret_alpha, first_alpha = cap_alpha.read()
    if not ret_rgb or not ret_alpha:
        print("❌ Failed to read first frame")
        return

    first_rgb_input = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    first_alpha_gray = cv2.cvtColor(first_alpha, cv2.COLOR_BGR2GRAY)
    first_depth_map = get_depth_map(first_rgb_input, model, transform)
    fixed_focus = compute_focus_from_alpha(first_alpha_gray, first_depth_map)
    print(f"📍 Focus depth fixed to: {fixed_focus:.3f}")

    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_alpha.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    while True:
        ret_rgb, frame = cap_rgb.read()
        ret_alpha, alpha = cap_alpha.read()
        if not ret_rgb or not ret_alpha:
            break

        alpha_gray = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
        rgb_for_depth = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_map = get_depth_map(rgb_for_depth, model, transform)

        blurred_rgb = apply_depth_blur(frame, depth_map, fixed_focus, focus_range, max_sigma)
        blurred_alpha = apply_depth_blur(alpha_gray, depth_map, fixed_focus, focus_range, max_sigma)

        blurred_alpha_3ch = cv2.cvtColor(blurred_alpha, cv2.COLOR_GRAY2BGR)
        out_rgb.write(blurred_rgb)
        out_alpha.write(blurred_alpha_3ch)

        frame_idx += 1
        print(f"🟢 Frame {frame_idx}", end='\r')

    cap_rgb.release()
    cap_alpha.release()
    out_rgb.release()
    out_alpha.release()
    print("\n✅ Done. Videos saved.")


if __name__ == "__main__":
    process_video_blur(
        "./data/video_defocused/train/fgr/0000.mp4",
        "./data/video_defocused/train/pha/0000.mp4",
        "blurred_rgb.mp4",
        "blurred_alpha_3ch.mp4"
    )
