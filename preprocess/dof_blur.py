import os
import sys
import gc
import psutil
from typing import Tuple, List

import random
from tqdm import tqdm

import cv2
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.abspath("Video_Depth_Anything"))
from video_depth_anything.video_depth import VideoDepthAnything

TEMP_ROOT = "./data/video_tmp"
MAX_SEGMENT_DURATION = 15          # cut long video to short clips to avoid memory leak
SPLIT_THRESHOLD_SEC = 20           # determine if a video need to be segmented
MAX_VIDEOS_BEFORE_RESTART = 5      # used by the shell script; kill the progress and restart due to uni pc cannot del variables properly


def load_vda_model(ckpt_path="./Video_Depth_Anything/checkpoints/video_depth_anything_vitl.pth", encoder="vitl"):
    """
    Load video-depth-anything model for depth estimation;

    ckpt_path : str, path to the downloaded pretrain vda;
    encoder   : str, encoder type used by vda;

    return    : Tuple[torch.nn.Module, str]
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    model = VideoDepthAnything(**model_configs[encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
    return model.to(device).eval(), device

def estimate_chunk_size(base=8, safety=0.85, min_c=2, max_c=32):
    """
    Dynamically calculate an appropriate "chunk size" for processing data on a CUDA-enabled GPU;

    base   : int, base multiplier for chunk size;
    safety : float, the max memory percentage allowed to be used for the process;
    min_c  : int, min chunk size;
    max_c  : int, max chunk size;

    return : int
    """

    if not torch.cuda.is_available(): return base
    try:
        total = torch.cuda.get_device_properties(0).total_memory   # get total gpy memory
        reserved = torch.cuda.memory_reserved(0)                   # reserved memory by torch caching allocator
        allocated = torch.cuda.memory_allocated(0)                 # memory currently used by tensors
        free = total - max(reserved, allocated)                    # estimate free memory
        # memory to be used by scaling the current free memory
        # then clamp the result for the available chunk size
        return max(min_c, min(max_c, int(base * ((free / total) * safety))))
    except: return base

def compute_focus_and_spread(alpha_gray, depth_map):
    """"
    Calculate the focus and spread for defocus simulation;
    This is guided by the foreground subject matte (or mask);
    Basically it performs  analysis of depth values within a central region and under a threshold condition.

    alpha_gray : np.ndarray, the gray scale matte from the all-in-focus frame/video;
    depth_map  : np.ndarray, estimated by the vda;

    return Tuple[float, float]
    """
    
    # We compute focus and spread using the intersection of the central frame region and the foreground mask
    # This avoids the need to compute the bounding box or centroid of the subject geometry
    # The reasoning is that we're simulating defocus with a *fixed focal plane*
    # similar to how focus works in real-world cinematography
    # In many filming scenarios, the focal setting doesn't change dynamically as subjects move
    # So this method assumes cinematic fixed-focus shots (a real-world analogy)
    h, w = alpha_gray.shape 
    center = np.zeros_like(alpha_gray, dtype=bool)     
    center[h//4:h*3//4, w//4:w*3//4] = True
    mask = (alpha_gray > 127) & center
    
    # if no mask
    if np.sum(mask) == 0: return 0.5, 0.05  # depth midpoint for neutral focus, and a safe spread

    vals = depth_map[mask]  # extract depth values where the mask is true
    focus = float(np.median(vals))  # the focus should be driven by the most typical foreground subject depth
    # spread as the difference between the 95th and 5th percentile of the depth and avoid extreme outliers
    spread_val = np.percentile(vals, 95) - np.percentile(vals, 5)
    # clip to ensure the spread is not too narrow or too wide
    spread = float(np.clip(spread_val, 0.02, 0.15))
    return focus, spread

def get_video_depths_chunked(frames, model, input_size=518, device='cuda', chunk_size=8):
    """
    Estimate depth in chunk to avoid OOM;

    frames     : List[np.ndarray], list of rgb frames (h*w*3)
    model      : torch.nn.model, loaded vda
    input_size : inpur resolution for depth estimation
    device     : device to tun; by default CUDA
    chunk_size : number of frames to process at once

    return: List[np.ndarray]
    """
    results = []
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i + chunk_size]
        data = np.stack(chunk, axis=0)
        try:
            depths, _ = model.infer_video_depth(data, target_fps=-1, input_size=input_size, device=device)
        except Exception as e:
            torch.cuda.empty_cache(); gc.collect()
            raise RuntimeError(f"Chunk OOM at frames {i}-{i+len(chunk)-1}") from e
        # normalize estimated depth to 0-1 by min-max normalization
        results.extend([(d - d.min()) / (d.max() - d.min() + 1e-8) for d in depths])
        torch.cuda.empty_cache(); gc.collect()
    return results

def temporally_smooth_depth(depth_maps, t, radius=2):
    """
    Smoothing out depth with a temporal gaussian kernel with a sliding window across frames;
    This is to reduce temporal jitter in depth maps, resulting in a more stable, 
    natural-looking defocus blur across video frames.

    We originally has this for depth estimation with midas;
    vda is more reliable since it is designed for sequence.
    Still we keep this for vda.

    depth_maps : List[np.ndarray], list of depth maps for the video
    t          : int, current timestep (frame) for smoothing
    radius     : int, sliding window range for neighboring frames on each side of t for smoothing

    return: np.ndarray
    """

    weights, frames = [], []
    for o in range(-radius, radius + 1):
        idx = np.clip(t + o, 0, len(depth_maps)-1)  # actual frame index in the sliding window
        # gaussian_w(o) = exp(-o^2/(2*radius^2))
        w = np.exp(- (o**2) / (2 * radius**2))  # gaussian weight; frame closer to t get higher weight
        weights.append(w); frames.append(depth_maps[idx])
    # pairs depth map and its assicated gaussian weight
    # multiply to get smoothed weighted depth for each frame
    # then sum up to get the final depth at time t
    return gaussian_filter(np.sum([w*f for w,f in zip(weights, frames)], axis=0) / np.sum(weights), sigma=1)

def apply_depth_blur(image, depth, focus, focus_range=0.1, max_sigma=6.0):
    """
    Apply depth-driven defocus simulation to the rgb frame.

    
    """
    if image.ndim == 2: image = image[..., None]
    blur_levels = np.zeros_like(depth)
    lower, upper = focus - focus_range, focus + focus_range
    blur_levels[depth < lower] = 1.0
    blur_levels[depth > upper] = 0.0
    mid = (depth >= lower) & (depth <= upper)
    t = np.clip((depth[mid] - lower) / (upper - lower), 0, 1)
    blur_levels[mid] = 1.0 - np.exp(-5 * t)
    blur_levels = gaussian_filter(blur_levels, sigma=4)
    if np.mean(blur_levels) < 0.15:
        blur_levels = 0.3 * blur_levels + 0.7 * np.power(blur_levels, 1.5)
    sigma_map = np.clip(blur_levels * max_sigma, 0, max_sigma * 0.95)
    levels = 8
    sigma_vals = np.linspace(0, max_sigma, levels)
    stack = [np.stack([gaussian_filter(image[..., c], s) if s > 0 else image[..., c]
                       for c in range(image.shape[2])], axis=-1) for s in sigma_vals]
    stack = np.stack(stack, axis=0)

    idx_float = sigma_map / max_sigma * (levels - 1)
    low = np.floor(idx_float).astype(int)
    high = np.clip(low + 1, 0, levels - 1)
    w_high = idx_float - low
    w_low = 1.0 - w_high
    yy, xx = np.indices(depth.shape)

    output = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        out = w_low * stack[low, yy, xx, c] + w_high * stack[high, yy, xx, c]
        output[..., c] = out
    return np.clip(output, 0, 255).astype(np.uint8).squeeze()

def release_writers(*ws):
    for w in ws:
        try: w.release()
        except: pass

def segment_video(input_path, output_dir, base_name, fps):
    cap = cv2.VideoCapture(input_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    seg_len = int(fps * MAX_SEGMENT_DURATION)
    os.makedirs(output_dir, exist_ok=True)
    idx = 1
    while True:
        out_path = os.path.join(output_dir, f"{base_name}_{idx:03d}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for _ in range(seg_len):
            ret, frame = cap.read()
            if not ret: break
            writer.write(frame)
        writer.release()
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= total:
            break
        idx += 1
    cap.release()

def get_segmented_paths(in_path, split, kind):
    base = os.path.splitext(os.path.basename(in_path))[0]
    return os.path.join(TEMP_ROOT, split, kind, base)

def process_segments(base, split, model, device, input_size, fgr_segs, pha_segs, out_root):
    for f in sorted(os.listdir(fgr_segs)):
        seg_idx = f.split('_')[-1].split('.')[0]
        name = f"{base}_{seg_idx}"
        input_rgb = os.path.join(fgr_segs, f)
        input_alpha = os.path.join(pha_segs, f)
        output_rgb = os.path.join(out_root["fgr"], f"{name}_blurred_rgb.mp4")
        output_alpha = os.path.join(out_root["pha"], f"{name}_blurred_alpha.mp4")
        output_depth = os.path.join(out_root["depth"], f"{name}_depth.mp4")
        try:
            process_video_blur(input_rgb, input_alpha, output_rgb, output_alpha, output_depth, model, device, input_size)
        except Exception as e:
            print(f"❌ Skipped segment {name} due to: {e}")
            torch.cuda.empty_cache(); gc.collect()

def process_video_blur(input_rgb, input_alpha, output_rgb, output_alpha, output_depth, model, device, input_size=518):
    chunk = estimate_chunk_size()
    sigma = random.uniform(2.0, 5.0)
    print(f"  🧠 chunk={chunk} | 🎯 sigma={sigma:.2f}")
    cap_rgb = cv2.VideoCapture(input_rgb)
    cap_alpha = cv2.VideoCapture(input_alpha)
    W, H = int(cap_rgb.get(3)), int(cap_rgb.get(4))
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    total = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    out_rgb = cv2.VideoWriter(output_rgb, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    out_alpha = cv2.VideoWriter(output_alpha, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    out_depth = cv2.VideoWriter(output_depth, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    ref_rgb, ref_alpha = [], []
    for _ in range(min(10, int(total * 0.05))):
        ret1, rgb = cap_rgb.read(); ret2, alpha = cap_alpha.read()
        if not (ret1 and ret2): break
        ref_rgb.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)); ref_alpha.append(alpha)

    init_depth = get_video_depths_chunked(ref_rgb, model, input_size=input_size, device=device, chunk_size=chunk)
    focus_vals, spreads = zip(*[compute_focus_and_spread(cv2.cvtColor(ref_alpha[i], cv2.COLOR_BGR2GRAY), init_depth[i]) for i in range(len(init_depth))])
    focus = np.mean(focus_vals)
    spread = max(np.median(spreads) * 0.8, 0.03)

    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0); cap_alpha.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rgb_frames, alpha_frames = [], []
    for _ in range(total):
        ret1, rgb = cap_rgb.read(); ret2, alpha = cap_alpha.read()
        if not (ret1 and ret2): break
        rgb_frames.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)); alpha_frames.append(alpha)

    depth_maps = get_video_depths_chunked(rgb_frames, model, input_size=input_size, device=device, chunk_size=chunk)

    for t in range(total):
        rgb = cv2.cvtColor(rgb_frames[t], cv2.COLOR_RGB2BGR)
        alpha = alpha_frames[t]
        alpha_gray = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
        depth = temporally_smooth_depth(depth_maps, t)
        out_depth.write(cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        out_rgb.write(apply_depth_blur(rgb, depth, focus, spread, sigma))
        out_alpha.write(cv2.cvtColor(apply_depth_blur(alpha_gray, depth, focus, spread, sigma), cv2.COLOR_GRAY2BGR))

    release_writers(out_rgb, out_alpha, out_depth)
    cap_rgb.release(); cap_alpha.release()
    del rgb_frames, alpha_frames, depth_maps
    torch.cuda.empty_cache(); gc.collect()

def is_valid_video(path):
    """Check if a video file is valid and can be properly read."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return False
    
    # Check for valid FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return False
    
    # Check for valid frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return False
    
    # Check for valid dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return False
    
    # Try to read at least one frame
    ret, _ = cap.read()
    cap.release()
    return ret

# Path to log corrupt videos
corrupt_log_path = "./corrupt_videos.txt"

def log_corrupt_video(video_name):
    """Log corrupt video to separate file"""
    with open(corrupt_log_path, "a") as f:
        f.write(video_name + "\n")

# Function to get log path for a specific split
def get_completed_log_path(split):
    """Get the path to the completed videos log for a specific split"""
    return f"./completed_videos_{split}.txt"

# Function to load completed videos for a specific split
def load_completed_videos(split):
    """Load the set of completed videos for a specific split"""
    log_path = get_completed_log_path(split)
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return set(f.read().splitlines())
    return set()

# Function to log a completed video for a specific split
def log_completed_video(split, base):
    """Log a completed video for a specific split"""
    log_path = get_completed_log_path(split)
    with open(log_path, "a") as f:
        f.write(base + "\n")

# 🔁 Main execution with restart trigger
if __name__ == "__main__":
    vda_ckpt = "./Video_Depth_Anything/checkpoints/video_depth_anything_vitl.pth"
    source_root = "./data/video_composed"
    target_root = "./data/video_defocused"
    splits = ["train", "test"]

    model, device = load_vda_model(vda_ckpt)
    
    # Create corrupt videos log if it doesn't exist
    if not os.path.exists(corrupt_log_path):
        open(corrupt_log_path, "w").close()
    
    # Load already identified corrupt videos
    corrupt_videos = set(open(corrupt_log_path).read().splitlines()) if os.path.exists(corrupt_log_path) else set()

    processed_count = 0  # 🆕 Track how many we processed in this run

    for split in splits:
        print(f"\n📂 Processing split: {split}")
        
        # Load completed videos specifically for this split
        completed = load_completed_videos(split)
        print(f"   Found {len(completed)} previously completed videos for {split} split")
        
        fgr_dir = os.path.join(source_root, split, "fgr")
        pha_dir = os.path.join(source_root, split, "pha")
        out_fgr = os.path.join(target_root, split, "fgr")
        out_pha = os.path.join(target_root, split, "pha")
        out_depth = os.path.join(target_root, split, "depth")
        os.makedirs(out_fgr, exist_ok=True)
        os.makedirs(out_pha, exist_ok=True)
        os.makedirs(out_depth, exist_ok=True)

        files = sorted([f for f in os.listdir(fgr_dir) if f.endswith(".mp4")])
        print(f"   Found {len(files)} videos in {split} split")

        for name in tqdm(files, desc=f"🎬 {split}"):
            base = os.path.splitext(name)[0]
            
            # Skip if already processed or identified as corrupt
            if base in completed or base in corrupt_videos:
                continue

            fgr_path = os.path.join(fgr_dir, name)
            pha_path = os.path.join(pha_dir, name)
            
            # First verify that both files are valid videos
            if not (is_valid_video(fgr_path) and is_valid_video(pha_path)):
                print(f"⚠️ Skipping corrupt file: {base}")
                log_corrupt_video(base)
                continue

            cap = cv2.VideoCapture(fgr_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            # Additional safety check for zero fps or frame count
            if fps <= 0 or frame_count <= 0:
                print(f"⚠️ Invalid video properties for {base}: fps={fps}, frames={frame_count}")
                log_corrupt_video(base)
                cap.release()
                continue
                
            duration = frame_count / fps
            cap.release()

            try:
                if duration > SPLIT_THRESHOLD_SEC:
                    fgr_out = get_segmented_paths(fgr_path, split, "fgr")
                    pha_out = get_segmented_paths(pha_path, split, "pha")
                    os.makedirs(os.path.dirname(fgr_out), exist_ok=True)
                    os.makedirs(os.path.dirname(pha_out), exist_ok=True)
                    segment_video(fgr_path, fgr_out, base, fps)
                    segment_video(pha_path, pha_out, base, fps)
                    process_segments(base, split, model, device, 518, fgr_out, pha_out,
                                     {"fgr": out_fgr, "pha": out_pha, "depth": out_depth})
                else:
                    process_video_blur(
                        fgr_path, pha_path,
                        os.path.join(out_fgr, f"{base}_blurred_rgb.mp4"),
                        os.path.join(out_pha, f"{base}_blurred_alpha.mp4"),
                        os.path.join(out_depth, f"{base}_depth.mp4"),
                        model, device)

                # Log completion to split-specific log file
                log_completed_video(split, base)
                processed_count += 1

                if processed_count >= MAX_VIDEOS_BEFORE_RESTART:
                    print("🔁 Batch complete, restarting after 10s...")
                    sys.exit(1)  # 🔁 Exit to restart the script

            except Exception as e:
                print(f"❌ Skipped {base} due to: {e}")
                continue

    print("✅ All videos processed.")
    sys.exit(0)