import os
import cv2
import torch
import numpy as np
from PIL import Image
import time

import sys
sys.path.append(os.path.abspath("../sam2"))
from ultralytics import YOLO
from sam2.sam2_image_predictor import SAM2ImagePredictor

from pathlib import Path
import concurrent.futures
import multiprocessing
import signal
import subprocess


YOLO_MEMORY_MB = 800
SAM2_MEMORY_MB = 2500

def get_frame_paths(frame_dir, first_frame_only=False):
    """Get path(s) to frames that need processing."""
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    if not frame_files:
        return []
    
    if first_frame_only:
        return [os.path.join(frame_dir, frame_files[0])]
    else:
        return [os.path.join(frame_dir, f) for f in frame_files]

def estimate_safe_workers(mem_per_worker_mb=3500, safety_margin=0.9, min_workers=1, max_workers=8):
    if not torch.cuda.is_available():
        return min(max_workers, os.cpu_count())
    try:
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        free = total - max(reserved, allocated)
        usable = free * safety_margin
        est_workers = int(usable // mem_per_worker_mb)
        return max(min_workers, min(max_workers, est_workers))
    except Exception:
        return min(max_workers, os.cpu_count())

def load_completed_log(split, first_frame_only):
    """Load log of completed videos with mode-specific filename."""
    prefix = "first_frame" if first_frame_only else "all_frames"
    path = f"completed_masks_{prefix}_{split}.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(f.read().splitlines())
    return set()

def log_completed(split, video_id, frame_name=None, first_frame_only=True):
    """Log completed processing with mode-specific filename."""
    prefix = "first_frame" if first_frame_only else "all_frames"
    path = f"completed_masks_{prefix}_{split}.txt"
    
    # For all_frames mode, only log when all frames are done
    if not first_frame_only and frame_name is not None:
        # Log individual frames to a different file
        frames_path = f"completed_frames_{video_id}.txt"
        with open(frames_path, "a") as f:
            f.write(frame_name + "\n")
        return
        
    # Log the video as completed
    with open(path, "a") as f:
        f.write(video_id + "\n")

def get_completed_frames(video_id):
    """Get set of already processed frames for a video."""
    frames_path = f"completed_frames_{video_id}.txt"
    if os.path.exists(frames_path):
        with open(frames_path, "r") as f:
            return set(f.read().splitlines())
    return set()

def save_mask(mask_array, output_path):
    """Save mask to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(mask_array.astype(np.uint8), mode='L').save(output_path)
    return os.path.exists(output_path)

def init_models():
    global yolo_model, predictor
    yolo_model = YOLO("yolov8n.pt")
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

@torch.inference_mode()
@torch.autocast("cuda")
def generate_mask_worker(args):
    frame_path, output_path, split, video_id, frame_name, first_frame_only = args
    try:
        if torch.cuda.is_available() and torch.cuda.memory_allocated() / (1024 ** 2) > torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) * 0.9:
            print(f"[SKIP] Memory usage too high. Skipping {video_id}/{frame_name}")
            return

        results = yolo_model(frame_path)
        detections = results[0].boxes
        person_boxes = [box.xyxy.cpu().numpy().astype(int).squeeze() for box in detections if int(box.cls) == 0]

        image = Image.open(frame_path).convert("RGB")
        image_np = np.array(image)

        if len(person_boxes) == 0:
            print(f"No person detected in {frame_path}, creating empty mask.")
            empty_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
            success = save_mask(empty_mask, output_path)
        else:
            predictor.set_image(image_np)
            box_tensor = torch.tensor(np.array(person_boxes), dtype=torch.float32, device=predictor.device)
            masks, _, _ = predictor.predict(box=box_tensor, multimask_output=False)

            combined_mask = np.any(masks.astype(bool), axis=0).astype(np.uint8) * 255
            if combined_mask.ndim == 3:
                combined_mask = combined_mask[0]

            success = save_mask(combined_mask, output_path)

        if success:
            log_completed(split, video_id, frame_name, first_frame_only)
            print(f"Saved mask to {output_path}")
        else:
            print(f"[ERROR] Failed to save mask to {output_path}")

        del image, image_np
        if 'masks' in locals():
            del masks
        if 'box_tensor' in locals():
            del box_tensor
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print(f"[OOM] Failed processing {frame_path}, CUDA out of memory.")
        torch.cuda.empty_cache()
        return
    except Exception as e:
        print(f"[ERROR] Failed processing {frame_path}: {e}")
        return

def process_dataset(root_dir, first_frame_only=True):
    tasks = []
    for split in ["train", "test"]:
        split_dir = os.path.join(root_dir, split, "fgr")
        completed_videos = load_completed_log(split, first_frame_only)

        for video_id in sorted(os.listdir(split_dir)):
            if video_id in completed_videos:
                continue
                
            frame_dir = os.path.join(split_dir, video_id, "frames")
            if not os.path.isdir(frame_dir):
                continue

            # Check if frames is empty and delete both fgr and pha if so
            frame_files = [f for f in os.listdir(frame_dir) if f.endswith(".png")]
            if not frame_files:
                print(f"[CLEANUP] Empty frames in {video_id}, removing folders...")
                fgr_path = os.path.join(root_dir, split, "fgr", video_id)
                pha_path = os.path.join(root_dir, split, "pha", video_id)
                try:
                    import shutil
                    shutil.rmtree(fgr_path)
                    print(f" - Removed {fgr_path}")
                    if os.path.exists(pha_path):
                        shutil.rmtree(pha_path)
                        print(f" - Removed {pha_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to remove dirs for {video_id}: {e}")
                continue
                
            # Get frame paths based on mode
            frame_paths = get_frame_paths(frame_dir, first_frame_only)
            completed_frames = get_completed_frames(video_id) if not first_frame_only else set()
            
            for frame_path in frame_paths:
                frame_name = os.path.basename(frame_path)
                if not first_frame_only and frame_name in completed_frames:
                    continue
                    
                # Generate output path based on mode
                if first_frame_only:
                    mask_output_path = os.path.join(root_dir, split, "fgr", video_id, "mask", "first_frame_mask.png")
                else:
                    # Keep the same filename structure in the mask directory
                    mask_output_path = os.path.join(root_dir, split, "fgr", video_id, "mask", frame_name)
                
                tasks.append((frame_path, mask_output_path, split, video_id, frame_name, first_frame_only))

    max_workers = estimate_safe_workers()
    print(f"Starting multiprocessing on {len(tasks)} tasks with {max_workers} workers...")
    print(f"Mode: {'First Frame Only' if first_frame_only else 'All Frames'}")

    if not tasks:
        print("[INFO] No tasks to process. All videos have been completed.")
        return

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=max_workers, initializer=init_models) as pool:
        pool.map(generate_mask_worker, tasks)

    print("[INFO] ✅ All masks processed successfully.")


def kill_orphan_python_workers(script_name="yolo_sam2_mask_extract.py"):
    try:
        output = subprocess.check_output(f"ps aux | grep {script_name} | grep -v grep", shell=True)
        for line in output.decode().splitlines():
            pid = int(line.split()[1])
            if pid != os.getpid():
                os.kill(pid, signal.SIGKILL)
    except subprocess.CalledProcessError:
        pass

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    kill_orphan_python_workers("yolo_sam2_mask_extract.py")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to video_defocused_processed")
    parser.add_argument("--mode", type=str, choices=["first_frame", "all_frames"], 
                        default="first_frame", help="Generate mask for first frame only or all frames")
    args = parser.parse_args()

    first_frame_only = args.mode == "first_frame"
    
    retries = 5
    for attempt in range(retries):
        print(f"[INFO] Attempt {attempt + 1}...")
        try:
            process_dataset(args.data_dir, first_frame_only)
            break
        except torch.cuda.OutOfMemoryError as e:
            print(f"[WARN] CUDA OOM on attempt {attempt + 1}: {e}")
            torch.cuda.empty_cache()
            time.sleep(5)
        except Exception as e:
            print(f"[WARN] General failure on attempt {attempt + 1}: {e}")
            time.sleep(5)