import os
import cv2
import torch
import numpy as np
from PIL import Image

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

YOLO_MEMORY_MB = 800
SAM2_MEMORY_MB = 2500

def get_first_frame_path(frame_dir):
    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    return os.path.join(frame_dir, frame_files[0]) if frame_files else None

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

def load_completed_log(split):
    path = f"completed_masks_{split}.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(f.read().splitlines())
    return set()

def log_completed(split, video_id):
    path = f"completed_masks_{split}.txt"
    with open(path, "a") as f:
        f.write(video_id + "\n")

def save_mask_and_log(mask_array, output_path, split, video_id):
    try:
        Image.fromarray(mask_array.astype(np.uint8), mode='L').save(output_path)
        if os.path.exists(output_path):
            log_completed(split, video_id)
            print(f"Saved mask to {output_path}")
        else:
            print(f"[ERROR] Failed to save mask to {output_path}")
    except Exception as e:
        print(f"[ERROR] Exception when saving mask to {output_path}: {e}")

def init_models():
    global yolo_model, predictor
    yolo_model = YOLO("yolov8n.pt")
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

@torch.inference_mode()
@torch.autocast("cuda")
def generate_mask_worker(args):
    frame_path, output_path, split, video_id = args
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if torch.cuda.memory_allocated() / (1024 ** 2) > torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) * 0.9:
            print(f"[SKIP] Memory usage too high. Skipping {video_id}")
            return

        results = yolo_model(frame_path)
        detections = results[0].boxes
        person_boxes = [box.xyxy.cpu().numpy().astype(int).squeeze() for box in detections if int(box.cls) == 0]

        image = Image.open(frame_path).convert("RGB")
        image_np = np.array(image)

        if len(person_boxes) == 0:
            print(f"No person detected in {frame_path}, creating empty mask.")
            empty_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
            save_mask_and_log(empty_mask, output_path, split, video_id)
            return

        predictor.set_image(image_np)

        box_tensor = torch.tensor(np.array(person_boxes), dtype=torch.float32, device=predictor.device)
        masks, _, _ = predictor.predict(box=box_tensor, multimask_output=False)

        combined_mask = np.any(masks.astype(bool), axis=0).astype(np.uint8) * 255
        if combined_mask.ndim == 3:
            combined_mask = combined_mask[0]

        save_mask_and_log(combined_mask, output_path, split, video_id)

        del image, image_np, masks, box_tensor
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print(f"[OOM] Failed processing {frame_path}, CUDA out of memory.")
        torch.cuda.empty_cache()
        return
    except Exception as e:
        print(f"[ERROR] Failed processing {frame_path}: {e}")
        return

def process_dataset(root_dir):
    tasks = []
    for split in ["train", "test"]:
        split_dir = os.path.join(root_dir, split, "fgr")
        completed = load_completed_log(split)

        for video_id in sorted(os.listdir(split_dir)):
            if video_id in completed:
                continue
            frame_dir = os.path.join(split_dir, video_id, "frames")

            if not os.path.isdir(frame_dir):
                continue

            # 🔽 New: check if frames is empty and delete both fgr and pha if so
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
            # 🔼 End new

            frame_path = os.path.join(frame_dir, sorted(frame_files)[0])
            mask_output_path = os.path.join(root_dir, split, "fgr", video_id, "mask", "first_frame_mask.png")
            tasks.append((frame_path, mask_output_path, split, video_id))

    max_workers = estimate_safe_workers()
    print(f"Starting multiprocessing on {len(tasks)} tasks with {max_workers} workers...")

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
    args = parser.parse_args()

    retries = 5
    for attempt in range(retries):
        print(f"[INFO] Attempt {attempt + 1}...")
        try:
            process_dataset(args.data_dir)
            break
        except torch.cuda.OutOfMemoryError as e:
            print(f"[WARN] CUDA OOM on attempt {attempt + 1}: {e}")
            torch.cuda.empty_cache()
            time.sleep(5)
        except Exception as e:
            print(f"[WARN] General failure on attempt {attempt + 1}: {e}")
            time.sleep(5)
