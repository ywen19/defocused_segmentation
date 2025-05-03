import os
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

import sys
sys.path.append(os.path.abspath("../sam2"))
sys.path.append(os.path.abspath("../MatAnyone"))
from sam2.sam2_image_predictor import SAM2ImagePredictor
from matanyone.utils.inference_utils import gen_dilate, gen_erosion, read_frame_from_videos
from matanyone.inference.inference_core import InferenceCore
from matanyone.utils.get_default_model import get_matanyone_model

import torch.nn.functional as F
import tqdm
import imageio

# Ensure efficient memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@torch.inference_mode()
@torch.autocast("cuda")
def run_full_pipeline(video_path, frame_path, output_dir, max_size=720):
    os.makedirs(output_dir, exist_ok=True)

    # === YOLO Detection ===
    print("Loading YOLO on cuda...")
    yolo_model = YOLO("yolov8n.pt")
    results = yolo_model(frame_path)
    detections = results[0].boxes
    person_boxes = [box.xyxy.cpu().numpy().astype(int).squeeze() for box in detections if int(box.cls) == 0]
    print(f"Detected {len(person_boxes)} person(s) with YOLO.")
    if len(person_boxes) == 0:
        raise ValueError("No person detected in the frame.")

    # === SAM2 Inference ===
    print("Loading SAM2...")
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    image = Image.open(frame_path).convert("RGB")
    image_np = np.array(image)
    predictor.set_image(image_np)

    print("Running SAM2...")
    mask_path = os.path.join(output_dir, "first_frame_mask.png")
    box_tensor = torch.tensor(np.array(person_boxes), dtype=torch.float32, device=predictor.device)
    masks, _, _ = predictor.predict(box=box_tensor, multimask_output=False)
    combined_mask = np.any(masks.astype(bool), axis=0).astype(np.uint8) * 255
    if combined_mask.ndim == 3:
        combined_mask = combined_mask[0]  # handle shape (1, H, W)
    Image.fromarray(combined_mask.astype(np.uint8), mode='L').save(mask_path)
    Image.fromarray(combined_mask.squeeze()).save(mask_path)
    print(f"Saved combined binary mask to {mask_path}")

    # === MatAnyone Inference ===
    print("Running MatAnyone...")

    # Download model
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id="pq-yang/MatAnyone", filename="matanyone.pth", local_dir="pretrained_models", local_dir_use_symlinks=False)

    matanyone_model = get_matanyone_model(ckpt_path)
    processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)

    vframes, fps, length, video_name = read_frame_from_videos(video_path)
    repeated_frames = vframes[0].unsqueeze(0).repeat(10, 1, 1, 1)
    vframes = torch.cat([repeated_frames, vframes], dim=0).float()
    length += 10

    new_h, new_w = None, None
    if max_size > 0:
        h, w = vframes.shape[-2:]
        min_side = min(h, w)
        if min_side > max_size:
            new_h = int(h / min_side * max_size)
            new_w = int(w / min_side * max_size)
            vframes = F.interpolate(vframes, size=(new_h, new_w), mode="area")

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    mask = torch.from_numpy(mask).cuda()

    if max_size > 0:
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest")[0, 0]

    bgr = (np.array([120, 255, 155], dtype=np.float32) / 255).reshape((1, 1, 3))
    objects = [1]
    phas, fgrs = [], []

    for ti in tqdm.tqdm(range(length)):
        image = vframes[ti]
        image_np = np.array(image.permute(1, 2, 0))
        image = (image / 255.).cuda().float()

        if ti == 0:
            processor.step(image, mask, objects=objects)
            output_prob = processor.step(image, first_frame_pred=True)
        else:
            if ti <= 10:
                output_prob = processor.step(image, first_frame_pred=True)
            else:
                output_prob = processor.step(image)

        pred_mask = processor.output_prob_to_mask(output_prob)
        pha = pred_mask.unsqueeze(2).cpu().numpy()
        com_np = image_np / 255. * pha + bgr * (1 - pha)

        if ti > 9:
            pha = (pha * 255).astype(np.uint8)
            com_np = (com_np * 255).astype(np.uint8)
            phas.append(pha)
            fgrs.append(com_np)

    out_pha = np.array(phas)
    out_fgr = np.array(fgrs)
    out_pha_rgb = np.repeat(out_pha, 3, axis=3)

    imageio.mimwrite(os.path.join(output_dir, f"{video_name}_pha.mp4"), out_pha_rgb, fps=fps, codec='libx264')
    imageio.mimwrite(os.path.join(output_dir, f"{video_name}_fgr.mp4"), out_fgr, fps=fps, codec='libx264')
    print("Matting completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--frame", type=str, default=None, help="Path to first frame image")
    parser.add_argument("--output_dir", type=str, default="./matanyone_results", help="Output directory")
    parser.add_argument("--max_size", type=int, default=720, help="Resize video if smaller side > max_size")
    args = parser.parse_args()

    if args.frame is None:
        cap = cv2.VideoCapture(args.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not extract frame from video.")
        frame_path = os.path.join(args.output_dir, "first_frame.png")
        os.makedirs(args.output_dir, exist_ok=True)
        cv2.imwrite(frame_path, frame)
    else:
        frame_path = args.frame

    run_full_pipeline(args.video_path, frame_path, args.output_dir, max_size=args.max_size)
