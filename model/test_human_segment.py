import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from ultralytics import YOLO

import sys
sys.path.append(os.path.abspath("../sam2"))
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Human segmentation using YOLO and SAM2")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="facebook/sam2-hiera-large", help="SAM2 model name or path")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLOv8 model path or name")
    parser.add_argument("--output", type=str, default="./sam2_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def detect_humans_with_yolo(image_path, yolo_model_path, device):
    print(f"Loading YOLO model from {yolo_model_path} on {device}...")
    model = YOLO(yolo_model_path)
    results = model(image_path, device=0 if device == "cuda" else "cpu")[0]

    person_boxes = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls.item()) == 0:  # 0 is the class index for "person"
            person_boxes.append(box.cpu().numpy())

    print(f"Found {len(person_boxes)} person(s).")
    return np.array(person_boxes)


def visualize_results(image, masks, scores, output_path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    cmap = plt.cm.get_cmap('jet', masks.shape[0])

    for i, mask in enumerate(masks):
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        color = np.array([int(c * 255) for c in cmap(i)[:3]])
        for c in range(3):
            color_mask[:, :, c] = mask * color[c]

        plt.imshow(color_mask, alpha=0.5)

        if scores is not None:
            plt.text(10, 30 + i * 20, f"Mask {i+1}: {scores[i]:.3f}",
                     fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5))

    plt.axis('off')
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved visualization to {output_path}")
    plt.show()


def run_sam2_test(args):
    image = load_image(args.image)
    person_boxes = detect_humans_with_yolo(args.image, args.yolo_model, args.device)

    if len(person_boxes) == 0:
        print("No humans detected.")
        return

    print(f"Running SAM2 with bounding boxes...")

    predictor = SAM2ImagePredictor.from_pretrained(args.model)
    predictor.model.to(args.device)
    predictor.set_image(image)

    all_masks = []
    all_scores = []

    for box in person_boxes:
        box_tensor = torch.tensor(box, dtype=torch.float32, device=args.device).unsqueeze(0)

        with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32):
            masks, scores, _ = predictor.predict(
                box=box_tensor,
                multimask_output=True
            )

        # masks, scores are numpy arrays already
        all_masks.append(masks)
        all_scores.append(scores)

    masks_np = np.concatenate(all_masks, axis=0)
    scores_np = np.concatenate(all_scores, axis=0)

    output_path = os.path.join(args.output, "sam2_yolo_human_segmentation.png")
    visualize_results(image, masks_np, scores_np, output_path)


if __name__ == "__main__":
    args = parse_args()
    run_sam2_test(args)
