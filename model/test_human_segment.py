import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import sys
sys.path.append(os.path.abspath("../sam2"))
# Import SAM2 components
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    parser = argparse.ArgumentParser(description="Test SAM2 with different prompts")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="facebook/sam2-hiera-large", 
                        help="Model name or path")
    parser.add_argument("--output", type=str, default="./sam2_output", 
                        help="Output directory")
    parser.add_argument("--prompt_type", type=str, default="point", 
                        choices=["point", "box", "text"], 
                        help="Type of prompt to use")
    parser.add_argument("--point_coords", type=str, default=None, 
                        help="Comma-separated list of point coordinates (x1,y1,x2,y2,...)")
    parser.add_argument("--box_coords", type=str, default=None, 
                        help="Comma-separated box coordinates (x1,y1,x2,y2)")
    parser.add_argument("--text", type=str, default=None, 
                        help="Text prompt describing what to segment")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to run on (cuda or cpu)")
    return parser.parse_args()

def load_image(image_path):
    """Load an image and convert to RGB numpy array."""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)

def visualize_results(image, masks, scores=None, output_path=None):
    """Visualize segmentation results."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Get color map for masks
    cmap = plt.cm.get_cmap('jet', masks.shape[0])
    
    for i, mask in enumerate(masks):
        # Create colored overlay
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color = np.array([int(c * 255) for c in cmap(i)[:3]])
        
        for c in range(3):
            color_mask[:, :, c] = mask * color[c]
        
        # Apply overlay with transparency
        alpha = 0.5
        plt.imshow(color_mask, alpha=alpha)
        
        # Show confidence score
        if scores is not None:
            score = scores[i]
            plt.text(10, 30 + i * 20, f"Mask {i+1}: {score:.3f}", 
                     fontsize=12, color='white', 
                     bbox=dict(facecolor='black', alpha=0.5))
    
    plt.axis('off')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved visualization to {output_path}")
    
    plt.show()

def run_sam2_test(args):
    """Run SAM2 test with the specified prompt type."""
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Loading SAM2 model {args.model} on {device}...")
    predictor = SAM2ImagePredictor.from_pretrained(args.model)
    
    # Load and prepare image
    print(f"Loading image from {args.image}...")
    image = load_image(args.image)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run inference based on prompt type
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        print(f"Setting image in predictor...")
        predictor.set_image(image)
        
        if args.prompt_type == "point":
            if args.point_coords:
                # Parse point coordinates
                coords = [int(c) for c in args.point_coords.split(',')]
                points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                point_coords = np.array(points)
                point_labels = np.ones(len(points))
            else:
                # Default to center point
                h, w = image.shape[:2]
                point_coords = np.array([[w // 2, h // 2]])
                point_labels = np.array([1])
            
            # Convert to tensors
            point_coords = torch.as_tensor(point_coords, device=device)
            point_labels = torch.as_tensor(point_labels, device=device)
            
            print(f"Running prediction with point prompt...")
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            
        elif args.prompt_type == "box":
            if args.box_coords:
                # Parse box coordinates
                box = np.array([[int(c) for c in args.box_coords.split(',')]])
            else:
                # Default to center box covering 50% of image
                h, w = image.shape[:2]
                box = np.array([[w // 4, h // 4, 3 * w // 4, 3 * h // 4]])
            
            # Convert to tensor
            box = torch.as_tensor(box, device=device)
            
            print(f"Running prediction with box prompt...")
            masks, scores, _ = predictor.predict(
                boxes=box,
                multimask_output=True
            )
            
        elif args.prompt_type == "text":
            if args.text:
                text = args.text
            else:
                text = "the main object in the image"
            
            print(f"Running prediction with text prompt: '{text}'...")
            masks, scores, _ = predictor.predict(
                text=text,
                multimask_output=True
            )
    
    # Convert results to numpy
    masks_np = masks.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    # Visualize and save results
    output_path = os.path.join(args.output, f"sam2_{args.prompt_type}_result.png")
    visualize_results(image, masks_np, scores_np, output_path)
    
    print("SAM2 test completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    run_sam2_test(args)