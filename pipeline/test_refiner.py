import argparse
import torch
from model.refiner import MatteRefiner
from main import infer_single
from utils import compute_mae, compute_sad, compute_grad, save_or_show_matte
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize
import json
import os


def load_model(ckpt_path, device):
    model = MatteRefiner().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def compute_metrics(pred, gt_path, resize_to, device):
    gt = Image.open(gt_path).convert("L")
    gt = resize(gt, resize_to)
    gt_tensor = to_tensor(gt).unsqueeze(0).to(device)

    pred = pred.unsqueeze(0).to(device)

    mae = compute_mae(pred, gt_tensor).item()
    sad = compute_sad(pred, gt_tensor).item()
    grad = compute_grad(pred, gt_tensor).item()

    return {"MAE": mae, "SAD": sad, "GRAD": grad}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    matte = infer_single(model, args.rgb, args.mask, device, resize_to=args.resize)

    # Save matte image
    save_or_show_matte(
        matte_tensor=matte,
        save_path=args.output,
        show=args.show,
        cmap="gray",
        title="Predicted Alpha Matte"
    )
    print(f"✅ Matte saved to: {args.output}")

    # Evaluate metrics
    if args.gt:
        metrics = compute_metrics(matte, args.gt, args.resize, device)
        print("📊 Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        if args.metrics_output:
            os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
            with open(args.metrics_output, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"📁 Metrics saved to: {args.metrics_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with MatteRefiner.")
    parser.add_argument("--rgb", type=str, required=True, help="Path to input RGB image")
    parser.add_argument("--mask", type=str, required=True, help="Path to initial mask (e.g. from YOLO+SAM)")
    parser.add_argument("--gt", type=str, help="Path to ground truth matte (for evaluation)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint .pth")
    parser.add_argument("--output", type=str, default="predicted_matte.png", help="Output path for predicted matte")
    parser.add_argument("--metrics-output", type=str, default="metrics.json", help="Path to save computed metrics")
    parser.add_argument("--resize", nargs=2, type=int, default=[720, 1280], help="Resize input (H W)")
    parser.add_argument("--show", action="store_true", help="Whether to display matte using matplotlib")

    args = parser.parse_args()
    main(args)
