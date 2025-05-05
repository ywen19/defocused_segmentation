"""
We prepare the csv for: rgb--yolo+sam2 mask--groundtruth matte for batch loading.
"""

import os
import csv
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

def is_all_black(image_path):
    try:
        img = Image.open(image_path).convert('L')
        arr = np.array(img)
        return np.max(arr) == 0
    except:
        return True

def generate_csv(data_root, output_csv):
    fgr_dir = os.path.join(data_root, 'fgr')
    alpha_dir = os.path.join(data_root, 'alpha')

    raw_rows = []
    total_masks = 0
    missing_pairs = 0

    for seq in sorted(os.listdir(fgr_dir)):
        fgr_rgb_dir = os.path.join(fgr_dir, seq, 'frames')
        fgr_mask_dir = os.path.join(fgr_dir, seq, 'mask')
        gt_dir = os.path.join(alpha_dir, seq, 'frames')

        if not (os.path.isdir(fgr_rgb_dir) and os.path.isdir(fgr_mask_dir) and os.path.isdir(gt_dir)):
            continue

        for fname in sorted(os.listdir(fgr_mask_dir)):
            total_masks += 1

            rgb = os.path.join(fgr_rgb_dir, fname)
            init_mask = os.path.join(fgr_mask_dir, fname)
            gt = os.path.join(gt_dir, fname)

            if not (os.path.exists(rgb) and os.path.exists(gt) and os.path.exists(init_mask)):
                missing_pairs += 1
                continue

            raw_rows.append({'rgb': rgb, 'gt': gt, 'init_mask': init_mask})

    print(f"\n📦 Total masks found: {total_masks}")
    print(f"❌ Skipped due to missing RGB/GT: {missing_pairs}")
    print(f"🔍 Valid candidate samples for further check: {len(raw_rows)}")

    valid_rows = []
    skipped_black = 0
    failed = 0

    for row in tqdm(raw_rows, desc="Validating samples"):
        try:
            rgb, gt, mask = row['rgb'], row['gt'], row['init_mask']

            if is_all_black(gt):
                skipped_black += 1
                continue

            # Load and verify size
            rgb_img = Image.open(rgb)
            gt_img = Image.open(gt)
            mask_img = Image.open(mask)

            if not (rgb_img.size == gt_img.size == mask_img.size):
                raise ValueError("Size mismatch")

            valid_rows.append(row)

        except Exception as e:
            failed += 1
            print(f"[SKIP] {row['rgb']} | Reason: {e}")

    print(f"\n✅ Valid image-mask-matte triplets: {len(valid_rows)}")
    print(f"❌ Skipped (black matte): {skipped_black}")
    print(f"❌ Skipped (loading errors): {failed}")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rgb', 'gt', 'init_mask'])
        writer.writeheader()
        writer.writerows(valid_rows)

    print(f"\n📁 CSV saved to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and validate CSV for matte training.")
    parser.add_argument('--data_root', type=str, default='../data/video_defocused_processed/train',
                        help="Root folder containing 'fgr' and 'alpha'")
    parser.add_argument('--output_csv', type=str, default='../data/pair_for_refiner.csv',
                        help="Path to output CSV file")

    args = parser.parse_args()
    generate_csv(args.data_root, args.output_csv)
