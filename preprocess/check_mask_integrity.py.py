import os

def check_masks(data_root):
    missing = []
    for split in ["train", "test"]:
        fgr_dir = os.path.join(data_root, split, "fgr")
        if not os.path.exists(fgr_dir):
            print(f"[WARN] Missing directory: {fgr_dir}")
            continue

        for video_id in sorted(os.listdir(fgr_dir)):
            video_path = os.path.join(fgr_dir, video_id)
            if not os.path.isdir(video_path):
                continue

            mask_path = os.path.join(video_path, "mask", "first_frame_mask.png")
            if not os.path.exists(mask_path):
                missing.append(mask_path)

    if not missing:
        print("✅ All mask files are present.")
    else:
        print(f"❌ Missing {len(missing)} mask file(s):")
        for path in missing:
            print(f" - {path}")

if __name__ == "__main__":
    # 修改为你的数据根路径
    data_root = "../data/video_defocused_processed"
    check_masks(data_root)
