"""
Parallel compositing of VideoMatte240K using multiprocessing.
This version ensures tqdm progress bar updates correctly.
"""

import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager

# Global variable for background images
background_images = []

# Composite function
def composite_foreground_background(fg, alpha, bg):
    alpha = alpha.astype(np.float32) / 255.0
    alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    fg = fg.astype(np.float32)
    bg = cv2.resize(bg, (fg.shape[1], fg.shape[0])).astype(np.float32)
    comp = fg * alpha_3ch + bg * (1 - alpha_3ch)
    return comp.astype(np.uint8)

# Process one video pair
def process_video_pair(args):
    fgr_path, pha_path, output_fgr_path, output_pha_path = args
    bg_img = cv2.imread(str(random.choice(background_images)))

    cap_fg = cv2.VideoCapture(str(fgr_path))
    cap_alpha = cv2.VideoCapture(str(pha_path))

    width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_fg.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_fg = cv2.VideoWriter(str(output_fgr_path), fourcc, fps, (width, height))
    out_alpha = cv2.VideoWriter(str(output_pha_path), fourcc, fps, (width, height), isColor=False)

    while True:
        ret_fg, frame_fg = cap_fg.read()
        ret_alpha, frame_alpha = cap_alpha.read()
        if not ret_fg or not ret_alpha:
            break
        frame_alpha_gray = cv2.cvtColor(frame_alpha, cv2.COLOR_BGR2GRAY)
        composited = composite_foreground_background(frame_fg, frame_alpha_gray, bg_img)
        out_fg.write(composited)
        out_alpha.write(frame_alpha_gray)

    cap_fg.release()
    cap_alpha.release()
    out_fg.release()
    out_alpha.release()
    return True  # indicate success

# For setting up global background_images per process
def init_pool(images):
    global background_images
    background_images = images


if __name__ == "__main__":
    from functools import partial

    root_dir = Path("../data/VideoMatte240K")
    backgrounds_dir = root_dir / "Backgrounds"
    output_root = Path("../data/video_defocused")

    # Ensure output folder structure
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'test']:
        (output_root / split / 'fgr').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'pha').mkdir(parents=True, exist_ok=True)

    # Load background images
    bg_images = list(backgrounds_dir.glob("*.jpg")) + list(backgrounds_dir.glob("*.png"))
    assert bg_images, "No background images found!"

    # Prepare jobs
    all_jobs = []
    for split in ["train", "test"]:
        fgr_videos = sorted((root_dir / split / "fgr").glob("*.mp4"))
        pha_videos = sorted((root_dir / split / "pha").glob("*.mp4"))
        assert len(fgr_videos) == len(pha_videos), f"Mismatch in video counts for {split}"

        for fgr_path, pha_path in zip(fgr_videos, pha_videos):
            name = fgr_path.stem
            output_fgr = output_root / split / "fgr" / f"{name}.mp4"
            output_pha = output_root / split / "pha" / f"{name}.mp4"
            all_jobs.append((fgr_path, pha_path, output_fgr, output_pha))

    # Multiprocessing with manual tqdm update
    with tqdm(total=len(all_jobs), desc="Compositing videos") as pbar:
        with Pool(processes=cpu_count(), initializer=init_pool, initargs=(bg_images,)) as pool:
            for _ in all_jobs:
                pool.apply_async(
                    process_video_pair,
                    args=(_,),
                    callback=lambda _: pbar.update()
                )
            pool.close()
            pool.join()
