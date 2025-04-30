"""
Parallel compositing of VideoMatte240K using multiprocessing and FFmpeg.
This version uses FFmpeg for encoding to avoid OpenCV VideoWriter issues.
"""
import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import subprocess
import tempfile
import shutil

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

# Check if ffmpeg is available
def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except:
        return False

# Process one video pair using FFmpeg approach
def process_video_pair(args):
    fgr_path, pha_path, output_fgr_path, output_pha_path = args
    
    # Print which files we're processing
    print(f"Processing: {fgr_path.name}")
    
    # Skip if output already exists and is valid size
    if Path(output_fgr_path).exists() and Path(output_pha_path).exists():
        if Path(output_fgr_path).stat().st_size > 1000 and Path(output_pha_path).stat().st_size > 1000:
            print(f"Skipping {fgr_path.name} - already processed")
            return True
    
    try:
        # Load background image
        bg_img = cv2.imread(str(random.choice(background_images)))
        if bg_img is None:
            print(f"⚠️ Could not load background image")
            return False
        
        # Open video files
        cap_fg = cv2.VideoCapture(str(fgr_path))
        cap_alpha = cv2.VideoCapture(str(pha_path))
        
        # Check if videos are valid
        if not cap_fg.isOpened() or not cap_alpha.isOpened():
            print(f"⚠️ Couldn't open video: {fgr_path}")
            return False
            
        # Read video properties
        width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_fg.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check for valid properties
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            print(f"⚠️ Invalid video properties for {fgr_path.name}: w={width}, h={height}, fps={fps}, frames={frame_count}")
            return False

        # Create temporary directories for frame sequences
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define directories for frames
            fgr_frames_dir = Path(temp_dir) / "fgr_frames"
            pha_frames_dir = Path(temp_dir) / "pha_frames"
            os.makedirs(fgr_frames_dir, exist_ok=True)
            os.makedirs(pha_frames_dir, exist_ok=True)
            
            # Process frames and save as images
            frame_idx = 0
            while True:
                ret_fg, frame_fg = cap_fg.read()
                ret_alpha, frame_alpha = cap_alpha.read()
                
                if not ret_fg or not ret_alpha:
                    break
                
                # Convert alpha to grayscale for compositing
                frame_alpha_gray = cv2.cvtColor(frame_alpha, cv2.COLOR_BGR2GRAY)
                
                # Do the compositing
                composited = composite_foreground_background(frame_fg, frame_alpha_gray, bg_img)
                
                # Make alpha 3-channel for consistent output
                alpha_3channel = cv2.cvtColor(frame_alpha_gray, cv2.COLOR_GRAY2BGR)
                
                # Save frames as images
                fgr_frame_path = fgr_frames_dir / f"frame_{frame_idx:05d}.png"
                pha_frame_path = pha_frames_dir / f"frame_{frame_idx:05d}.png"
                
                cv2.imwrite(str(fgr_frame_path), composited)
                cv2.imwrite(str(pha_frame_path), alpha_3channel)
                
                frame_idx += 1
            
            # Close video capture
            cap_fg.release()
            cap_alpha.release()
            
            # Check if we have frames
            if frame_idx == 0:
                print(f"⚠️ No frames processed for {fgr_path.name}")
                return False
            
            # Make temporary output files
            temp_fgr_output = str(output_fgr_path) + ".temp.mp4"
            temp_pha_output = str(output_pha_path) + ".temp.mp4"
            
            # Use FFmpeg to convert images to video
            try:
                # FFmpeg command for FGR
                ffmpeg_cmd_fgr = [
                    'ffmpeg', '-y',  # Overwrite output files
                    '-framerate', str(fps),
                    '-i', str(fgr_frames_dir / "frame_%05d.png"),
                    '-c:v', 'libx264',  # Use H.264 codec
                    '-pix_fmt', 'yuv420p',  # Standard pixel format
                    '-crf', '23',  # Quality setting (lower is better)
                    temp_fgr_output
                ]
                
                # FFmpeg command for alpha
                ffmpeg_cmd_pha = [
                    'ffmpeg', '-y',  # Overwrite output files
                    '-framerate', str(fps),
                    '-i', str(pha_frames_dir / "frame_%05d.png"),
                    '-c:v', 'libx264',  # Use H.264 codec
                    '-pix_fmt', 'yuv420p',  # Standard pixel format
                    '-crf', '23',  # Quality setting (lower is better)
                    temp_pha_output
                ]
                
                # Execute FFmpeg commands
                subprocess.run(ffmpeg_cmd_fgr, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                subprocess.run(ffmpeg_cmd_pha, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Check if output files exist and have proper size
                if (os.path.exists(temp_fgr_output) and os.path.exists(temp_pha_output) and
                    os.path.getsize(temp_fgr_output) > 1000 and os.path.getsize(temp_pha_output) > 1000):
                    
                    # Rename temp files to final names
                    os.rename(temp_fgr_output, output_fgr_path)
                    os.rename(temp_pha_output, output_pha_path)
                    print(f"✅ Successfully processed {fgr_path.name} with FFmpeg ({frame_idx} frames)")
                    return True
                else:
                    print(f"⚠️ FFmpeg created invalid output for {fgr_path.name}")
                    # Remove invalid files
                    Path(temp_fgr_output).unlink(missing_ok=True)
                    Path(temp_pha_output).unlink(missing_ok=True)
                    return False
                
            except subprocess.CalledProcessError as e:
                print(f"⚠️ FFmpeg error for {fgr_path.name}: {e}")
                # Remove temp files
                Path(temp_fgr_output).unlink(missing_ok=True)
                Path(temp_pha_output).unlink(missing_ok=True)
                return False
                
    except Exception as e:
        print(f"❌ Unexpected error processing {fgr_path.name}: {str(e)}")
        # Clean up temp files
        if 'temp_fgr_output' in locals():
            Path(temp_fgr_output).unlink(missing_ok=True)
        if 'temp_pha_output' in locals():
            Path(temp_pha_output).unlink(missing_ok=True)
        return False

# For setting up global background_images per process
def init_pool(images):
    global background_images
    background_images = images

if __name__ == "__main__":
    # Check if FFmpeg is available
    if not check_ffmpeg():
        print("❌ FFmpeg is not installed or not in PATH. Please install FFmpeg first.")
        exit(1)
    
    root_dir = Path("../data/VideoMatte240K")
    backgrounds_dir = root_dir / "Backgrounds"
    output_root = Path("../data/video_composed")
    
    # Ensure output folder structure
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'test']:
        (output_root / split / 'fgr').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'pha').mkdir(parents=True, exist_ok=True)
    
    # Load background images
    bg_images = list(backgrounds_dir.glob("*.jpg")) + list(backgrounds_dir.glob("*.png"))
    assert bg_images, "No background images found!"
    print(f"Found {len(bg_images)} background images")
    
    # Prepare jobs
    all_jobs = []
    
    # Process specific range from train set (0402-0478)
    fgr_train_videos = sorted((root_dir / "train" / "fgr").glob("*.mp4"))
    pha_train_videos = sorted((root_dir / "train" / "pha").glob("*.mp4"))
    
    # Filter train videos to only include 0402-0478
    filtered_train_pairs = []
    for fgr_path, pha_path in zip(fgr_train_videos, pha_train_videos):
        try:
            video_num = int(fgr_path.stem)
            if 402 <= video_num <= 478:
                filtered_train_pairs.append((fgr_path, pha_path))
        except ValueError:
            print(f"Skipping non-numeric filename: {fgr_path.name}")
    
    print(f"Processing {len(filtered_train_pairs)} videos from train set (range 0402-0478)")
    
    for fgr_path, pha_path in filtered_train_pairs:
        name = fgr_path.stem
        output_fgr = output_root / "train" / "fgr" / f"{name}.mp4"
        output_pha = output_root / "train" / "pha" / f"{name}.mp4"
        all_jobs.append((fgr_path, pha_path, output_fgr, output_pha))
    
    # Process all from test set
    fgr_test_videos = sorted((root_dir / "test" / "fgr").glob("*.mp4"))
    pha_test_videos = sorted((root_dir / "test" / "pha").glob("*.mp4"))
    
    print(f"Processing all {len(fgr_test_videos)} videos from test set")
    
    for fgr_path, pha_path in zip(fgr_test_videos, pha_test_videos):
        name = fgr_path.stem
        output_fgr = output_root / "test" / "fgr" / f"{name}.mp4"
        output_pha = output_root / "test" / "pha" / f"{name}.mp4"
        all_jobs.append((fgr_path, pha_path, output_fgr, output_pha))
    
    # Multiprocessing with manual tqdm update
    # Use fewer processes because FFmpeg is resource-intensive
    num_processes = max(1, min(cpu_count() // 2, 4))
    print(f"Using {num_processes} processes")
    
    with tqdm(total=len(all_jobs), desc="Compositing videos") as pbar:
        # Use a smaller batch size for better error handling
        with Pool(processes=num_processes, initializer=init_pool, initargs=(bg_images,)) as pool:
            # Process videos in smaller batches to better handle errors
            for i in range(0, len(all_jobs), 5):  # Process 5 at a time
                batch = all_jobs[i:i+5]
                results = []
                for result in pool.imap_unordered(process_video_pair, batch):
                    results.append(result)
                    pbar.update(1)
                
                # Print batch summary
                successful = results.count(True)
                print(f"Batch {i//5 + 1}: {successful}/{len(batch)} videos processed successfully")
                
    print("✅ Video compositing complete!")