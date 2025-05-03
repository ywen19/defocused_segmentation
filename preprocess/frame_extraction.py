import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import concurrent.futures
import torch
import argparse
import multiprocessing

# Set the start method to 'spawn' for CUDA compatibility with multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# --------- CONFIG ---------
sampling_ratio = (0.1, 0.3)  # Sample 10-30% of frames
batch_size = 16              # Process frames in batches
num_workers = 8              # Number of parallel CPU workers
use_cuda = torch.cuda.is_available()  # Auto-detect GPU

def parse_args():
    parser = argparse.ArgumentParser(description='Extract frames from defocused videos')
    parser.add_argument('--input_dir', type=str, default='../data/video_defocused', 
                        help='Input directory containing train and test folders')
    parser.add_argument('--output_dir', type=str, default='../data/video_defocused_processed', 
                        help='Output directory for extracted frames')
    parser.add_argument('--dataset', type=str, default='both', choices=['train', 'test', 'both'],
                        help='Which dataset to process (train, test, or both)')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    return parser.parse_args()

def sample_frame_indices(total_frames, ratio_range=(0.1, 0.3)):
    """Sample random frame indices based on the given ratio range"""
    min_ratio, max_ratio = ratio_range
    k = random.randint(int(min_ratio * total_frames), int(max_ratio * total_frames))
    return sorted(random.sample(range(total_frames), k))

def extract_and_save_frames(video_path, frame_indices, output_dir):
    """Extract specific frames from a video and save them to the output directory"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames in batches
    cap = cv2.VideoCapture(video_path)
    
    # Process frames in batches to manage memory
    for i in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[i:i+batch_size]
        
        for idx in batch_indices:
            # Set frame position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert to RGB and save
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                idx_str = f"{idx:04d}"
                
                # Save as PNG
                output_path = os.path.join(output_dir, f"{idx_str}.png")
                Image.fromarray(frame_rgb).save(output_path)
    
    # Release video capture
    cap.release()

def process_video_gpu(video_id, fgr_path, pha_path, output_fgr_dir, output_pha_dir):
    """Process a video using GPU acceleration"""
    try:
        device = torch.device("cuda")
        
        # Create output directories
        video_fgr_dir = os.path.join(output_fgr_dir, video_id)
        video_pha_dir = os.path.join(output_pha_dir, video_id)
        os.makedirs(video_fgr_dir, exist_ok=True)
        os.makedirs(video_pha_dir, exist_ok=True)
        
        # Get total frames
        cap = cv2.VideoCapture(fgr_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Sample frame indices
        frame_ids = sample_frame_indices(total_frames, sampling_ratio)
        
        # Process in batches
        for i in range(0, len(frame_ids), batch_size):
            batch_indices = frame_ids[i:i+batch_size]
            
            # Process FGR
            fgr_frames = []
            fgr_cap = cv2.VideoCapture(fgr_path)
            for idx in batch_indices:
                fgr_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = fgr_cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fgr_frames.append((idx, rgb))
            fgr_cap.release()
            
            # Process PHA
            pha_frames = []
            pha_cap = cv2.VideoCapture(pha_path)
            for idx in batch_indices:
                pha_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = pha_cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pha_frames.append((idx, rgb))
            pha_cap.release()
            
            # GPU processing for FGR frames
            if fgr_frames:
                indices = [x[0] for x in fgr_frames]
                frames = np.stack([x[1] for x in fgr_frames])
                
                # Convert to tensor and move to GPU
                tensor = torch.from_numpy(frames).to(device)
                
                # Move back to CPU and save
                processed = tensor.cpu().numpy()
                
                for idx, frame in zip(indices, processed):
                    idx_str = f"{idx:04d}"
                    out_path = os.path.join(video_fgr_dir, f"{idx_str}.png")
                    Image.fromarray(frame.astype(np.uint8)).save(out_path)
                    # Debug print to verify saving
                    print(f"Saved FGR frame to {out_path}")
            
            # GPU processing for PHA frames
            if pha_frames:
                indices = [x[0] for x in pha_frames]
                frames = np.stack([x[1] for x in pha_frames])
                
                # Convert to tensor and move to GPU
                tensor = torch.from_numpy(frames).to(device)
                
                # Move back to CPU and save
                processed = tensor.cpu().numpy()
                
                for idx, frame in zip(indices, processed):
                    idx_str = f"{idx:04d}"
                    out_path = os.path.join(video_pha_dir, f"{idx_str}.png")
                    Image.fromarray(frame.astype(np.uint8)).save(out_path)
                    # Debug print to verify saving
                    print(f"Saved PHA frame to {out_path}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"Error in GPU processing for {video_id}: {str(e)}")
        return False

def process_video_cpu(video_id, fgr_path, pha_path, output_fgr_dir, output_pha_dir):
    """Process a video using CPU"""
    try:
        # Get total frames
        cap = cv2.VideoCapture(fgr_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Sample frame indices
        frame_ids = sample_frame_indices(total_frames, sampling_ratio)
        
        # Create output directories
        video_fgr_dir = os.path.join(output_fgr_dir, video_id)
        video_pha_dir = os.path.join(output_pha_dir, video_id)
        os.makedirs(video_fgr_dir, exist_ok=True)
        os.makedirs(video_pha_dir, exist_ok=True)
        
        # Extract FGR frames
        fgr_cap = cv2.VideoCapture(fgr_path)
        for idx in frame_ids:
            fgr_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = fgr_cap.read()
            if ret:
                # Convert to RGB and save
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                idx_str = f"{idx:04d}"
                
                # Save as PNG
                output_path = os.path.join(video_fgr_dir, f"{idx_str}.png")
                Image.fromarray(frame_rgb).save(output_path)
                
                # Debug print
                print(f"Saved FGR frame to {output_path}")
        fgr_cap.release()
        
        # Extract PHA frames
        pha_cap = cv2.VideoCapture(pha_path)
        for idx in frame_ids:
            pha_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = pha_cap.read()
            if ret:
                # Convert to RGB and save
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                idx_str = f"{idx:04d}"
                
                # Save as PNG
                output_path = os.path.join(video_pha_dir, f"{idx_str}.png")
                Image.fromarray(frame_rgb).save(output_path)
                
                # Debug print
                print(f"Saved PHA frame to {output_path}")
        pha_cap.release()
        
        return True
    except Exception as e:
        print(f"Error in CPU processing for {video_id}: {str(e)}")
        return False

def process_video(args):
    """Process a single video (can be called by ProcessPoolExecutor)"""
    video_id, fgr_path, pha_path, output_fgr_dir, output_pha_dir, use_gpu = args
    
    try:
        # Print debug info about paths
        print(f"Processing video {video_id}")
        print(f"  FGR path: {fgr_path}")
        print(f"  PHA path: {pha_path}")
        print(f"  Output FGR dir: {os.path.join(output_fgr_dir, video_id)}")
        print(f"  Output PHA dir: {os.path.join(output_pha_dir, video_id)}")
        
        # Verify video files exist
        if not os.path.exists(fgr_path):
            return f"Error: FGR video file not found: {fgr_path}"
        if not os.path.exists(pha_path):
            return f"Error: PHA video file not found: {pha_path}"
        
        if use_gpu and use_cuda:
            success = process_video_gpu(video_id, fgr_path, pha_path, output_fgr_dir, output_pha_dir)
        else:
            success = process_video_cpu(video_id, fgr_path, pha_path, output_fgr_dir, output_pha_dir)
            
        if success:
            return f"Successfully processed {video_id}"
        else:
            return f"Error processing {video_id}"
    except Exception as e:
        return f"Error processing {video_id}: {str(e)}"

def process_dataset(dataset_type, input_dir, output_dir, force_gpu=False, force_cpu=False):
    """Process an entire dataset (train or test)"""
    print(f"Processing {dataset_type} dataset...")
    
    # Determine whether to use GPU
    use_gpu = use_cuda and not force_cpu
    if force_gpu and not use_cuda:
        print("Warning: GPU requested but CUDA is not available")
    
    # Setup paths
    input_fgr_dir = os.path.join(input_dir, dataset_type, "fgr")
    input_pha_dir = os.path.join(input_dir, dataset_type, "pha")
    output_fgr_dir = os.path.join(output_dir, dataset_type, "fgr")
    output_pha_dir = os.path.join(output_dir, dataset_type, "alpha")
    
    # Ensure directories exist
    os.makedirs(output_fgr_dir, exist_ok=True)
    os.makedirs(output_pha_dir, exist_ok=True)
    
    # Get video files
    video_files = [f for f in os.listdir(input_fgr_dir) if f.endswith("_blurred_rgb.mp4")]
    
    if not video_files:
        print(f"No video files found in {input_fgr_dir}")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    # Prepare arguments for parallel processing
    process_args = []
    for video_file in video_files:
        video_id = video_file.replace("_blurred_rgb.mp4", "")
        fgr_path = os.path.join(input_fgr_dir, video_file)
        pha_path = os.path.join(input_pha_dir, f"{video_id}_blurred_alpha.mp4")
        
        # Skip if alpha file doesn't exist
        if not os.path.exists(pha_path):
            print(f"Warning: Alpha file not found for {video_id}, skipping")
            continue
        
        process_args.append((video_id, fgr_path, pha_path, output_fgr_dir, output_pha_dir, use_gpu))
    
    # When using GPU with multiple processes, we need to be careful
    if use_gpu:
        # Process videos sequentially instead of with ProcessPoolExecutor
        for args in tqdm(process_args, desc=f"Extracting frames from {dataset_type}"):
            result = process_video(args)
            print(result)
    else:
        # CPU mode can use ProcessPoolExecutor safely
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_video, process_args),
                total=len(process_args),
                desc=f"Extracting frames from {dataset_type}"
            ))
        
        # Print results
        for result in results:
            print(result)

def main():
    args = parse_args()
    
    # Print configuration
    print("Configuration:")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"CUDA available: {use_cuda}")
    
    if args.gpu:
        print("Forcing GPU usage")
    if args.cpu:
        print("Forcing CPU usage")
    
    # Check if output directory exists and create if not
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which datasets to process
    datasets = []
    if args.dataset == 'both' or args.dataset == 'train':
        datasets.append('train')
    if args.dataset == 'both' or args.dataset == 'test':
        datasets.append('test')
    
    # Process each dataset
    for dataset in datasets:
        # Verify dataset directory exists
        dataset_dir = os.path.join(args.input_dir, dataset)
        if not os.path.exists(dataset_dir):
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            continue
            
        # Check if fgr and pha subdirectories exist
        fgr_dir = os.path.join(dataset_dir, "fgr")
        pha_dir = os.path.join(dataset_dir, "pha")
        
        if not os.path.exists(fgr_dir):
            print(f"Warning: FGR directory not found: {fgr_dir}")
            continue
            
        if not os.path.exists(pha_dir):
            print(f"Warning: PHA directory not found: {pha_dir}")
            continue
            
        # Process the dataset
        process_dataset(dataset, args.input_dir, args.output_dir, args.gpu, args.cpu)
    
    print("All processing complete!")

if __name__ == "__main__":
    # Ensure multiprocessing.set_start_method('spawn') is called
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Method already set
        pass
    
    main()