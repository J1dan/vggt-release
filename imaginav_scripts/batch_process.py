#!/usr/bin/env python3
"""
batch_process.py

Batch process videos in a folder through the full pipeline:
1. Extract frames from video
2. Run demo_new.py for motion analysis
3. Convert visualization frames back to video

Usage:
    python batch_process.py --input_folder inference_outputs/vln_n1 --fps 3 --device cpu
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully\n")

def process_video(video_path: Path, fps: float, device: str, subsample: int, yaw_thresh: float, forward_thresh: float):
    """Process a single video through the pipeline."""
    video_name = video_path.stem
    print(f"\n{'#'*60}")
    print(f"Processing video: {video_name}")
    print(f"{'#'*60}")
    
    # Step 1: Extract frames
    frames_dir = Path("frames")
    run_command(
        ["python", "video_to_frames.py", 
         "--video_path", str(video_path),
         "--fps", str(fps),
         "--output_dir", str(frames_dir)],
        f"Step 1: Extracting frames from {video_name}"
    )
    
    # Step 2: Run demo_new.py
    video_frames_dir = frames_dir / video_name
    run_command(
        ["python", "demo_new.py",
         "--frames_dir", str(video_frames_dir),
         "--device", device,
         "--subsample", str(subsample),
         "--yaw_thresh_deg", str(yaw_thresh),
         "--forward_thresh_m", str(forward_thresh),
         "--verbose"],
        f"Step 2: Running motion analysis on {video_name}"
    )
    
    # Step 3: Convert visualization frames to video
    viz_folder = video_frames_dir.parent / (video_name + "_viz")
    run_command(
        ["python", "frames_to_video.py",
         "--folder", f"{video_name}_viz",
         "--fps", str(fps)],
        f"Step 3: Creating output video for {video_name}"
    )
    
    # Copy output video to original location if desired
    output_video = viz_folder / "video.mp4"
    if output_video.exists():
        final_output = video_path.parent / f"{video_name}_output.mp4"
        import shutil
        shutil.copy(str(output_video), str(final_output))
        print(f"\n✓ Final video saved to: {final_output}")
    
    print(f"\n{'#'*60}")
    print(f"Completed processing: {video_name}")
    print(f"{'#'*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Batch process videos through the full pipeline.")
    parser.add_argument("--input_folder", type=str, required=True, 
                        help="Folder containing input videos (e.g., inference_outputs/vln_n1)")
    parser.add_argument("--fps", type=float, default=3.0, 
                        help="Frames per second for extraction and output video")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Subsample every Nth frame in demo_new")
    parser.add_argument("--yaw_thresh_deg", type=float, default=3.0,
                        help="Yaw threshold in degrees")
    parser.add_argument("--forward_thresh_m", type=float, default=0.02,
                        help="Forward movement threshold in meters")
    parser.add_argument("--pattern", type=str, default="*.mp4",
                        help="File pattern to match (default: *.mp4)")
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    
    if not input_folder.exists():
        print(f"ERROR: Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    # Find all videos matching the pattern
    video_files = list(input_folder.glob(args.pattern))
    
    if not video_files:
        print(f"ERROR: No videos found matching pattern '{args.pattern}' in {input_folder}")
        sys.exit(1)
    
    print(f"\nFound {len(video_files)} video(s) to process:")
    for video in video_files:
        print(f"  - {video.name}")
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        print(f"\n\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}")
        print(f"{'='*60}")
        
        try:
            process_video(video_path, args.fps, args.device, args.subsample, 
                         args.yaw_thresh_deg, args.forward_thresh_m)
        except Exception as e:
            print(f"\nERROR processing {video_path.name}: {e}")
            print("Continuing with next video...\n")
            continue
    
    print(f"\n\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Processed {len(video_files)} video(s)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
