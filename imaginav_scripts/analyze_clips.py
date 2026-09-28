#!/usr/bin/env python3
"""
analyze_clips.py

Analyze existing video clips using the same motion analysis logic from extract_motion_clips.py.
Useful for debugging rejected clips or analyzing individual sequences.

Usage:
    python analyze_clips.py --clips clip1.avi clip2.avi clip3.avi
    python analyze_clips.py --clips motion_clips/rejected/*.avi
"""

import os
import sys
import argparse
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
import torch

# Import VGGT and helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infer_move import (
    ensure_device_and_dtype,
    sorted_image_list,
    get_motion_trajectory,
    VGGT
)

# Import analysis functions from extract_motion_clips.py
from extract_motion_clips import analyze_window

def analyze_clip(clip_path, model, device, dtype, args):
    """
    Analyze a single video clip and return motion analysis results.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {clip_path.name}")
    print(f"{'='*60}")
    
    # 1. Extract frames from clip
    temp_frames_dir = Path("temp_analyze_frames")
    if temp_frames_dir.exists():
        shutil.rmtree(temp_frames_dir)
    temp_frames_dir.mkdir()
    
    clip_name = clip_path.stem
    frames_dir = temp_frames_dir / clip_name
    frames_dir.mkdir()
    
    # Extract all frames
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open {clip_path}")
        return None
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(frames_dir / f"{frame_count:06d}.jpg"), frame)
        frame_count += 1
    cap.release()
    
    print(f"Extracted {frame_count} frames")
    
    if frame_count < 2:
        print("ERROR: Not enough frames for analysis")
        shutil.rmtree(temp_frames_dir)
        return None
    
    # 2. Subsample images for analysis (if analysis_stride > 1)
    imgs = sorted_image_list(frames_dir)
    if hasattr(args, 'analysis_stride') and args.analysis_stride > 1:
        imgs_subsampled = imgs[::args.analysis_stride]
        print(f"Subsampling images: {len(imgs)} -> {len(imgs_subsampled)} (stride={args.analysis_stride})")
        imgs = imgs_subsampled
    
    print(f"Running VGGT on {len(imgs)} frames...")
    trajectory = get_motion_trajectory(model, device, dtype, imgs)
    
    # 3. Apply analysis logic using the same function from extract_motion_clips.py
    is_valid, primitives, metrics = analyze_window(trajectory, args)
    
    # Generate rejection reason if invalid
    rejection_reason = None
    if not is_valid:
        if len(primitives) == 0:
            rejection_reason = "REJECT_NO_PRIMITIVES: No valid motion primitives detected"
        else:
            rejection_reason = "REJECT_UNKNOWN: Failed validation check"
    
    # Extract detailed info for printing
    yaws = np.array([s['yaw_deg'] for s in trajectory])
    forwards = np.array([s['forward_m'] for s in trajectory])
    T = len(yaws)
    
    print(f"\nBasic Metrics:")
    if 'mean_yaw' in metrics:
        print(f"  Mean yaw: {metrics.get('mean_yaw', np.mean(yaws)):.3f} deg/frame")
    print(f"  Std yaw: {metrics['std_yaw']:.3f} deg/frame")
    print(f"  Total yaw: {metrics['total_yaw']:.3f} deg")
    print(f"  Mean forward: {metrics['mean_forward']:.4f} m/frame")
    print(f"  Total forward: {float(np.sum(forwards)):.4f} m")
    
    # Motion detection details from metrics
    if 'labels' in metrics:
        labels = np.array(metrics['labels'])
        tl_count = np.sum(labels == "TL")
        tr_count = np.sum(labels == "TR")
        f_count = np.sum(labels == "F")
        s_count = np.sum(labels == "S")
        
        print(f"\nMotion Detection:")
        print(f"  TURN_LEFT frames: {tl_count}/{T} (ratio: {metrics['ratio_TL']:.3f})")
        print(f"  TURN_RIGHT frames: {tr_count}/{T} (ratio: {metrics['ratio_TR']:.3f})")
        print(f"  MOVE_FORWARD frames: {f_count}/{T} (ratio: {metrics['ratio_F']:.3f})")
        print(f"  STATIC frames: {s_count}/{T}")
        
        print(f"\nTemporal Centers:")
        if metrics.get('center_TL') is not None:
            print(f"  TURN_LEFT center: frame {metrics['center_TL']:.1f}")
        if metrics.get('center_TR') is not None:
            print(f"  TURN_RIGHT center: frame {metrics['center_TR']:.1f}")
        if metrics.get('center_F') is not None:
            print(f"  MOVE_FORWARD center: frame {metrics['center_F']:.1f}")
    
    print(f"\nPrimitives Detected: {primitives if primitives else 'None'}")
    
    if not is_valid:
        print(f"\nRejection Reason:")
        print(f"  - {rejection_reason}")
    
    result = {
        'clip': clip_path.name,
        'is_valid': is_valid,
        'primitives': primitives,
        'rejection_reason': rejection_reason if not is_valid else None,
        'metrics': metrics,
    }
    
    # Cleanup
    shutil.rmtree(temp_frames_dir)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Analyze video clips using VGGT motion analysis")
    parser.add_argument("--clips", nargs="+", required=True, help="Path(s) to video clip(s) to analyze")
    parser.add_argument("--model_name", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, help="Optional output JSON file for results")
    parser.add_argument("--analysis_stride", type=int, default=6, help="Subsample stride for motion analysis (2-3x recommended)")
    
    # Analysis Params (same as extract_motion_clips.py)
    parser.add_argument("--turn_yaw_thresh", type=float, default=2.0, help="Yaw threshold for turning regime detection (deg/frame)")
    parser.add_argument("--forward_thresh", type=float, default=0.01, help="Forward speed threshold for forward regime (m/frame)")
    parser.add_argument("--min_consecutive_frames", type=int, default=5, help="Min consecutive frames for a valid primitive")
    parser.add_argument("--min_ratio", type=float, default=0.15, help="Minimum ratio of frames for a motion type to be considered")
    
    args = parser.parse_args()
    
    # Setup
    device, dtype = ensure_device_and_dtype(args.device)
    print(f"Loading model {args.model_name} on {device}...")
    model = VGGT.from_pretrained(args.model_name).to(device)
    model.eval()
    
    # Process each clip
    results = []
    for clip_path_str in args.clips:
        clip_path = Path(clip_path_str)
        if not clip_path.exists():
            print(f"WARNING: {clip_path} does not exist, skipping...")
            continue
        
        result = analyze_clip(clip_path, model, device, dtype, args)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total clips analyzed: {len(results)}")
    valid_count = sum(1 for r in results if r['is_valid'])
    print(f"Valid clips: {valid_count}")
    print(f"Invalid clips: {len(results) - valid_count}")
    
    # Count primitives
    if valid_count > 0:
        print(f"\nPrimitive Distribution:")
        primitive_counts = {}
        for r in results:
            if r['is_valid']:
                key = '_'.join(r['primitives'])
                primitive_counts[key] = primitive_counts.get(key, 0) + 1
        
        for prim, count in sorted(primitive_counts.items(), key=lambda x: -x[1]):
            print(f"  {prim}: {count}")
    
    # Count rejection reasons
    if len(results) - valid_count > 0:
        print(f"\nRejection Reason Distribution:")
        reason_counts = {}
        for r in results:
            if not r['is_valid'] and r['rejection_reason']:
                reason_type = r['rejection_reason'].split(':')[0]
                reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
        
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
