#!/usr/bin/env python3
"""
extract_motion_clips.py

Extracts locally consistent ego-motion clips from videos using VGGT.
Implements a sliding window approach with motion statistics filtering.

Design Rationale:
1.  **Sliding Window**: Instead of segmenting by semantic actions, we slide a fixed-length window
    (default 121 frames) to capture "regimes" of motion. This allows for overlapping clips
    that represent valid control sequences (e.g., "continue turning").
2.  **Motion Statistics**: We compute yaw rate, variance, and forward speed from VGGT inference.
3.  **Filtering**:
    *   **Consistency**: Reject if yaw direction flips significantly within the window.
    *   **Stability**: Reject if yaw rate variance is too high (jittery).
    *   **Signal**: Reject if the clip is effectively static.
4.  **Categorization**: Assigns a motion category (FORWARD, SLIGHT ARC, TURNING) for dataset balancing.

Usage:
    python extract_motion_clips.py --input_folders datasets/raw_videos/Indoor datasets/raw_videos/Outdoor --output_folder extracted_outputs
"""

import os
import sys
import argparse
import json
import shutil
import subprocess
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import torch

# Import VGGT and helpers from infer_move.py
# Ensure the current directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Ensure the project root is in sys.path for vggt package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_move import (
    ensure_device_and_dtype,
    sorted_image_list,
    get_motion_trajectory,
    VGGT
)

# Also need video_to_frames logic, importing from video_to_frames.py if available
# or implementing a simple version here.
try:
    from video_to_frames import video_to_frames
except ImportError:
    # Simple fallback if not found
    def video_to_frames(video_path, fps, output_dir):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        
        video_name = Path(video_path).stem
        save_dir = Path(output_dir) / video_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # We want to extract ALL frames for VGGT analysis, 
        # but the user might want specific FPS. 
        # For motion analysis, higher FPS is better for smoothness, 
        # but VGGT is robust. Let's assume we extract at the video's FPS 
        # or a target FPS.
        
        # If fps is None, keep original.
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Save frame
            cv2.imwrite(str(save_dir / f"frame_{count:06d}.jpg"), frame)
            count += 1
        cap.release()
def longest_consecutive_run(mask):
    """
    mask: boolean array
    returns: length of longest consecutive True run
    """
    max_run = run = 0
    for v in mask:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run

def classify_step(yaw, forward, args):
    print(f"Classify step: yaw={yaw:.2f}, forward={forward:.3f}")
    if abs(yaw) > args.turn_yaw_thresh:
        return "TL" if yaw > 0 else "TR"
    elif forward > args.forward_thresh:
        return "F"
    else:
        return "S"

def run_length_encode(seq):
    runs = []
    current = seq[0]
    length = 1
    for s in seq[1:]:
        if s == current:
            length += 1
        else:
            runs.append((current, length))
            current = s
            length = 1
    runs.append((current, length))
    return runs

def get_clean_motion_sequence(labels, min_len=5):
    """
    Converts per-frame labels into a clean sequence.
    1. Run Length Encode (group neighbors).
    2. Denoise (remove short 'glitches').
    3. Re-Merge (combine identical neighbors created by step 2).
    """
    # --- 1. Initial Run Length Encode ---
    runs = [] 
    if len(labels) == 0:
        return []
        
    current_label = labels[0]
    current_len = 1
    
    for l in labels[1:]:
        if l == current_label:
            current_len += 1
        else:
            runs.append([current_label, current_len])
            current_label = l
            current_len = 1
    runs.append([current_label, current_len])
    
    # --- 2. Denoise Loop ---
    # Repeatedly merge short segments until stable
    changed = True
    while changed:
        changed = False
        new_runs = []
        i = 0
        while i < len(runs):
            label, length = runs[i]
            
            # If short glitch (and not the only segment)
            if length < min_len and len(runs) > 1:
                changed = True
                # Try to merge into PREVIOUS
                if len(new_runs) > 0:
                    new_runs[-1][1] += length
                # Else merge into NEXT
                elif i + 1 < len(runs):
                    runs[i+1][1] += length
                else:
                    new_runs.append([label, length]) # Orphan, keep it
            else:
                new_runs.append([label, length])
            i += 1
        runs = new_runs

    # --- 3. FINAL MERGE (The Fix for 'MOVE_FORWARD_MOVE_FORWARD') ---
    # Denoising might have left [('F', 50), ('F', 60)]. Merge them now.
    final_runs = []
    if not runs:
        return []
        
    curr_l, curr_len = runs[0]
    for next_l, next_len in runs[1:]:
        if next_l == curr_l:
            curr_len += next_len
        else:
            final_runs.append([curr_l, curr_len])
            curr_l = next_l
            curr_len = next_len
    final_runs.append([curr_l, curr_len])

    return final_runs

def analyze_window(window_stats, args):
    yaws = np.array([s['yaw_deg'] for s in window_stats])
    forwards = np.array([s['forward_m'] for s in window_stats])
    T = len(yaws)

    # ---- Step 1: JITTER FILTER ----
    # Quick panning L/R increases the standard deviation of the yaw.
    # If the variance is too high, reject the clip immediately.
    std_yaw = np.std(yaws)
    if std_yaw > args.sigma_max:
        return False, [], {"reason": f"unstable_jitter_std_{std_yaw:.2f}"}

    # ---- Step 2: Elemental labels ----
    raw_labels = []
    for yaw, fwd in zip(yaws, forwards):
        if abs(yaw) > args.turn_yaw_thresh:
            raw_labels.append("TL" if yaw > 0 else "TR")
        elif fwd > args.forward_thresh:
            raw_labels.append("F")
        else:
            raw_labels.append("S")
    raw_labels = np.array(raw_labels)

    # ---- Step 3: Get Clean Sequence ----
    # This now handles the merging of split segments automatically
    clean_runs = get_clean_motion_sequence(raw_labels, min_len=args.min_consecutive_frames)
    
    # Extract only MOTION segments (ignore Static 'S')
    motion_segments = [r for r in clean_runs if r[0] != 'S']
    
    # ---- Step 4: Filter Logic ----
    
    # REJECT if empty
    if len(motion_segments) == 0:
        return False, [], {"reason": "static"}

    # REJECT if > 2 primitives
    if len(motion_segments) > 2:
        seg_str = "->".join([x[0] for x in motion_segments])
        return False, [], {"reason": f"too_complex_{seg_str}"}

    # REJECT if 2 primitives AND neither is 'F' (Forward)
    # This rejects "TL -> TR" or "TR -> TL"
    if len(motion_segments) == 2:
        types = set(x[0] for x in motion_segments)
        if 'F' not in types:
             seg_str = "->".join([x[0] for x in motion_segments])
             return False, [], {"reason": f"no_forward_in_compound_{seg_str}"}

    # ---- Step 5: Construct Output Primitives ----
    primitives = []
    for label, length in motion_segments:
        if label == "TL":
            primitives.append("TURN_LEFT")
        elif label == "TR":
            primitives.append("TURN_RIGHT")
        elif label == "F":
            primitives.append("MOVE_FORWARD")

    # If the motion sequence is something like F -> S -> F, the primitives become
    # ["MOVE_FORWARD", "MOVE_FORWARD"]. Treat that as a single MOVE_FORWARD for
    # naming/categorization purposes.
    if len(primitives) == 2 and primitives[0] == primitives[1]:
        primitives = [primitives[0]]

    # ---- Step 6: Metrics & Centers ----
    # Re-calculate centers based on the clean runs
    current_idx = 0
    center_map = {} 
    
    for label, length in clean_runs: # iterate over full list including S
        if label != 'S':
            seg_center = current_idx + (length / 2.0)
            
            # Since we now strictly enforce merged neighbors, 
            # we might have "F -> S -> F". 
            # If so, we pick the LONGER segment as the 'main' center for that action type,
            # or just overwrite (since we only keep max 2 primitives, and F-S-F would be 2 Fs? 
            # Wait: F-S-F counts as TWO motion segments 'F' and 'F'.
            # If F-S-F happens, that is 2 segments. If they are both F, 
            # we just merged them in get_clean_motion_sequence? 
            # NO: 'S' separates them.
            # So F-S-F is 2 segments. That triggers "len > 2" check? No, len=2.
            # But the primitives would be ["MOVE_FORWARD", "MOVE_FORWARD"].
            # We should probably reject F-S-F as "stuttering" or allow it.
            # If you want to merge F-S-F into one F, you need to increase min_len or change logic.
            # For now, F-S-F will be rejected by the "Merged duplicate" check? 
            # Actually, F-S-F results in primitives ["MOVE_FORWARD", "MOVE_FORWARD"].
            # That looks weird. Let's add a quick fix for that below.

            key = f"center_{label}"
            # If we have multiple of the same type (rare now), keep the one with longer length
            if key not in center_map:
                center_map[key] = (seg_center, length)
            else:
                if length > center_map[key][1]:
                    center_map[key] = (seg_center, length)
            
        current_idx += length

    # Unpack centers
    final_centers = {
        "center_TL": center_map["center_TL"][0] if "center_TL" in center_map else None,
        "center_TR": center_map["center_TR"][0] if "center_TR" in center_map else None,
        "center_F":  center_map["center_F"][0]  if "center_F"  in center_map else None,
    }

    ratio_TL = np.sum(raw_labels == "TL") / T
    ratio_TR = np.sum(raw_labels == "TR") / T
    ratio_F  = np.sum(raw_labels == "F")  / T

    metrics = {
        "ratio_TL": float(ratio_TL),
        "ratio_TR": float(ratio_TR),
        "ratio_F": float(ratio_F),
        "center_TL": final_centers["center_TL"],
        "center_TR": final_centers["center_TR"],
        "center_F":  final_centers["center_F"],
        "total_yaw": float(np.sum(yaws)),
        "mean_forward": float(np.mean(forwards)),
        "std_yaw": float(np.std(yaws)),
        "labels": raw_labels.tolist(),
        "topology": [x[0] for x in motion_segments]
    }

    # Keep topology consistent with the collapsed primitives logic above.
    if len(metrics["topology"]) == 2 and metrics["topology"][0] == metrics["topology"][1]:
        metrics["topology"] = [metrics["topology"][0]]

    return True, primitives, metrics


def extract_clip(video_path, start_frame, end_frame, output_path, step=1, output_fps=None):
    """
    Extract a clip from the source video using OpenCV.
    supports subsampling by `step`.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if output_fps is None:
        output_fps = fps / step
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Change extension to .mp4
    output_path = Path(str(output_path)).with_suffix('.mp4')
    
    # Use mp4v codec which is widely supported
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    
    if not out.isOpened():
        print(f"Warning: Could not open video writer for {output_path}")
        cap.release()
        return False
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_written = 0
    # We need to iterate through the range in the original video
    # and write only every step-th frame.
    for i in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            out.write(frame)
            frames_written += 1
        
    cap.release()
    out.release()
    
    if frames_written == 0:
        print(f"Warning: No frames written for {output_path}")
        if output_path.exists():
            output_path.unlink()
        return False

    # Re-encode using ffmpeg to ensure VS Code compatibility (h264, yuv420p)
    # The user provided command: ffmpeg -i "$1" -c:v libx264 -pix_fmt yuv420p -an "${1%.mp4}_tmp.mp4"
    if shutil.which("ffmpeg"):
        temp_path = output_path.with_name(f"{output_path.stem}_temp.mp4")
        try:
            # Move the opencv output to a temp file
            os.rename(str(output_path), str(temp_path))
            
            cmd = [
                "ffmpeg", "-y", "-i", str(temp_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                str(output_path)
            ]
            # Run silently
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Remove temp if successful
            if temp_path.exists():
                temp_path.unlink()
                
        except Exception as e:
            print(f"Warning: ffmpeg post-processing failed for {output_path}: {e}")
            # Try to restore original if ffmpeg failed
            if temp_path.exists() and not output_path.exists():
                os.rename(str(temp_path), str(output_path))
            elif temp_path.exists():
                temp_path.unlink()
    else:
        print("Warning: ffmpeg not found. Output videos may not play in some players.")

    # Verify the saved video
    verify_cap = cv2.VideoCapture(str(output_path))
    if not verify_cap.isOpened():
        print(f"Warning: Failed to open saved video for verification: {output_path}")
        return False
        
    actual_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    verify_fps = verify_cap.get(cv2.CAP_PROP_FPS)
    if verify_fps > 0:
        duration = actual_frames / verify_fps
    else:
        duration = 0
        
    verify_cap.release()
    
    if actual_frames == 0 or duration < 3.0: # Minimum 4 seconds
        print(f"Warning: Invalid video content (duration={duration:.2f}s, frames={actual_frames}) for {output_path}")
        if output_path.exists():
            output_path.unlink()
        return False

    return True

def _load_existing_metadata(output_folder):
    sum_metadata_path = output_folder / "sum_metadata.json"
    sum_rejected_path = output_folder / "sum_rejected.json"

    existing_clips = []
    existing_rejected = []
    processed_sources = set()

    clip_root_candidates = set()
    try:
        for p in output_folder.iterdir():
            if p.is_dir():
                clip_root_candidates.add(p.name)
    except Exception:
        pass

    if sum_metadata_path.exists():
        try:
            with open(sum_metadata_path, 'r') as f:
                existing_clips = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {sum_metadata_path}: {e}")

    if sum_rejected_path.exists():
        try:
            with open(sum_rejected_path, 'r') as f:
                existing_rejected = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {sum_rejected_path}: {e}")

    def _add_source(entry):
        source_relpath = entry.get('source_relpath')
        if source_relpath:
            processed_sources.add(source_relpath)
            return

        rel_path = entry.get('relative_path')
        video = entry.get('video')
        if rel_path and video:
            parts = Path(rel_path).parts
            if len(parts) >= 2:
                if parts[0] in clip_root_candidates and len(parts) >= 3:
                    input_root = parts[1]
                    rel_parent = Path(*parts[2:-1])
                else:
                    input_root = parts[0]
                    rel_parent = Path(*parts[1:-1])

                for ext in ('.mp4', '.mov'):
                    processed_sources.add(str(Path(input_root) / rel_parent / f"{video}{ext}"))

    for entry in existing_clips:
        _add_source(entry)
    for entry in existing_rejected:
        _add_source(entry)

    return existing_clips, existing_rejected, processed_sources

def _is_already_processed(source_relpath, processed_sources):
    source_relpath = Path(source_relpath)
    candidates = {source_relpath}

    if source_relpath.suffix.lower() in ('.mp4', '.mov'):
        candidates.add(source_relpath.with_suffix('.mp4' if source_relpath.suffix.lower() == '.mov' else '.mov'))

    parts = source_relpath.parts
    if len(parts) > 1:
        stripped = Path(*parts[1:])
        candidates.add(stripped)
        if stripped.suffix.lower() in ('.mp4', '.mov'):
            candidates.add(stripped.with_suffix('.mp4' if stripped.suffix.lower() == '.mov' else '.mov'))

    for c in candidates:
        if str(c) in processed_sources:
            return True
    return False

def process_video(video_path, model, device, dtype, args):
    print(f"Processing {video_path.name}...")
    
    # 1. Inspect Video FPS to determine downsampling
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Heuristic: If > 50 FPS, downsample by 2 (target ~30 or 25 fps)
    # The user wants to process as if 24fps.
    # 60fps / 2 = 30fps. 60 / 2.5 = 24.
    # We use integer steps for stability. Step=2 for high FPS.
    # Lower threshold to 45 to catch 48/50fps videos
    frame_step = 1
    if src_fps > 45:
        frame_step = 2
        print(f"  Detected High FPS ({src_fps:.2f}). Downsampling by {frame_step}x for analysis.")
    
    # 2. Extract frames to temp dir
    temp_frames_dir = Path("temp_frames_extraction")
    if temp_frames_dir.exists():
        shutil.rmtree(temp_frames_dir)
    temp_frames_dir.mkdir()
    
    video_name = video_path.stem
    video_frames_dir = temp_frames_dir / video_name
    video_frames_dir.mkdir()
    
    current_frame = 0 # Original index
    saved_count = 0   # Saved index
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame % frame_step == 0:
            cv2.imwrite(str(video_frames_dir / f"{saved_count:06d}.jpg"), frame)
            saved_count += 1
            
        current_frame += 1
    cap.release()
    
    # 2. Subsample images for analysis (improves SNR and reduces noise)
    # Note: imgs now refers to the ALREADY downsampled frames if frame_step > 1
    imgs = sorted_image_list(video_frames_dir)
    analysis_subsample_step = args.analysis_subsample_step
    imgs_subsampled = imgs[::analysis_subsample_step]
    
    steps_needed = (args.window_len // analysis_subsample_step)
    
    if len(imgs_subsampled) < steps_needed + 1:
        print(f"Video too short ({len(imgs_subsampled)} subsampled frames) for window {args.window_len}")
        shutil.rmtree(temp_frames_dir)
        return [], []

    print(f"Running VGGT on {len(imgs_subsampled)} training frames (analysis stride={analysis_subsample_step})...")
    # Pass the global batch size (computed at startup). If for some reason it's missing, fall back to batch_size_per_gpu.
    batch_sz = getattr(args, 'global_batch', args.batch_size_per_gpu)
    trajectory = get_motion_trajectory(model, device, dtype, imgs_subsampled, batch_size=batch_sz, verbose=args.verbose)
    # trajectory has length N-1 for N frames
    
    # 3. Sliding Window Analysis
    valid_clips = []
    rejected_windows = []
    
    # We need window_len frames in "extracted" space
    stride_in_subsampled = args.stride // analysis_subsample_step
    
    for start_idx in range(0, len(trajectory) - steps_needed + 1, stride_in_subsampled):
        end_idx = start_idx + steps_needed
        window_stats = trajectory[start_idx:end_idx]
        
        is_valid, primitives, metrics = analyze_window(window_stats, args)
        
        # Map back to frame indices in the temp folder (which might be downsampled from original)
        clip_start_in_temp = start_idx * analysis_subsample_step
        clip_end_in_temp = clip_start_in_temp + args.window_len
        
        # Map back to ORIGINAL video frame indices
        clip_start_orig = clip_start_in_temp * frame_step
        clip_end_orig = clip_end_in_temp * frame_step
        
        clip_info_base = {
            'video': video_name,
            'start_frame': clip_start_orig,
            'end_frame': clip_end_orig,
            'frame_step': frame_step,  # Store step used
            'metrics': metrics
        }
        
        if is_valid:
            clip_info = clip_info_base.copy()
            clip_info.update({
                'primitives': primitives,
                'category': '_'.join(primitives)
            })
            valid_clips.append(clip_info)
        else:
            rejection_reason = "REJECT_NO_MOTION: No valid primitives detected"
            rejected_info = clip_info_base.copy()
            rejected_info['rejection_reason'] = rejection_reason
            rejected_windows.append(rejected_info)
            if args.verbose:
                print(f"  Window [{clip_start_orig}-{clip_end_orig}]: {rejection_reason}")
    
    print(f"  Accepted: {len(valid_clips)} clips, Rejected: {len(rejected_windows)} windows")
    
    # Cleanup
    shutil.rmtree(temp_frames_dir)
    
    return valid_clips, rejected_windows


def main():
    parser = argparse.ArgumentParser(description="Extract motion clips using VGGT")
    parser.add_argument("--input_folders", nargs='+', required=True, help="Folders containing input videos (accepts multiple)")
    parser.add_argument("--output_folder", type=str, default="extracted_outputs", help="Root folder to save clips and metadata")
    parser.add_argument("--model_name", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Sliding Window Params
    parser.add_argument("--window_len", type=int, default=121, help="Window length in frames")
    parser.add_argument("--stride", type=int, default=24, help="Stride for sliding window (in analysis space)")
    parser.add_argument("--analysis_subsample_step", type=int, default=2, help="Subsample stride for motion analysis (2-3x recommended)")
    parser.add_argument("--batch_size_per_gpu", type=int, default=20, help="Batch size per GPU for pairwise inference (pairs per GPU)")
    parser.add_argument("--gpus", type=str, default="", help="Comma separated GPU ids to use (e.g. '0,1'). If empty, use --device as specified")
    
    # Filtering Params
    parser.add_argument("--sigma_max", type=float, default=1.5, help="Max allowed std dev of yaw rate")
    parser.add_argument("--theta_min", type=float, default=3.0, help="Min total yaw change for validity")
    parser.add_argument("--d_min", type=float, default=0.05, help="Min total forward distance for validity")
    
    # Primitive Detection Params
    parser.add_argument("--turn_yaw_thresh", type=float, default=2.0, help="Yaw threshold for turning regime detection (deg/frame)")
    parser.add_argument("--forward_thresh", type=float, default=0.01, help="Forward speed threshold for forward regime (m/frame)")
    parser.add_argument("--min_consecutive_frames", type=int, default=5, help="Min consecutive frames for a valid primitive")
    parser.add_argument("--min_ratio", type=float, default=0.15, help="Minimum ratio of frames for a motion type to be considered")
    parser.add_argument("--verbose", action="store_true", help="Print detailed rejection reasons")
    parser.add_argument("--save_rejected", action="store_true", help="Save rejected windows as video clips")
    
    args = parser.parse_args()
    
    # Validate stride is compatible with analysis_subsample_step
    if args.stride % args.analysis_subsample_step != 0:
        parser.error(f"--stride ({args.stride}) must be a multiple of --analysis_subsample_step ({args.analysis_subsample_step})")
    
    # Setup
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    existing_clips, existing_rejected, processed_sources = _load_existing_metadata(output_folder)
    if processed_sources:
        print(f"Found {len(processed_sources)} already processed source videos in metadata.")
    
    # Determine device string (if user requested multiple GPUs via --gpus, pick first as primary)
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',') if x.strip() != '']
        if len(gpu_ids) > 0:
            device_str = f"cuda:{gpu_ids[0]}"
            num_gpus = len(gpu_ids)
        else:
            device_str = args.device
            num_gpus = 1
    else:
        device_str = args.device
        num_gpus = 1

    device, dtype = ensure_device_and_dtype(device_str)
    print(f"Loading model {args.model_name} on {device}...")
    model = VGGT.from_pretrained(args.model_name).to(device)
    model.eval()

    # If multiple GPUs requested, wrap in DataParallel (simple split across batch dim)
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',') if x.strip() != '']
        if len(gpu_ids) > 1:
            print(f"Using DataParallel on GPUs: {gpu_ids}")
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # Compute global batch (per-gpu * num_gpus)
    num_gpus_used = max(1, num_gpus)
    args.num_gpus = num_gpus_used
    args.global_batch = args.batch_size_per_gpu * num_gpus_used
    print(f"Batch size per GPU: {args.batch_size_per_gpu}, num_gpus: {num_gpus_used}, global batch: {args.global_batch}")
    
    all_clips = []
    all_rejected = []
    
    # Per-directory storage for local metadata
    dir_clips_map = defaultdict(list)
    dir_rejected_map = defaultdict(list)

    for root_str in args.input_folders:
        input_root = Path(root_str).resolve()
        print(f"Processing input folder: {input_root}")
        
        # Recursively find videos using os.walk to support symlinks
        video_files = []
        for root, dirs, files in os.walk(input_root, followlinks=True):
            # Sort dirs and files to ensure deterministic order
            dirs.sort()
            files.sort()
            root_path = Path(root)
            for file in files:
                if file.lower().endswith(('.mp4', '.mov')):
                    video_files.append(root_path / file)
        
        print(f"  Found {len(video_files)} videos in {input_root.name}")
        
        for video_path in video_files:
            # Note: We do NOT resolve video_path here to preserve the logical structure 
            # if symlinks are involved.
            
            # Compute relative path to maintain structure
            # e.g., input_root=/data/Indoor, video_path=/data/Indoor/Scene1/vid.mp4
            # rel_path = Scene1/vid.mp4
            try:
                rel_path = video_path.relative_to(input_root)
            except ValueError:
                # Fallback if path handling is weird
                rel_path = Path(video_path.name)

            source_relpath = Path(input_root.name) / rel_path
            if _is_already_processed(source_relpath, processed_sources):
                print(f"Skipping already processed video: {source_relpath}")
                continue

            # Destination directory mapping:
            # For inputs named like 'raw_videos_24' we store clips under 'raw_video_clips_24'
            clip_root_name = input_root.name
            if 'raw_videos' in clip_root_name:
                clip_root_name = clip_root_name.replace('raw_videos', 'raw_video_clips')
            else:
                # fallback: append _clips to indicate these are extracted clips
                clip_root_name = clip_root_name + '_clips'

            # e.g., extracted_outputs/raw_video_clips_24/Scene1
            dest_dir = output_folder / clip_root_name / rel_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing {video_path.name} -> {dest_dir} (clip root: {clip_root_name})")
            clips, rejected = process_video(video_path, model, device, dtype, args)
            
            # Save accepted clips
            for i, clip in enumerate(clips):
                clip_name = f"{clip['video']}_{clip['start_frame']}_{clip['end_frame']}_{clip['category']}.mp4"
                clip_path = dest_dir / clip_name
                
                # Extract and save the actual video clip if not already present
                frame_step = clip.get('frame_step', 1)
                if not clip_path.exists():
                    ok = extract_clip(video_path, clip['start_frame'], clip['end_frame'], clip_path, step=frame_step)
                    if not ok:
                        print(f"Warning: failed to extract {clip_path}")
                else:
                    # Already exists; skip re-extraction
                    print(f"Skipping extraction, file already exists: {clip_path}")

                # Update clip metadata (relative_path should use clip_root_name)
                clip['filename'] = clip_name
                clip['relative_path'] = str(Path(clip_root_name) / rel_path.parent / clip_name)
                clip['source_relpath'] = str(source_relpath)

                all_clips.append(clip)
                dir_clips_map[dest_dir].append(clip)
            
            # Record/Save rejected windows
            # Always record metadata for local summary
            for i, rej in enumerate(rejected):
                frame_step = rej.get('frame_step', 1)

                rej['source_relpath'] = str(source_relpath)
                
                if args.save_rejected:
                    reason_short = rej['rejection_reason'].split(':')[0].replace(' ', '_')
                    rej_name = f"{rej['video']}_{rej['start_frame']}_{rej['end_frame']}_{reason_short}.mp4"

                    rejected_subfolder = dest_dir / "rejected"
                    rejected_subfolder.mkdir(exist_ok=True)
                    rej_path = rejected_subfolder / rej_name
                    
                    if not rej_path.exists():
                        ok = extract_clip(video_path, rej['start_frame'], rej['end_frame'], rej_path, step=frame_step)
                        if not ok:
                            print(f"Warning: failed to extract rejected clip {rej_path}")
                    else:
                        print(f"Skipping extraction of rejected clip; already exists: {rej_path}")

                    rej['filename'] = rej_name
                    rej['relative_path'] = str(Path(clip_root_name) / rel_path.parent / "rejected" / rej_name)
                else:
                    # Provide info even if not saving video
                    rej['filename'] = None
                    rej['relative_path'] = None
                
                # Store locally
                dir_rejected_map[dest_dir].append(rej)
            
            # Collect all rejected for global summary regardless of saving video
            all_rejected.extend(rejected)
    
    # Write local metadata (for each directory that has content)
    for dest_dir, clips in dir_clips_map.items():
        local_meta_path = dest_dir / "clips_metadata.json"
        existing_local = []
        if local_meta_path.exists():
            try:
                with open(local_meta_path, 'r') as f:
                    existing_local = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to read {local_meta_path}: {e}")

        with open(local_meta_path, 'w') as f:
            json.dump(existing_local + clips, f, indent=2)

    for dest_dir, rejs in dir_rejected_map.items():
        # Maybe put rejected metadata in the rejected subfolder if it exists?
        # Or in the main dir? Let's put in main dir as rejected_windows.json
        local_rej_path = dest_dir / "rejected_windows.json"
        existing_local = []
        if local_rej_path.exists():
            try:
                with open(local_rej_path, 'r') as f:
                    existing_local = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to read {local_rej_path}: {e}")

        with open(local_rej_path, 'w') as f:
            json.dump(existing_local + rejs, f, indent=2)
            
    # Write global summary metadata
    sum_metadata_path = output_folder / "sum_metadata.json"
    with open(sum_metadata_path, 'w') as f:
        json.dump(existing_clips + all_clips, f, indent=2)
    
    sum_rejected_path = output_folder / "sum_rejected.json"
    with open(sum_rejected_path, 'w') as f:
        json.dump(existing_rejected + all_rejected, f, indent=2)
        
    print(f"Done. Processed {len(all_clips)} clips across {len(args.input_folders)} input roots.")
    print(f"Global metadata saved to {sum_metadata_path}")

if __name__ == "__main__":
    main()
