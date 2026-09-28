#!/usr/bin/env python3
"""
demo_new.py

Run VGGT on consecutive frames under frames/<seq> and save visualizations to frames/<seq>_viz.

Usage:
    python demo_new.py --frames_dir ./frames --device cuda --subsample 2
"""

import os
import math
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import sys

# Ensure the project root is in sys.path for vggt package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- VGGT imports (assumes vggt is installed / in PYTHONPATH) ---
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# --- Helpers ---

def ensure_device_and_dtype(device_str):
    device = torch.device(device_str)
    # choose dtype same as repo example
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device.index)
        dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    return device, dtype

def sorted_image_list(folder: Path, exts=(".png", ".jpg", ".jpeg")):
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    return sorted(imgs, key=lambda p: int(''.join(filter(str.isdigit, p.stem))))

def load_single_image_to_tensor(path: Path, device, dtype):
    # load_and_preprocess_images expects a list of paths (strings)
    imgs = load_and_preprocess_images([str(path)])
    imgs = imgs.to(device)
    if dtype is not None and imgs.dtype != dtype:
        imgs = imgs.to(dtype)
    return imgs  # shape (C,H,W) or depending on repo; example expects batch later

def stack_side_by_side(img1: Image.Image, img2: Image.Image):
    # convert to same size
    w = max(img1.width, img2.width)
    h = max(img1.height, img2.height)
    new = Image.new("RGB", (w * 2, h))
    img1_resized = img1.resize((w, h))
    img2_resized = img2.resize((w, h))
    new.paste(img1_resized, (0, 0))
    new.paste(img2_resized, (w, 0))
    return new

def draw_overlay(img: Image.Image, action_text: str, yaw_deg: float, forward_m: float):
    draw = ImageDraw.Draw(img)
    # choose a simple font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    # top-left text
    text = f"Action: {action_text}\nYaw (deg): {yaw_deg:.2f}\nForward (m): {forward_m:.3f}"
    draw.rectangle(((5,5),(320,85)), fill=(0,0,0,160))
    draw.multiline_text((10,10), text, fill=(255,255,255), font=font)

    return img

def compute_action_from_poses(extrinsic1, extrinsic2, yaw_thresh_deg=5.0, forward_thresh_m=0.03):
    """
    extrinsic matrices expected shape (4,4) numpy arrays (camera_from_world)
    Returns (yaw_deg, forward_m, action_str)
    """
    # Ensure inputs are on CPU and numpy if they are tensors
    if isinstance(extrinsic1, torch.Tensor):
        extrinsic1 = extrinsic1.detach().cpu().numpy()
    if isinstance(extrinsic2, torch.Tensor):
        extrinsic2 = extrinsic2.detach().cpu().numpy()

    # Convert 3x4 extrinsics to 4x4 if needed
    if extrinsic1.shape == (3, 4):
        T1 = np.eye(4)
        T1[:3] = extrinsic1
    else:
        T1 = extrinsic1
    
    if extrinsic2.shape == (3, 4):
        T2 = np.eye(4)
        T2[:3] = extrinsic2
    else:
        T2 = extrinsic2
    
    # convert to numpy if tensors
    if isinstance(T1, torch.Tensor):
        T1 = T1.cpu().numpy()
    if isinstance(T2, torch.Tensor):
        T2 = T2.cpu().numpy()

    # relative transform mapping coords in cam1 to cam2
    T1_inv = np.linalg.inv(T1)
    T_rel = T2 @ T1_inv

    R = T_rel[:3,:3]
    t = T_rel[:3,3]

    # yaw extraction: atan2(R[0,2], R[2,2]) - follows the derivation for OpenCV camera coords
    yaw_rad = math.atan2(R[0,2], R[2,2])
    yaw_deg = math.degrees(yaw_rad)

    # forward distance: -t[2] as explained in notes above
    forward_m = float(-t[2])

    # determine discrete action
    action = "none"
    if yaw_deg > yaw_thresh_deg:
        action = "rotate_left"
    elif yaw_deg < -yaw_thresh_deg:
        action = "rotate_right"
    elif forward_m > forward_thresh_m:
        action = "forward"
    else:
        action = "none"

    return yaw_deg, forward_m, action

def get_motion_trajectory(model, device, dtype, imgs, batch_size=32, verbose=False):
    """
    Compute motion trajectory (yaw, forward) for a list of image paths using batched pairwise inference.

    Args:
        model: VGGT model
        device: torch.device (first device for the model)
        dtype: autocast dtype (torch.float16 / torch.bfloat16 / None)
        imgs: list of image paths
        batch_size: global batch size (total pairs per forward pass). If using multiple GPUs via
                    DataParallel, set --batch_size_per_gpu and `global_batch = per_gpu * num_gpus`.
        verbose: print informative messages

    Returns:
        list of dicts: [{'yaw_deg': float, 'forward_m': float}, ...]
    """
    if len(imgs) < 2:
        return []

    # Preload and preprocess all images once (on CPU)
    imgs_tensor = load_and_preprocess_images([str(p) for p in imgs])
    # imgs_tensor shape: (N, C, H, W)
    N = imgs_tensor.shape[0]
    num_pairs = N - 1

    trajectory = []

    if verbose:
        print(f"Running batched inference for {num_pairs} pairs (batch_size={batch_size})")

    # iterate over batches of pair indices
    start = 0
    cur_batch = batch_size
    while start < num_pairs:
        cur_batch = min(batch_size, num_pairs - start)

        # build batch of pairs shape (B, 2, C, H, W)
        batch_pairs = torch.stack([
            torch.stack((imgs_tensor[i], imgs_tensor[i+1]), dim=0) for i in range(start, start + cur_batch)
        ], dim=0)

        # Move to device & correct dtype
        batch_pairs = batch_pairs.to(device, non_blocking=True)
        if dtype is not None and batch_pairs.dtype != dtype:
            batch_pairs = batch_pairs.to(dtype)

        # Forward pass (handle OOM by reducing batch size)
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    images_scene = batch_pairs  # shape (B, 2, C, H, W)
                    # Call the model's forward so that DataParallel can distribute the batch across GPUs.
                    preds = model(images_scene)

                    # The model returns 'pose_enc' (B, S, 9) under predictions
                    pose_enc = preds.get("pose_enc")
                    if pose_enc is None:
                        # Fallback: some model variants may expose 'pose_enc_list'
                        pose_enc_list = preds.get("pose_enc_list")
                        if pose_enc_list is not None:
                            pose_enc = pose_enc_list[-1]
                        else:
                            raise RuntimeError("Model did not return pose_enc in predictions")

                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_scene.shape[-2:])

            # extrinsic shape: (B, 2, 3, 4) or similar
            extrinsic0 = extrinsic[:, 0]
            extrinsic1 = extrinsic[:, 1]

            for k in range(cur_batch):
                yaw_deg, forward_m, _ = compute_action_from_poses(extrinsic0[k], extrinsic1[k])
                trajectory.append({'yaw_deg': yaw_deg, 'forward_m': forward_m})

            start += cur_batch

        except RuntimeError as e:
            msg = str(e).lower()
            if 'out of memory' in msg or 'cuda out of memory' in msg:
                # Reduce batch size and retry
                if batch_size <= 1:
                    # Can't reduce further; re-raise
                    raise
                old_batch = batch_size
                batch_size = max(1, batch_size // 2)
                if verbose:
                    print(f"CUDA OOM during batched inference, reducing batch_size from {old_batch} to {batch_size} and retrying...")
                # Give GPU a chance to recover
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                raise

    return trajectory

# --- Main processing routine ---

def process_sequence(model, device, dtype, seq_folder: Path, out_folder: Path, args):
    seq_folder = Path(seq_folder)
    imgs = sorted_image_list(seq_folder, exts=tuple(args.image_exts))
    if len(imgs) < 2:
        return 0

    # Subsample images
    if args.subsample > 1:
        imgs = imgs[::args.subsample]
        if len(imgs) < 2:
            return 0

    out_folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    
    # Use the new trajectory function
    trajectory = get_motion_trajectory(model, device, dtype, imgs)
    actions_list = [[step['forward_m'], step['yaw_deg']] for step in trajectory]

    for i, step in enumerate(trajectory):
        cur = imgs[i]
        nxt = imgs[i+1]
        yaw_deg = step['yaw_deg']
        forward_m = step['forward_m']
        
        # Re-compute action string for visualization
        action = "none"
        if yaw_deg > args.yaw_thresh_deg:
            action = "rotate_left"
        elif yaw_deg < -args.yaw_thresh_deg:
            action = "rotate_right"
        elif forward_m > args.forward_thresh_m:
            action = "forward"
        
        # create visualization and save
        pil_cur = Image.open(cur).convert("RGB")
        pil_nxt = Image.open(nxt).convert("RGB")
        combined = stack_side_by_side(pil_cur, pil_nxt)
        combined = draw_overlay(combined, action, yaw_deg, forward_m)

        outname = out_folder / f"{i:06d}_{cur.stem}__{nxt.stem}.jpg"
        combined.save(outname, quality=90)
        saved += 1
        if args.verbose:
            print(f"Saved: {outname}  action={action} yaw={yaw_deg:.2f}deg forward={forward_m:.3f}m")

    print(f"Intermediate actions for {seq_folder.name}: {actions_list}")
    return saved

def extract_frames_from_video(video_path: Path, output_dir: Path, target_fps: float = 24.0):
    """
    Extract frames from video at target FPS.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        target_fps: Target frame rate for extraction (default: 24.0)
    
    Returns:
        Number of frames extracted
    """
    import cv2
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(original_fps / target_fps)))
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at target FPS intervals
        if frame_count % frame_interval == 0:
            frame_filename = output_dir / f"{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path.name} (original FPS: {original_fps:.2f}, target: {target_fps})")
    return saved_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=str, help="Root folder containing sequence subfolders.")
    parser.add_argument("--video", type=str, help="Path to video file (alternative to --frames_dir)")
    parser.add_argument("--target_fps", type=float, default=24.0, help="Target FPS when extracting from video. Controls frame sampling rate (e.g., 1.0 = 1 frame/sec). Only used with --video.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", type=str, default="facebook/VGGT-1B", help="VGGT pretrained model name or local.")
    parser.add_argument("--image_exts", nargs="+", default=[".png", ".jpg", ".jpeg"])
    parser.add_argument("--out_suffix", type=str, default="_viz")
    parser.add_argument("--yaw_thresh_deg", type=float, default=3.0, help="Degrees threshold to decide rotate left/right.")
    parser.add_argument("--forward_thresh_m", type=float, default=0.02, help="Meters threshold for forward movement.")
    parser.add_argument("--subsample", type=int, default=1, help="Subsample every Nth frame (only for --frames_dir, not --video)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Validate input arguments
    if not args.frames_dir and not args.video:
        parser.error("Either --frames_dir or --video must be specified")
    if args.frames_dir and args.video:
        parser.error("Cannot specify both --frames_dir and --video")
    
    # Handle video input
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            parser.error(f"Video file does not exist: {args.video}")
        
        if args.subsample != 1:
            print("Warning: --subsample is ignored when using --video. Use --target_fps to control sampling rate.")
        
        # Extract frames to temporary directory at target_fps (no additional subsampling needed)
        temp_frames_dir = Path("temp_video_frames") / video_path.stem
        extract_frames_from_video(video_path, temp_frames_dir, args.target_fps)
        frames_root = temp_frames_dir
        cleanup_temp = True
        # Override subsample to 1 since target_fps already controls sampling
        args.subsample = 1
    else:
        frames_root = Path(args.frames_dir)
        cleanup_temp = False
    device, dtype = ensure_device_and_dtype(args.device)
    print(f"Using device={device}, dtype={dtype}")

    # Initialize model
    print("Loading VGGT model (this may download weights the first time)...")
    model = VGGT.from_pretrained(args.model_name).to(device)
    model.eval()

    total_saved = 0
    # Check if frames_root contains image files directly (single sequence)
    image_files = sorted_image_list(frames_root, exts=tuple(args.image_exts))
    if len(image_files) >= 2:
        # Treat frames_root as a single sequence
        out_folder = frames_root.parent / (frames_root.name + args.out_suffix)
        saved = process_sequence(model, device, dtype, frames_root, out_folder, args)
        total_saved += saved
        if args.verbose:
            print(f"Processed sequence {frames_root.name}: saved {saved} visualizations -> {out_folder}")
    else:
        # Iterate over subdirectories as sequences
        for seq in sorted(frames_root.iterdir()):
            if not seq.is_dir():
                continue
            out_folder = seq.parent / (seq.name + args.out_suffix)
            saved = process_sequence(model, device, dtype, seq, out_folder, args)
            total_saved += saved
            if args.verbose:
                print(f"Processed sequence {seq.name}: saved {saved} visualizations -> {out_folder}")

    print(f"Done. Total saved: {total_saved}")
    
    # Cleanup temporary frames if video input was used
    if cleanup_temp:
        import shutil
        shutil.rmtree(frames_root.parent)
        print(f"Cleaned up temporary frames directory")

if __name__ == "__main__":
    main()
