#!/usr/bin/env python3
"""
data_augmentation_mirror.py

Creates mirror-augmented dataset by horizontally flipping videos that involve left/right turns
and swapping the direction words in the captions.

Usage:
    python data_augmentation_mirror.py --input <dataset.json> [--target_dir <path>] [--dry_run] [--mirror_forward_only]

Examples:
    # Dry-run in-place (no files written)
    python data_augmentation_mirror.py --input imaginav-dataset/all_clips_augmented_0118/dataset_panned.json --dry_run

    # Run and write mirrored videos and augmented JSON into a separate target folder
    python data_augmentation_mirror.py --input imaginav-dataset/all_clips_augmented_0118/dataset_panned.json --target_dir /path/to/target

Notes:
    - If --target_dir is omitted or equals the input folder name, the script operates in-place.
    - The script detects TURN_/PAN_ primitives in filenames and skips entries already containing '_mirrored'.
"""

import argparse
import json
import re
import cv2
import numpy as np
import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

def swap_left_right(text):
    """
    Swap left/right words in text.
    Handles: left, right, Left, Right, LEFT, RIGHT
    """
    # Create a temporary placeholder to avoid double-swapping
    text = re.sub(r'\bleft\b', '<TEMP_LEFT>', text, flags=re.IGNORECASE)
    text = re.sub(r'\bright\b', '<TEMP_RIGHT>', text, flags=re.IGNORECASE)
    
    # Now swap using the placeholders
    text = text.replace('<TEMP_LEFT>', 'right')
    text = text.replace('<TEMP_RIGHT>', 'left')
    text = text.replace('<TEMP_LEFT>'.upper(), 'RIGHT')
    text = text.replace('<TEMP_RIGHT>'.upper(), 'LEFT')
    
    # Handle capitalized versions
    text = re.sub(r'\bRight\b', lambda m: 'Left', text)
    text = re.sub(r'\bLeft\b', lambda m: 'Right', text)
    
    return text

def swap_filename_left_right(filename):
    """
    Swap LEFT/RIGHT in filename and add _mirrored suffix.
    Supports both old naming (TURN_*, MOVE_FORWARD_*) and new naming
    (PAN_*, DOLLY_FORWARD_*). Uses temporary placeholders to avoid double
    swapping and forces output to .mp4.
    Example: IMG_3685_TURN_LEFT.avi -> IMG_3685_TURN_RIGHT_mirrored.mp4
    Example: IMG_3685_DOLLY_FORWARD_PAN_LEFT.mov -> IMG_3685_DOLLY_FORWARD_PAN_RIGHT_mirrored.mp4
    """
    # Split filename and extension
    name_parts = filename.rsplit('.', 1)
    base_name = name_parts[0]

    # Avoid double-appending _mirrored
    if base_name.endswith('_mirrored'):
        base_root = base_name[: -len('_mirrored')]
    else:
        base_root = base_name

    # Mapping of token -> swapped token. Include both old and new token styles.
    swap_map = {
        'MOVE_FORWARD_TURN_LEFT': 'MOVE_FORWARD_TURN_RIGHT',
        'MOVE_FORWARD_TURN_RIGHT': 'MOVE_FORWARD_TURN_LEFT',
        'TURN_LEFT_MOVE_FORWARD': 'TURN_RIGHT_MOVE_FORWARD',
        'TURN_RIGHT_MOVE_FORWARD': 'TURN_LEFT_MOVE_FORWARD',
        'TURN_LEFT': 'TURN_RIGHT',
        'TURN_RIGHT': 'TURN_LEFT',
        'PAN_LEFT': 'PAN_RIGHT',
        'PAN_RIGHT': 'PAN_LEFT',
        'DOLLY_FORWARD_PAN_LEFT': 'DOLLY_FORWARD_PAN_RIGHT',
        'DOLLY_FORWARD_PAN_RIGHT': 'DOLLY_FORWARD_PAN_LEFT',
        'PAN_LEFT_DOLLY_FORWARD': 'PAN_RIGHT_DOLLY_FORWARD',
        'PAN_RIGHT_DOLLY_FORWARD': 'PAN_LEFT_DOLLY_FORWARD',
    }

    # Replace longer tokens first using temporary placeholders to avoid double-swapping
    base = base_root
    placeholder_map = {}
    idx = 0
    for key in sorted(swap_map.keys(), key=len, reverse=True):
        if key in base:
            placeholder = f'__TMP_{idx}__'
            base = base.replace(key, placeholder)
            placeholder_map[placeholder] = swap_map[key]
            idx += 1

    for placeholder, target in placeholder_map.items():
        base = base.replace(placeholder, target)

    # Add _mirrored suffix and preserve original extension
    # Determine original extension from input filename and preserve it (default to .mp4)
    orig_ext = '.' + name_parts[1].lower() if len(name_parts) > 1 else '.mp4'
    return f"{base}_mirrored{orig_ext}"

def add_mirrored_suffix(filename):
    """
    Add _mirrored suffix to a filename without swapping left/right tokens.
    Preserves the original extension (default to .mp4 if missing).
    Example: IMG_0001_MOVE_FORWARD.avi -> IMG_0001_MOVE_FORWARD_mirrored.avi
    """
    name_parts = filename.rsplit('.', 1)
    base_name = name_parts[0]

    if base_name.endswith('_mirrored'):
        base_root = base_name[: -len('_mirrored')]
    else:
        base_root = base_name

    orig_ext = '.' + name_parts[1].lower() if len(name_parts) > 1 else '.mp4'
    return f"{base_root}_mirrored{orig_ext}"

def has_left_right_turn(caption, filename):
    """
    Check if the video involves left/right turns or pans.
    Returns True if the caption mentions left/right and the filename contains
    either TURN_* or PAN_* motion primitives. Works case-insensitively.
    """
    caption_lower = caption.lower()
    filename_upper = filename.upper()

    # Check caption for direction words
    has_direction_in_caption = bool(re.search(r'\b(left|right)\b', caption_lower))

    # Check filename for either turn or pan motion primitives
    has_turn_in_filename = bool(re.search(r'(TURN|PAN)_(LEFT|RIGHT)', filename_upper))

    return has_direction_in_caption and has_turn_in_filename

def mirror_video(input_path, output_path):
    """
    Mirror (horizontally flip) a video and save it.
    Uses ffmpeg for final encoding to ensure compatibility (h264, yuv420p).
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use mp4v codec which is widely supported for temp file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Check output extension, recommend .mp4
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.mp4':
        print(f"Warning: Output extension {output_path.suffix} might not be compatible. forcing .mp4")
        output_path = output_path.with_suffix('.mp4')

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        cap.release()
        return False
    
    # Read, flip, and write each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Horizontal flip
        flipped_frame = cv2.flip(frame, 1)
        out.write(flipped_frame)
    
    cap.release()
    out.release()

    # Re-encode using ffmpeg to ensure VS Code compatibility (h264, yuv420p)
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
            # Run silently. 
            # Note: If this fails, we will be left without the file or with the temp file.
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

    return True

def process_dataset(input_json, target_dir=None, dry_run=False, mirror_forward_only=False):
    """
    Simplified core processing:
    - Read `input_json` (expects entries with `media_path` and `caption`).
    - Determine `input_base_dir` (json parent) and `target_dir` (where to write results). If
      `target_dir` is omitted or equals input base directory name, operate in-place.
    - For each entry in the input dataset:
      * Skip entries whose filename already contains '_mirrored' (we don't mirror mirrors).
      * Ensure the original video is present in the target (create a symlink if target is different).
      * If the filename contains TURN_* or PAN_ (detected via filename), generate the mirrored filename
        and caption, and if the mirrored media path does not already exist in the dataset or on disk,
        create the mirrored video file (or simulate if dry-run).
    - Append newly-created mirrored entries to `dataset_augmented.json` in the target directory.
    """
    input_json_path = Path(input_json).resolve()
    input_base_dir = input_json_path.parent

    # Resolve target directory
    if target_dir:
        target_base_dir = Path(target_dir).resolve()
        # If only the name matches the input folder (user-friendly behavior), treat as in-place
        if target_base_dir.name == input_base_dir.name:
            print(f"Target name matches input folder name; operating in-place in {input_base_dir}")
            target_base_dir = input_base_dir
    else:
        target_base_dir = input_base_dir

    in_place = (target_base_dir.resolve() == input_base_dir.resolve())

    output_json_path = target_base_dir / "dataset_augmented.json"

    print(f"Loading dataset from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} entries")

    # Build set of existing media paths (case-insensitive) from dataset
    existing_media = set(e.get('media_path').upper() for e in dataset if 'media_path' in e)

    new_entries = []
    created_mirrors = 0
    created_symlinks = 0
    skipped_already_mirrored = 0
    skipped_already_exists = 0
    missing_source = []
    # Track cases where a mirrored file exists but with a different extension (e.g., .mp4 exists while the original is .avi)
    skipped_due_to_ext_mismatch = []

    # Common video extensions to check for extension-mismatch detection
    VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.m4v']

    # Ensure target root exists
    target_base_dir.mkdir(parents=True, exist_ok=True)

    for entry in dataset:
        caption = entry.get('caption','')
        media_path = entry.get('media_path')
        if not media_path:
            continue

        filename = Path(media_path).name
        stem = Path(filename).stem

        # If entry is already a mirrored file, we still want to ensure the mirrored file
        # is present in the TARGET (by creating a symlink) when operating out-of-place.
        if '_mirrored' in stem.lower():
            if not in_place:
                target_original_path = target_base_dir / media_path
                input_original_path = input_base_dir / media_path
                target_original_path.parent.mkdir(parents=True, exist_ok=True)
                if not target_original_path.exists():
                    if dry_run:
                        created_symlinks += 1
                    else:
                        try:
                            os.symlink(input_original_path, target_original_path)
                            created_symlinks += 1
                        except OSError as e:
                            print(f"Warning: failed to symlink {input_original_path} -> {target_original_path}: {e}")
            skipped_already_mirrored += 1
            # Don't attempt to mirror a mirrored file; move to next entry
            continue

        # Ensure original exists in target: if not in-place, create symlink for original
        target_original_path = target_base_dir / media_path
        input_original_path = input_base_dir / media_path
        if not in_place:
            target_original_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_original_path.exists():
                if dry_run:
                    created_symlinks += 1
                else:
                    try:
                        os.symlink(input_original_path, target_original_path)
                        created_symlinks += 1
                    except OSError as e:
                        print(f"Warning: failed to symlink {input_original_path} -> {target_original_path}: {e}")

        # Determine if we should mirror
        filename_upper = filename.upper()
        is_turn_or_pan = bool(re.search(r'(TURN|PAN)_(LEFT|RIGHT)', filename_upper))
        is_forward_only = bool(re.search(r'(MOVE_FORWARD|DOLLY_FORWARD)', filename_upper)) and not is_turn_or_pan

        should_mirror = is_turn_or_pan or (mirror_forward_only and is_forward_only)
        if should_mirror:
            if is_turn_or_pan:
                mirrored_filename = swap_filename_left_right(filename)
                mirrored_caption = swap_left_right(caption)
            else:
                mirrored_filename = add_mirrored_suffix(filename)
                mirrored_caption = caption

            mirrored_media_path = str(Path(media_path).parent / mirrored_filename)

            # Skip if mirrored entry already present in original dataset
            if mirrored_media_path.upper() in existing_media:
                skipped_already_exists += 1
                continue

            # Check on-disk existence in target (exact extension)
            target_mirrored_full = target_base_dir / mirrored_media_path
            if target_mirrored_full.exists():
                skipped_already_exists += 1
                existing_media.add(mirrored_media_path.upper())
                continue

            # Check for mirrored files with a different extension, but prefer .mp4
            stem = Path(mirrored_media_path).stem  # includes _mirrored suffix
            mir_dir = Path(mirrored_media_path).parent

            # Prefer .mp4 as canonical mirrored format
            mp4_candidate = target_base_dir / mir_dir / f"{stem}.mp4"
            found_variant = None
            if mp4_candidate.exists() or str(Path(mir_dir / f"{stem}.mp4")).upper() in existing_media:
                # .mp4 exists -> treat as present and do NOT warn
                found_variant = mp4_candidate
            else:
                # Look for any other extension variants and warn only if no .mp4 is found
                for ext in [e for e in VIDEO_EXTS if e != '.mp4']:
                    candidate = target_base_dir / mir_dir / f"{stem}{ext}"
                    if candidate.exists() or str(Path(mir_dir / f"{stem}{ext}")).upper() in existing_media:
                        found_variant = candidate
                        break

            if found_variant is not None:
                skipped_already_exists += 1
                # Only record an extension-mismatch warning if the found variant is NOT .mp4
                if not (str(found_variant).lower().endswith('.mp4')):
                    skipped_due_to_ext_mismatch.append({'expected': str(target_mirrored_full), 'found': str(found_variant)})
                existing_media.add(str(mir_dir / stem).upper())
                # Do not create a new mirrored file; move on
                continue
            # Check source existence
            if not (input_original_path.exists()):
                # source missing, cannot create mirrored video
                missing_source.append(str(input_original_path))
                continue

            # Create mirrored video
            target_mirrored_full.parent.mkdir(parents=True, exist_ok=True)
            if dry_run:
                created_mirrors += 1
            else:
                success = mirror_video(input_original_path, target_mirrored_full)
                if success:
                    created_mirrors += 1
                else:
                    print(f"Warning: failed to create mirrored video for {input_original_path}")

            # Record new entry
            new_entries.append({
                'caption': mirrored_caption,
                'media_path': mirrored_media_path
            })
            existing_media.add(mirrored_media_path.upper())

    # Append new entries and save
    if dry_run:
        print('\nDry-run summary:')
        print('Would create symlinks for originals:', created_symlinks)
        print('Would create mirrored videos:', created_mirrors)
        print('Skipped (already mirrored entries in input):', skipped_already_mirrored)
        print('Skipped (mirrored already exists):', skipped_already_exists)
        print('Skipped (mirrored exists but with different extension):', len(skipped_due_to_ext_mismatch))
        if skipped_due_to_ext_mismatch:
            print('Examples of extension mismatches (expected -> found):')
            for ex in skipped_due_to_ext_mismatch[:10]:
                print(' -', ex['expected'], '->', ex['found'])
        print('Missing sources (cannot mirror):', len(missing_source))
        if missing_source:
            print('Examples:', missing_source[:10])
        return

    if len(new_entries) > 0:
        # Append and save augmented JSON in the target folder
        augmented = list(dataset) + new_entries
        with open(output_json_path, 'w') as f:
            json.dump(augmented, f, indent=2)
        print(f"Appended {len(new_entries)} mirrored entries and saved to {output_json_path}")
    else:
        print("No new mirrored entries to append.")

    print('\nSummary:')
    print('Mirrored videos created:', created_mirrors)
    print('Symlinks for originals created:', created_symlinks)
    print('Skipped mirrored-in-input entries:', skipped_already_mirrored)
    print('Skipped because mirrored already existed:', skipped_already_exists)
    if missing_source:
        print('Missing sources (could not mirror):', len(missing_source))

    # End of processing — (legacy output block removed in refactor).
    # Summary already printed above for in-place operations.

def main():
    parser = argparse.ArgumentParser(description="Simple mirror-augment: mirror TURN_/PAN_ videos and append to dataset.")
    parser.add_argument("--input", type=str, required=True, help="Input dataset JSON file (required)")
    parser.add_argument("--target_dir", type=str, required=False, help="Target directory to write mirrored videos and augmented JSON. If omitted, operate in-place")
    parser.add_argument("--mirror_forward_only", action="store_true", default=False, help="Also mirror forward-only clips (default: False)")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Simulate operations without writing files (dry run)")

    args = parser.parse_args()

    process_dataset(
        input_json=args.input,
        target_dir=args.target_dir,
        dry_run=args.dry_run,
        mirror_forward_only=args.mirror_forward_only
    )

if __name__ == "__main__":
    main()
