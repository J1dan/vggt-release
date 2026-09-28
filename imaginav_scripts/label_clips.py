#!/usr/bin/env python3
"""
label_clips.py

Uses Gemini API to generate navigation instructions for video clips based on their motion primitives.
Reads metadata from clips_metadata.json and processes videos to create labeled dataset.

Usage:
    python label_clips.py --clips clip1.avi clip2.avi clip3.avi
    python label_clips.py --input_folder motion_clips/
    python label_clips.py --clips motion_clips/*.avi --output labels.json
"""

import os
import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
from PIL import Image as PILImage

from google import genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_and_parse_json(response: str):
    """
    Remove markdown code blocks, handle double-braces, and parse JSON.
    """
    # 1. Remove Markdown code blocks (```json ... ```)
    cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # 2. Find the first opening brace and last closing brace
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')

    if start_idx == -1 or end_idx == -1:
        raise ValueError("VLM response does not contain JSON braces.")

    # Extract just the JSON part
    cleaned = cleaned[start_idx:end_idx+1]

    # 3. DETECT AND FIX DOUBLE BRACES
    # If the string starts with {{ and ends with }}, strip the outer layer
    if cleaned.startswith("{{") and cleaned.endswith("}}"):
        cleaned = cleaned[1:-1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("Raw VLM response:", repr(response))
        print("Cleaned string:", repr(cleaned))
        raise

class VideoLabeler:
    """
    Handles labeling of video clips using Gemini VLM.
    """
    def __init__(self, api_key: Optional[str] = None, dummy_mode: bool = False):
        self.dummy_mode = dummy_mode
        if self.dummy_mode:
             logging.info("Initializing Video Labeler in DUMMY MODE (no API calls).")
             return

        logging.info("Initializing Video Labeler...")
        
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logging.error("GOOGLE_API_KEY environment variable not set!")
                raise ValueError("API Key not found. Set GOOGLE_API_KEY or pass api_key parameter.")
        
        self.client = genai.Client(api_key=api_key)
        logging.info("Video Labeler initialized successfully.")

    def get_prompt(self, primitives: List[str], include_auxiliary: bool = True) -> str:
        """
        Constructs the labeling prompt with motion primitives.
        
        Args:
            primitives: List of motion primitives in temporal order (e.g., ["TURN_LEFT", "MOVE_FORWARD"])
            include_auxiliary: Whether to include the auxiliary information section
        
        Returns:
            The complete prompt string
        """
        primitives_str = ", ".join(primitives)
        
        prompt = f"""Role: You are an expert robot navigation data labeler.

Task: Your goal is to reverse-engineer the navigation command that caused the robot's motion.

Instructions:
1. Analyze the entire video: Observe the robot's full trajectory within the clip.
2. Formulate the Instruction: Write the specific command that would prompt this behavior.

MANDATORY CONSTRAINTS:
- Video-Based Reasoning: You must deduce the intent from the full motion shown in the video (e.g., if the robot moves forward and stops at a chair, the intent is navigating to that chair).
- First-Frame Grounding: While you use the full video to understand the intent, the specific landmarks or objects referenced in your text must be visible in the first frame. Do not hallucinate objects that only appear later in the video.
- Brevity: The instruction must be a single, imperative sentence.
- Motion Primitives: Prioritize camera-centric descriptions.

Examples:
- "Dolly forward towards the [object]"
- "Pan left and dolly forward"
- "Dolly forward and pan right at the [object/location]"
- "Dolly forward down the [corridor/path]"

"""
        
        if include_auxiliary and primitives:
            # Translate primitive tokens for clarity in auxiliary info (e.g., MOVE_FORWARD -> "dolly forward")
            mapping = {
                'MOVE_FORWARD': 'Dolly_forward',
                'TURN_LEFT': 'Pan_left',
                'TURN_RIGHT': 'Pan_right'
            }

            def _translate(p: str) -> str:
                key = p.upper()
                if key in mapping:
                    return mapping[key]
                # Fallback: replace underscores with spaces and lowercase
                return p.replace('_', ' ').lower()

            translated_primitives = ", ".join(_translate(p) for p in primitives)

            prompt += f"""Auxiliary Information:
You are provided with the pre-extracted motions from a deterministic analyzer in temporal order: [{translated_primitives}]

Note that this information may have noise and is just for auxiliary purposes. You should decide the main motion pattern and discard the unreliable ones when necessary.

"""
        
        prompt += """Output Format: Provide your response in valid JSON:
{{
  "instruction": "The single-sentence navigation command.",
  "justification": "A brief explanation citing specific visual evidence from the video. Describe the motion primitives."
}}"""
        
        return prompt

    def extract_frames_from_video(self, video_path: Path, fps: float = 2.0) -> List[PILImage.Image]:
        """
        Extract frames from video at specified fps rate.
        
        Args:
            video_path: Path to video file
            fps: Frames per second to extract (default: 2.0)
        
        Returns:
            List of PIL Images
        """
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if video_fps <= 0:
            raise ValueError(f"Could not get FPS from video: {video_path}")
        
        duration = total_frames / video_fps
        
        # Validation checks
        if duration <= 3.0:
            raise ValueError(f"Video duration {duration:.2f}s is too short (must be > 3s)")
        
        if total_frames != 121:
            raise ValueError(f"Video has {total_frames} frames (must have exactly 121)")

        # Calculate frame indices at desired fps
        frame_indices = []
        time_step = 1.0 / fps
        current_time = 0.0
        while current_time < duration:
            frame_idx = int(current_time * video_fps)
            if frame_idx < total_frames:
                frame_indices.append(frame_idx)
            current_time += time_step
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = PILImage.fromarray(frame_rgb)
                frames.append(pil_image)
        
        cap.release()
        return frames

    def validate_video_file(self, video_path: Path) -> Dict[str, float]:
        """
        Lightweight validation of the video file (does not extract frames).
        Checks FPS and duration (> 3s).
        Returns dict with 'total_frames', 'video_fps', 'duration'.
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        if video_fps <= 0:
            cap.release()
            raise ValueError(f"Could not get FPS from video: {video_path}")

        duration = total_frames / video_fps

        if duration <= 3.0:
            cap.release()
            raise ValueError(f"Video duration {duration:.2f}s is too short (must be > 3s)")

        cap.release()
        return {"total_frames": total_frames, "video_fps": video_fps, "duration": duration}

    def label_video(self, video_path: Path, primitives: List[str], fps: float = 2.0, include_auxiliary: bool = True) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Generate navigation instruction for a single video clip.
        
        Args:
            video_path: Path to video file
            primitives: List of motion primitives in temporal order
            fps: Frames per second to extract from video (default: 2.0)
            include_auxiliary: Whether to include auxiliary motion primitives in the prompt
        
        Returns:
            Tuple containing:
            - Dictionary with 'instruction' and 'justification' keys (or None on failure)
            - Error message string (or None on success)
        """
        import time
        
        logging.info(f"Labeling video: {video_path.name}")
        logging.info(f"Motion primitives: {primitives}")
        
        # Lightweight validation (don't extract frames when sending video directly)
        try:
            meta = self.validate_video_file(video_path)
            logging.info(f"Video validated: duration={meta['duration']:.2f}s, fps={meta['video_fps']:.2f}, frames={meta['total_frames']}")
        except Exception as e:
            msg = f"Failed to validate video {video_path.name}: {e}"
            logging.error(msg)
            return None, msg

        if self.dummy_mode:
            logging.info("DUMMY MODE: Skipping API call, returning placeholder.")
            return {"instruction": "haha", "justification": "baba"}, None

        # Build prompt (the video file will be uploaded and sent alongside the prompt)
        prompt_text = self.get_prompt(primitives, include_auxiliary)
        # Clarify that the video file is attached
        prompt_text = "Attached is the video file showing the motion. " + prompt_text

        try:
            logging.info(f"Uploading video {video_path.name} to API...")
            file_obj = self.client.files.upload(file=str(video_path))

            logging.info("Waiting for video processing...")
            while file_obj.state == "PROCESSING":
                time.sleep(2)
                # Refresh file status
                file_obj = self.client.files.get(name=file_obj.name)
            
            if file_obj.state != "ACTIVE":
                raise ValueError(f"Video processing failed. State: {file_obj.state}")

            logging.info("Sending video and prompt to Gemini model via generate_content...")
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview", # NOTE: See Model Recommendation below
                contents=[file_obj, prompt_text]
            )
            raw_text = getattr(response, 'text', str(response))

            # Parse JSON response
            parsed = clean_and_parse_json(raw_text)

            if 'instruction' not in parsed or 'justification' not in parsed:
                msg = f"Response missing required fields: {parsed}"
                logging.warning(msg)
                return None, msg

            logging.info(f"Generated instruction: {parsed['instruction']}")
            return parsed, None

        except Exception as e:
            msg = f"Failed to label {video_path.name}: {e}"
            logging.warning(msg)
            return None, msg

def load_metadata(metadata_path: Path) -> Dict[str, Dict]:
    """
    Load clips metadata and create a mapping from filename to metadata.

    Args:
        metadata_path: Path to clips_metadata.json (or sum_metadata.json)

    Returns:
        Dictionary mapping filename and any available relative paths to metadata entries
    """
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    # Create a flexible mapping: filename -> metadata, and include 'relative_path' if present
    metadata_map = {}
    for entry in metadata_list:
        filename = entry.get('filename')
        if filename:
            metadata_map[filename] = entry
        # Some metadata lists may store a relative path key
        rel = entry.get('relative_path') or entry.get('path')
        if rel:
            metadata_map[rel] = entry

    logging.info(
        f"Loaded metadata list size: {len(metadata_list)}; unique lookup keys: {len(metadata_map)} "
        f"(keys may include filenames and relative paths)"
    )
    return metadata_map


def get_metadata_for_video(metadata_map: Dict[str, Dict], video_path: Path, metadata_root: Path) -> Optional[Dict]:
    """
    Attempt to find the metadata entry for a given video path.

    Tries, in order:
    - Exact filename match (e.g., 'video.mp4')
    - Relative path from the metadata root (e.g., 'canteen/video.mp4')
    - POSIX-style relative path

    Returns the metadata entry or None if not found.
    """
    filename = video_path.name
    if filename in metadata_map:
        return metadata_map[filename]

    # Try relative to metadata root
    try:
        rel = video_path.relative_to(metadata_root)
        rel_str = str(rel)
        if rel_str in metadata_map:
            return metadata_map[rel_str]
        # also try POSIX
        rel_posix = rel.as_posix()
        if rel_posix in metadata_map:
            return metadata_map[rel_posix]
        # Try just the name from the relative path
        if Path(rel_str).name in metadata_map:
            return metadata_map[Path(rel_str).name]
    except Exception:
        pass

    # As a last resort, try full posix path
    full_posix = video_path.as_posix()
    if full_posix in metadata_map:
        return metadata_map[full_posix]

    return None

def find_metadata_file(video_path: Path) -> Optional[Path]:
    """
    Find the clips_metadata.json file in the same directory as the video.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Path to metadata file, or None if not found
    """
    metadata_path = video_path.parent / "clips_metadata.json"
    if metadata_path.exists():
        return metadata_path
    
    # Try parent directory
    metadata_path = video_path.parent.parent / "clips_metadata.json"
    if metadata_path.exists():
        return metadata_path
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Label video clips using Gemini API")
    parser.add_argument("--clips", nargs="+", help="Path(s) to video clip(s) to label")
    parser.add_argument("--input_folder", type=str, help="Folder containing video clips")
    parser.add_argument("--output", type=str, default="labeled_results/labeled_clips.json", help="Output JSON file for labels")
    parser.add_argument("--api_key", type=str, help="Google API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract from videos")
    parser.add_argument("--metadata", type=str, help="Path to clips_metadata.json (auto-detected if not provided)")
    parser.add_argument("--resume_from", type=str, help="Path to a previously processed JSON file to resume from (skips already labeled videos)")
    parser.add_argument("--dummy", action="store_true", help="Run in dummy mode (no API calls, returns 'haha' instruction)")
    parser.add_argument("--label_rejected", action="store_true", help="Include videos in 'rejected' folders (omit auxiliary information in prompts)")

    args = parser.parse_args()
    
    # Auto-infer input_folder from metadata path if not provided
    if not args.input_folder and not args.clips and args.metadata:
        inferred_folder = Path(args.metadata).parent
        logging.info(f"Inferring input folder from metadata path: {inferred_folder}")
        args.input_folder = str(inferred_folder)

    # Collect video files
    video_files = []
    
    if args.clips:
        video_files = [Path(clip) for clip in args.clips if Path(clip).exists()]
        if len(video_files) < len(args.clips):
            logging.warning(f"Some specified clips do not exist")
    
    if args.input_folder:
        folder = Path(args.input_folder)
        if folder.exists():
            logging.info(f"Searching for videos recursively under {folder}")
            # Use recursive search to find videos in nested subdirectories
            video_files.extend(folder.rglob("*.avi"))
            video_files.extend(folder.rglob("*.mp4"))
            video_files.extend(folder.rglob("*.MOV"))
        else:
            logging.error(f"Input folder does not exist: {args.input_folder}")
            return
    
    if not video_files:
        logging.error("No video files found to process")
        return
    
    logging.info(f"Found {len(video_files)} video files to process")
    
    # Find and load metadata
    if args.metadata:
        metadata_path = Path(args.metadata)
    elif args.input_folder:
        # Prefer a single metadata file located at the root of the input folder
        folder = Path(args.input_folder)
        metadata_path = None
        for name in ("sum_metadata.json", "clips_metadata.json"):
            candidate = folder / name
            if candidate.exists():
                metadata_path = candidate
                logging.info(f"Using metadata file: {metadata_path}")
                break
        if metadata_path is None:
            logging.error(f"Could not find sum_metadata.json or clips_metadata.json in {folder}. Please specify with --metadata")
            return
    else:
        # Fallback to legacy behavior (search near first video)
        metadata_path = find_metadata_file(video_files[0])
        if not metadata_path:
            logging.error("Could not find clips_metadata.json. Please specify with --metadata")
            return
    
    logging.info(f"Loading metadata from: {metadata_path}")
    metadata_map = load_metadata(metadata_path)
    
    # Also load rejected metadata if available and --label_rejected is set
    if args.label_rejected:
        rejected_metadata_path = metadata_path.parent / "sum_rejected.json"
        if rejected_metadata_path.exists():
            logging.info(f"Loading rejected metadata from: {rejected_metadata_path}")
            rejected_metadata_map = load_metadata(rejected_metadata_path)
            metadata_map.update(rejected_metadata_map)
            logging.info(f"Total metadata entries after merging: {len(metadata_map)}")
    
    metadata_root = metadata_path.parent
    
    # Initialize labeler
    labeler = VideoLabeler(api_key=args.api_key, dummy_mode=args.dummy)
    
    # Generate output filename with date stamp
    from datetime import datetime
    date_stamp = datetime.now().strftime("%Y%m%d")
    
    # Determine folder name from input
    if args.input_folder:
        folder_name = Path(args.input_folder).name
    elif args.clips:
        folder_name = Path(args.clips[0]).parent.name
    else:
        folder_name = "clips"
    
    # Update output path with timestamp and folder name
    dataset_json_path = Path(args.output)
    if dataset_json_path.name == "labeled_clips.json":  # Default output name
        dataset_json_path = dataset_json_path.parent / f"dataset-{date_stamp}.json"
    else:
        # User specified custom name, use it directly (or add prefix if it's just a filename)
        if dataset_json_path.parent == Path('.'):
             dataset_json_path = Path("labeled_results") / dataset_json_path
        # Append date stamp before suffix
        if dataset_json_path.suffix == ".json":
            dataset_json_path = dataset_json_path.with_name(f"{dataset_json_path.stem}-{date_stamp}.json")
        else:
            dataset_json_path = dataset_json_path.with_name(f"{dataset_json_path.name}-{date_stamp}")
    
    # Create output directory if it doesn't exist
    dataset_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define failure log path
    failure_output_path = dataset_json_path.parent / f"{dataset_json_path.stem}_failures.json"

    # Dataset metadata path (full labels)
    metadata_output_path = dataset_json_path.with_name(f"{dataset_json_path.stem}-metadata.json")
    
    # Load resume data if provided
    processed_files = set()
    results = []  # Full label results (instruction + metadata)
    existing_captions = []  # Caption-only entries from resume, if any
    initial_results_count = 0
    
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            try:
                with open(resume_path, 'r') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        # Determine if resume file contains full label entries or caption-only entries
                        has_instruction = any(isinstance(e, dict) and 'instruction' in e for e in existing_data)
                        if has_instruction:
                            results = existing_data
                            initial_results_count = len(results)
                        else:
                            existing_captions = existing_data
                        for entry in existing_data:
                            # Use multiple possible keys to track processed entries
                            for key in ("filename", "relative_path", "path", "video", "media_path"):
                                val = entry.get(key)
                                if val:
                                    processed_files.add(val)
                                    processed_files.add(Path(val).name)
                        logging.info(
                            f"Resuming from {resume_path}: Loaded {len(existing_data)} entries; "
                            f"tracked {len(processed_files)} processed identifiers (paths + basenames)."
                        )
                    else:
                        logging.warning(f"Resume file {resume_path} is not a valid list. Starting fresh.")
            except Exception as e:
                logging.error(f"Failed to load resume file {resume_path}: {e}")
        else:
            logging.error(f"Resume file {resume_path} does not exist.")
            raise FileNotFoundError(f"Resume file {resume_path} does not exist.")
    elif metadata_output_path.exists() or dataset_json_path.exists():
         # Convenience: if output files already exist, load them to avoid duplicates automatically
         auto_resume_path = metadata_output_path if metadata_output_path.exists() else dataset_json_path
         logging.info(f"Output file {auto_resume_path} exists. Automatically resuming from it.")
         try:
            with open(auto_resume_path, 'r') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    has_instruction = any(isinstance(e, dict) and 'instruction' in e for e in existing_data)
                    if has_instruction:
                        results = existing_data
                        initial_results_count = len(results)
                    else:
                        existing_captions = existing_data
                    for entry in existing_data:
                        for key in ("filename", "relative_path", "path", "video", "media_path"):
                            val = entry.get(key)
                            if val:
                                processed_files.add(val)
                                processed_files.add(Path(val).name)
                    logging.info(
                        f"Loaded {len(existing_data)} existing entries; tracked {len(processed_files)} processed identifiers "
                        f"(paths + basenames)."
                    )
         except Exception:
             pass

    # Global retry counter (max 5 failures total)
    max_global_failures = 20
    global_failure_count = 0
    
    # If running in dummy mode, print a summary of the labeling workload and exit early
    if args.dummy:
        require_cnt = 0
        already_cnt = 0
        will_cnt = 0
        for video_path in video_files:
            filename = video_path.name
            # Check if this is a rejected video by path or filename (case-insensitive)
            path_str = str(video_path).lower()
            is_rejected = ('rejected' in path_str) or (re.search(r'\breject\b', video_path.name, re.IGNORECASE) is not None)

            # Respect --label_rejected flag
            if is_rejected and not args.label_rejected:
                continue

            # Get metadata for this video
            metadata = get_metadata_for_video(metadata_map, video_path, metadata_root)
            if metadata is None:
                continue

            primitives = metadata.get('primitives', [])
            if not primitives and not is_rejected:
                continue

            display_filename = metadata.get('relative_path') or filename
            require_cnt += 1
            if display_filename in processed_files or filename in processed_files:
                already_cnt += 1
            else:
                will_cnt += 1

        print(f"Total videos requiring labeling: {require_cnt}")
        print(f"Already labeled: {already_cnt}")
        print(f"Will be labeled (if run normally): {will_cnt}")
        return

    # Process each video
    failures = []
    
    try:
        for video_path in video_files:
            filename = video_path.name
            
            # Check if this is a rejected video by path or filename (case-insensitive)
            path_str = str(video_path).lower()
            is_rejected = ('rejected' in path_str) or (re.search(r'\breject\b', video_path.name, re.IGNORECASE) is not None)

            # If it's a rejected clip and the user did not request labeling rejected clips, skip silently
            if is_rejected and not args.label_rejected:
                logging.info(f"Skipping {filename} (detected as rejected, --label_rejected not set).")
                continue

            # Get metadata for this video
            metadata = get_metadata_for_video(metadata_map, video_path, metadata_root)
            if metadata is None:
                msg = f"No metadata found for {filename}"
                logging.warning(f"{msg}, skipping...")
                failures.append({"filename": filename, "reason": msg})
                continue

            # Use relative_path from metadata if available, otherwise fallback to filename
            display_filename = metadata.get('relative_path') or filename

            # SKIP CHECK: Check if already processed
            if display_filename in processed_files or filename in processed_files:
                logging.info(f"Skipping {display_filename} (already present in results).")
                continue

            primitives = metadata.get('primitives', [])
            if not primitives and not is_rejected:
                msg = f"No primitives found for {display_filename}"
                logging.warning(f"{msg}, skipping...")
                failures.append({"filename": display_filename, "video": metadata.get('video'), "reason": msg})
                continue
            
            # Check if we've hit the global failure limit
            if global_failure_count >= max_global_failures:
                logging.error(f"Reached maximum global failure limit ({max_global_failures}). Stopping processing.")
                break
            
            # Label the video
            include_auxiliary = not is_rejected
            label_result, error_msg = labeler.label_video(video_path, primitives, fps=args.fps, include_auxiliary=include_auxiliary)
            
            if label_result:
                media_path = metadata.get('relative_path') or metadata.get('path') or filename

                result_entry = {
                    'filename': display_filename,
                    'video': metadata.get('video'),
                    'start_frame': metadata.get('start_frame'),
                    'end_frame': metadata.get('end_frame'),
                    'primitives': primitives,
                    'instruction': label_result['instruction'],
                    'justification': label_result['justification'],
                    'metrics': metadata.get('metrics', {}),
                    'media_path': media_path
                }
                results.append(result_entry)
                
                # Update processed set
                processed_files.add(display_filename)
                
                # Save incrementally after each successful label to prevent data loss
                with open(metadata_output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logging.info(f"Progress saved: {len(results)} videos labeled so far")
                
                # Print summary
                print(f"\n{'='*60}")
                print(f"Video: {display_filename}")
                print(f"Primitives: {' -> '.join(primitives)}")
                print(f"Instruction: {label_result['instruction']}")
                print(f"{'='*60}\n")
            else:
                global_failure_count += 1
                failure_entry = {
                    "filename": display_filename,
                    "video": metadata.get('video'),
                    "reason": error_msg or "Unknown error"
                }
                failures.append(failure_entry)
                
                # Save failures incrementally as well
                with open(failure_output_path, 'w') as f:
                    json.dump(failures, f, indent=2)
                    
                logging.warning(f"Failed to label {display_filename} (global failures: {global_failure_count}/{max_global_failures})")
    
    except KeyboardInterrupt:
        logging.info("\nProcessing interrupted by user (Ctrl+C). Saving current results and failures...")
        
    finally:
        # Final save of results
        if results:
            with open(metadata_output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Final metadata saved to: {metadata_output_path}")

        # Save dataset.json (caption + media_path)
        dataset_entries = []
        seen_media = set()

        # Include any existing caption-only entries from resume
        for entry in existing_captions:
            caption = entry.get('caption')
            media_path = entry.get('media_path')
            if caption and media_path and media_path not in seen_media:
                dataset_entries.append({
                    "caption": caption,
                    "media_path": media_path
                })
                seen_media.add(media_path)

        # Add captions from newly labeled results
        for entry in results:
            caption = entry.get('instruction')
            media_path = entry.get('media_path') or entry.get('filename')
            if caption and media_path and media_path not in seen_media:
                dataset_entries.append({
                    "caption": caption,
                    "media_path": media_path
                })
                seen_media.add(media_path)

        if dataset_entries:
            with open(dataset_json_path, 'w') as f:
                json.dump(dataset_entries, f, indent=2)
            logging.info(f"Dataset saved to: {dataset_json_path}")
            
        # Final save of failures
        if failures:
            with open(failure_output_path, 'w') as f:
                json.dump(failures, f, indent=2)
            logging.info(f"Failures saved to: {failure_output_path}")

    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos processed (new + skipped): {len(video_files)}")
    newly_labeled = max(0, len(results) - initial_results_count)
    print(f"Actually labeled (new): {newly_labeled}")
    print(f"Total in output: {len(results)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print(f"See failure details in: {failure_output_path}")
    
    # Results are already saved, just report path
    if len(results) > 0:
        print(f"Metadata Output: {metadata_output_path}")
    if len(existing_captions) > 0 or len(results) > 0:
        print(f"Dataset Output: {dataset_json_path}")
    else:
        logging.warning("No videos were successfully labeled.")
        print(f"Output: None (no successful labels)")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
