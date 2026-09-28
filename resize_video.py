import cv2
import os
import argparse
from pathlib import Path

def resize_video(input_path, output_path, target_width=480, target_height=256):
    """
    Resize a video to the specified resolution and save it.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        target_width (int): Target width in pixels
        target_height (int): Target height in pixels
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return False
    
    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Original video: {original_width}x{original_height} at {original_fps} FPS")
    print(f"Target resolution: {target_width}x{target_height}")
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (target_width, target_height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file {output_path}")
        cap.release()
        return False
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # Write resized frame
        out.write(resized_frame)
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Successfully processed {frame_count} frames")
    print(f"Output saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Resize video(s) to specific resolution")
    parser.add_argument("--input", "-i", type=str, 
                        help="Path to input video file (mutually exclusive with --input_folder)")
    parser.add_argument("--input_folder", "-f", type=str,
                        help="Path to input folder containing videos (mutually exclusive with --input)")
    parser.add_argument("--output_folder", "-o", type=str, default="processed_videos",
                        help="Output folder (default: processed_videos)")
    parser.add_argument("--width", "-w", type=int, default=480,
                        help="Target width in pixels (default: 480)")
    parser.add_argument("--height", "-H", type=int, default=256,
                        help="Target height in pixels (default: 256)")
    parser.add_argument("--extensions", type=str, default="mp4,avi,mov,MOV,mkv,webm",
                        help="Comma-separated list of video extensions to process (default: mp4,avi,mov,MOV,mkv,webm)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.input_folder:
        parser.error("Either --input or --input_folder must be specified")
    if args.input and args.input_folder:
        parser.error("--input and --input_folder are mutually exclusive")
    
    # Parse extensions
    video_extensions = set(f".{ext.strip().lower()}" for ext in args.extensions.split(","))
    
    # Create output folder if it doesn't exist
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine input files
    if args.input:
        # Single file processing
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file does not exist: {input_path}")
            exit(1)
        input_files = [input_path]
    else:
        # Batch processing
        input_folder = Path(args.input_folder)
        if not input_folder.exists():
            print(f"Error: Input folder does not exist: {input_folder}")
            exit(1)
        
        input_files = []
        for ext in video_extensions:
            # Check for both lowercase and uppercase variants
            input_files.extend(input_folder.glob(f"*{ext}"))
            input_files.extend(input_folder.glob(f"*{ext.upper()}"))
        
        # Remove duplicates
        input_files = list(set(input_files))
        
        if not input_files:
            print(f"No video files found in {input_folder} with extensions: {', '.join(video_extensions)}")
            exit(1)
        
        print(f"Found {len(input_files)} video file(s) to process:")
        for video_file in input_files:
            print(f"  - {video_file.name}")
    
    # Process each video
    processed_count = 0
    failed_count = 0
    
    for input_path in input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {input_path.name}")
        print(f"{'='*60}")
        
        # Generate output filename
        output_filename = f"{input_path.stem}_resized_{args.width}x{args.height}.mp4"
        output_path = output_folder / output_filename
        
        # Process the video
        success = resize_video(str(input_path), str(output_path), args.width, args.height)
        
        if success:
            processed_count += 1
            print(f"✓ Successfully processed: {input_path.name}")
        else:
            failed_count += 1
            print(f"✗ Failed to process: {input_path.name}")
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files found: {len(input_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Output folder: {output_folder}")
    
    if failed_count > 0:
        print(f"\nSome files failed to process. Check the output above for details.")
        exit(1)
    else:
        print(f"\nAll videos processed successfully!")

if __name__ == "__main__":
    main()