import cv2
import os
import re
import numpy as np
from glob import glob

def get_sorted_frames(folder_path):
    """
    Retrieves all image file paths from a folder and sorts them numerically
    based on the digits in their filenames. Handles common image extensions.
    """
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    all_files = []
    
    # Use glob to find files matching common image extensions
    for ext in image_extensions:
        all_files.extend(glob(os.path.join(folder_path, ext)))

    # Function to extract the numerical part of the filename for sorting
    def extract_number(filepath):
        # Look for the last sequence of digits in the filename
        filename = os.path.basename(filepath)
        # Finds the first contiguous sequence of digits
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(0))
        return float('inf') # Ensures files without numbers are sorted last

    # Sort files using the extracted number
    all_files.sort(key=extract_number)
    return all_files

def create_comparison_video(folder1_path, folder2_path, output_video_path, fps=10, target_height=480):
    """
    Creates a side-by-side video comparing frames from two folders.
    The speed is synchronized so both videos have the same duration,
    by stretching the shorter sequence to match the longer one.

    :param folder1_path: Path to the first folder (will be on the left).
    :param folder2_path: Path to the second folder (will be on the right).
    :param output_video_path: Path to save the final MP4 video.
    :param fps: Frames per second for the output video (e.g., 10).
    :param target_height: The height (in pixels) to which all frames will be resized.
                          Since FOLDER_2 is cropped to a square, both panels will 
                          be resized to (target_height x target_height).
    """
    # 1. Get and sort frame paths
    print(f"Reading frames from: {folder1_path}")
    paths1 = get_sorted_frames(folder1_path)
    print(f"Reading frames from: {folder2_path}")
    paths2 = get_sorted_frames(folder2_path)
    
    N1 = len(paths1)
    N2 = len(paths2)

    if N1 == 0 or N2 == 0:
        print("Error: One or both folders contain no images after sorting.")
        return

    # Determine the total number of frames in the output video (N_max determines the duration)
    N_max = max(N1, N2)
    print(f"\nFolder 1 frames: {N1}")
    print(f"Folder 2 frames: {N2}")
    print(f"Output duration will be based on {N_max} frames at {fps} FPS.")
    
    # 2. Determine target dimensions for panels and video
    
    # Since frames from FOLDER_2 are explicitly cropped to a square (512x512), 
    # we will treat both output panels as squares of size target_height x target_height.
    target_panel_width = target_height
    
    video_width = target_panel_width * 2
    video_height = target_height
    
    print(f"Panel size (W x H) will be {target_panel_width}x{target_height} for each side.")
    print(f"Output video size (W x H) is {video_width}x{video_height}.")

    # 3. Initialize VideoWriter
    # Define the codec and create VideoWriter object (e.g., MP4V for MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        return

    # 4. Process and write frames
    for i in range(N_max):
        # Synchronization Logic: Scale the output index (i) to the folder index (j)
        j1 = min(int(i * N1 / N_max), N1 - 1)
        j2 = min(int(i * N2 / N_max), N2 - 1)

        # Load frames using the calculated indices
        frame1 = cv2.imread(paths1[j1])
        frame2 = cv2.imread(paths2[j2])

        if frame1 is None or frame2 is None:
            print(f"Warning: Frame missing at index {j1} or {j2}. Skipping frame {i}.")
            continue
            
        # --- Apply Cropping for FOLDER 2 frames ---
        # The user requested cropping 1024x512 frames to the left 512x512 half.
        if frame2.shape[1] == 1024 and frame2.shape[0] == 512:
            # Crop to the left 512 pixels: [All Rows, Columns 0 through 511]
            frame2 = frame2[:, :512]
            
        # Resize frames: Both are resized to the square panel size (target_height x target_height)
        # Note: This will resize both frames to the same square dimension.
        # If frame1 is not a square, it will be stretched/compressed to fit.
        resized_frame1 = cv2.resize(frame1, (target_panel_width, target_height))
        resized_frame2 = cv2.resize(frame2, (target_panel_width, target_height))

        # Concatenate horizontally (frame1 on left, frame2 on right)
        comparison_frame = np.concatenate((resized_frame1, resized_frame2), axis=1)

        # Write the combined frame to the video file
        out.write(comparison_frame)
        
        # Optional: Print progress
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{N_max} frames...")


    # 5. Cleanup
    out.release()
    print(f"\nSuccessfully created comparison video at: {output_video_path}")
    print(f"Total frames written: {N_max}")


# --- Configuration ---
# IMPORTANT: Update these paths to match your folder locations and desired output.

# Folder containing frames for the left side (VGG-T)
FOLDER_1 = "/media/jc/data/jc_ws/dev/vggt/frames/navigate_to_bed" 

# Folder containing frames for the right side (InternNav)
# Frames from this folder will be cropped to the left 512x512 half if they are 1024x512.
FOLDER_2 = "/media/jc/data/jc_ws/dev/InternNav/manual_logs/manual_control_eval/video/5593_571/frames" 

# Output video file path
# Changed name to indicate cropping
OUTPUT_FILE = "comparison_video_cropped.mp4" 

# Desired video frame rate (FPS) and resolution height
FRAME_RATE = 10 
HEIGHT = 480 # The output height for each side (total video height)
            # Both panels will be 480x480 (square)

if __name__ == '__main__':
    create_comparison_video(
        folder1_path=FOLDER_1,
        folder2_path=FOLDER_2,
        output_video_path=OUTPUT_FILE,
        fps=FRAME_RATE,
        target_height=HEIGHT
    )
