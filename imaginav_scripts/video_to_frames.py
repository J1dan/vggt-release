import cv2
import os
import argparse

def video_to_frames(video_path, fps, output_dir):
    # Extract the video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create the output directory for the frames
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get the original frame rate of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / fps)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every nth frame based on the desired sampling rate
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(video_output_dir, f"frame_{saved_frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Frames saved to {video_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video at a specified sampling rate.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--fps", type=float, required=True, help="Sampling rate in frames per second.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted frames.")

    args = parser.parse_args()

    video_to_frames(args.video_path, args.fps, args.output_dir)