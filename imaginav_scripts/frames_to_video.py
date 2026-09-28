import cv2
import os
from pathlib import Path
import argparse

def sorted_image_list(folder: Path, exts=(".png", ".jpg", ".jpeg")):
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    return sorted(imgs, key=lambda p: int(''.join(filter(str.isdigit, p.stem))))

def frames_to_video(folder_path: Path, output_video_path: str, fps: float = 24, crop_square: bool = False):
    images = sorted_image_list(folder_path)
    if not images:
        print(f"No images found in {folder_path}")
        return

    # Read first image to get dimensions
    first_img = cv2.imread(str(images[0]))
    height, width, layers = first_img.shape

    # Apply cropping if requested
    if crop_square:
        # Cut off the rightmost square region of size height x height
        crop_width = width - height
        crop_height = height
        print(f"Cropping from {width}x{height} to {crop_width}x{crop_height} (cutting off right {height}x{height} square)")
    else:
        crop_width = width
        crop_height = height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (crop_width, crop_height))

    for img_path in images:
        img = cv2.imread(str(img_path))
        if crop_square:
            # Cut off the rightmost square region of size height x height
            img = img[0:crop_height, 0:crop_width]
        video.write(img)

    video.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert frames to video.")
    parser.add_argument("--folder", type=str, required=True, help="Folder name inside frames/ containing the image files.")
    parser.add_argument("--fps", type=float, default=2, help="Frames per second for the output video.")
    parser.add_argument("--crop_square", action="store_true", help="Cut off the rightmost square region of size height x height, resulting in (width-height) x height video.")

    args = parser.parse_args()

    frames_dir = Path("frames") / args.folder
    output_video = frames_dir / "video.mp4"

    frames_to_video(frames_dir, str(output_video), args.fps, args.crop_square)