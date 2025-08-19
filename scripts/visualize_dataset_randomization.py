# python script/visualize_dataset_randomization.py --root-dir /home/yujustin/dataset/dp_gs/sim_coffee_maker/successes_033125_1619
"""
root_dir
    - env_15_2025_03_31_15_15_00
        - camera_0
            - rgb 
                - 0000.jpg
        - camera_1
        - robot_data

Create a visualization script that will take in the root_dir and 
combines first image from each trajectory with name env_* into a video at 30fps so that we can see the randomization
"""

import os
import glob
import cv2
import tyro
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class VisualizationArgs:
    """Arguments for dataset randomization visualization."""
    
    root_dir: str
    """Path to the root directory containing environment trajectories."""
    
    output_path: str = "randomization_visualization.mp4"
    """Path to save the output video."""
    
    fps: int = 30
    """Frames per second for the output video."""
    
    camera_id: int = 0
    """Camera ID to use for visualization (e.g., 0 for camera_0)."""


def main(args: VisualizationArgs) -> None:
    """Create a video visualizing the randomization across environment trajectories."""
    
    # Find all environment directories
    env_dirs = sorted(glob.glob(os.path.join(args.root_dir, "env_*")))
    
    if not env_dirs:
        print(f"No environment directories found in {args.root_dir}")
        return
    
    # Collect first images from each environment
    images = []
    for env_dir in tqdm(env_dirs, desc="Processing environments"):
        rgb_dir = os.path.join(env_dir, f"camera_{args.camera_id}", "rgb")
        if not os.path.exists(rgb_dir):
            print(f"RGB directory not found in {env_dir}")
            continue
            
        image_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
        if not image_files:
            print(f"No images found in {rgb_dir}")
            continue
            
        # Get the first image
        img = cv2.imread(image_files[0])
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to read image: {image_files[0]}")
    
    if not images:
        print("No valid images found")
        return
    
    # Create video writer
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    video_writer = cv2.VideoWriter(
        args.output_path,
        fourcc,
        args.fps,
        (width, height),
        isColor=True
    )
    
    # Write images to video
    for img in tqdm(images, desc="Writing video frames"):
        video_writer.write(img)
    
    video_writer.release()
    print(f"Video saved to {args.output_path}")


if __name__ == "__main__":
    main(tyro.cli(VisualizationArgs))