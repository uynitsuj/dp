"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import os 
# os.environ["HF_LEROBOT_HOME"] = "/shared/projects/icrl/data/dpgs/lerobot"
import shutil
import h5py 
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm, trange
import zarr
from PIL import Image
from openpi_client.image_tools import resize_with_pad
from glob import glob

RAW_DATASET_FOLDERS = [
    # "/shared/projects/dpgs_dataset/yumi_coffee_maker/real_data/yumi_mug_to_coffee_maker_041525_2142"
    # "/shared/projects/dpgs_dataset/yumi_drawer_open/real_data/yumi_drawer_open_041525_2142"
    # "/shared/projects/dpgs_dataset/yumi_faucet/real_data",
    # "/shared/projects/dpgs_dataset/yumi_pick_tiger_r2r2r/real_data",
    "/shared/projects/dpgs_dataset/yumi_cardboard_lift/real_data"
    # "/shared/projects/dpgs_dataset/yumi_pick_tiger_r2r2r/yumi_tiger_pick_right_arm_250424_2146",
    # "/shared/projects/dpgs_dataset/yumi_pick_tiger_r2r2r/yumi_tiger_pick_left_arm_250424_2108"
    # "/shared/projects/dpgs_dataset/yumi_pick_tiger_r2r2r/left_right_combined_250424"
]
LANGUAGE_INSTRUCTIONS = [
    # "put the white cup on the coffee machine"
    # "open the drawer"
    # "turn off the faucet"
    # "pick up the tiger"
    "pick up the cardboard box"
    # "pick up the tiger with the right arm"
    # "pick up the tiger with the left arm"
    # "pick up the tiger"
]
# REPO_NAME = "mlfu7/dpgs_real_faucet_150"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_real_tiger_150"  # Name of the output dataset, also used for the Hugging Face Hub
REPO_NAME = "mlfu7/dpgs_real_cardboard_lift_150"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_real_tiger_pick_right_arm_80"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_real_tiger_pick_left_arm_80"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_real_tiger_pick_combined_160"  # Name of the output dataset, also used for the Hugging Face Hub

CAMERA_KEYS = [
    "_left", 
    "_right"
] # folder of rgb images
CAMERA_KEY_MAPPING = {
    "_left": "exterior_image_1_left",
    "_right": "exterior_image_2_left",
}
STATE_KEY = "_joint.zarr"
RESIZE_SIZE = 224

def main():
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    print("Dataset saved to ", output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="yumi",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "video",
                "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "video",
                "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["joint_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["actions"],
            },
        },
        image_writer_threads=20,
        image_writer_processes=10,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name, language_instruction in zip(RAW_DATASET_FOLDERS, LANGUAGE_INSTRUCTIONS):
        # get all the tasks that are collected that day 
        data_day_dir = raw_dataset_name
        print("Processing folder: ", data_day_dir)
        traj_base_names = [i.replace("_joint.zarr", "") for i in glob(f"{data_day_dir}/*_joint.zarr")]
        for idx, task in enumerate(traj_base_names):
            print(f"Trajectory {idx}/{len(traj_base_names)}: {task} is being processed")
            proprio_data = zarr.load(task + STATE_KEY)
            seq_length = proprio_data.shape[0] - 1 # remove the last proprio state since we need to calculate the action
            images = {
                key : [
                    os.path.join(task + key, i) for i in sorted(os.listdir(task + key))
                ] for key in CAMERA_KEYS
            }
            images_per_step = [
                {key : images[key][i] for key in CAMERA_KEYS} for i in range(seq_length) 
            ]
        
            for step in range(seq_length):
                # load proprio data
                proprio_t = proprio_data[step]
                # create delta action
                action_t = proprio_data[step + 1] - proprio_t
                # change the gripper to absolute 
                action_t[-2:] = proprio_data[step + 1][-2:]
                # get the images for this step
                images_t = {
                    CAMERA_KEY_MAPPING[key]: resize_with_pad(
                        np.array(Image.open(images_per_step[step][key])),
                        RESIZE_SIZE,
                        RESIZE_SIZE
                    ) for key in CAMERA_KEYS
                }
                dataset.add_frame(
                    {
                        "joint_position": proprio_t,
                        "actions": action_t,
                        **images_t
                    }
                )
            dataset.save_episode(task=language_instruction)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    print("Dataset saved to ", output_path)

    # # Optionally push to the Hugging Face Hub
    # dataset.push_to_hub(
    #     tags=["otter", "franka", "pi_0", "multitask"],
    #     private=True,
    #     push_videos=True,
    #     license="apache-2.0",
    # )


if __name__ == "__main__":
    main()