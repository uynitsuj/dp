import h5py 
import mediapy
from glob import glob
import os
from tqdm import tqdm
import zarr
import imageio

if __name__ == "__main__":
    root_dir = "/home/mfu/dataset/dp_gs/transfer_tiger_241204"
    h5_files = glob("**/*.h5", root_dir=root_dir, recursive=True)
    h5_files = [os.path.join(root_dir, h5_file) for h5_file in h5_files]
    
    action_key = "action/cartesian_pose"
    proprio_key = "state/cartesian/cartesian_pose"
    joint_key = "state/joint/joint_angle_rad"
    for h5_file in tqdm(h5_files):
        output_action_path = h5_file.replace(".h5", "_action.zarr")
        output_proprio_path = h5_file.replace(".h5", "_proprio.zarr")
        output_joint_path = h5_file.replace(".h5", "_joint.zarr")
        with h5py.File(h5_file, "r") as f:
            actions = f[action_key][:]
            proprio = f[proprio_key][:]
            joints = f[joint_key][:]
            zarr.save_array(output_action_path, actions)
            zarr.save_array(output_proprio_path, proprio)
            zarr.save_array(output_joint_path, joints)

    # # save to mp4
    # for h5_file in tqdm(h5_files):
    #     output_left_mp4_path = h5_file.replace(".h5", "_left.mp4")
    #     output_right_mp4_path = h5_file.replace(".h5", "_right.mp4")

    #     with h5py.File(h5_file, "r") as f:
    #         for key, out_path in zip(f['observation'].keys(), [output_left_mp4_path, output_right_mp4_path]):
    #             camera_key = f'observation/{key}/image/camera_rgb'
    #             images = f[camera_key][:] # T, H, W, 3 uint 8
    #             mediapy.write_video(out_path, images)
    
    # save to jpg
    for h5_file in tqdm(h5_files):
        output_left_folder_path = h5_file.replace(".h5", "_left")
        output_right_folder_path = h5_file.replace(".h5", "_right")

        # make folders 
        print("output_left_folder_path", output_left_folder_path)
        print("output_right_folder_path", output_right_folder_path)
        os.makedirs(output_left_folder_path, exist_ok=True)
        os.makedirs(output_right_folder_path, exist_ok=True)

        with h5py.File(h5_file, "r") as f:
            for key, out_path in zip(f['observation'].keys(), [output_left_folder_path, output_right_folder_path]):
                camera_key = f'observation/{key}/image/camera_rgb'
                images = f[camera_key][:]
                for i, img in enumerate(images):
                    imageio.imwrite(os.path.join(out_path, f"{i:04d}.jpg"), img)
