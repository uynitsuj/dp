import os 
import h5py
import torch 
import numpy as np 
from .utils import quat_to_rot_6d, quat_to_euler, euler_to_quat, convert_multi_step_np, convert_delta_action, scale_action
import torchvision.transforms as transforms
from dp_gs.util.args import DatasetConfig, SharedConfig, LoggingConfig
from glob import glob
import json 
from tqdm import tqdm
import zarr 
import line_profiler
import imageio.v3 as iio
from tqdm import tqdm 

class SequenceDataset(torch.utils.data.Dataset):
    action_key : str = "action/cartesian_pose" # [LEFT ARM] w, x, y, z, -- x, y, z + [RIGHT ARM] w, x, y, z -- x, y, z
    proprio_key : str = "state/cartesian/cartesian_pose" # [LEFT ARM] w, x, y, z, -- x, y, z + [RIGHT ARM] w, x, y, z -- x, y, z
    # camera_key : str = "observation/camera/image/camera_rgb" # (n, 480, 848, 3) dtype('uint8') # updates by reading h5 file
    # b'yumi_joint_1_r', b'yumi_joint_2_r', b'yumi_joint_7_r',
    # b'yumi_joint_3_r', b'yumi_joint_4_r', b'yumi_joint_5_r',
    # b'yumi_joint_6_r', b'yumi_joint_1_l', b'yumi_joint_2_l',
    # b'yumi_joint_7_l', b'yumi_joint_3_l', b'yumi_joint_4_l',
    # b'yumi_joint_5_l', b'yumi_joint_6_l', b'gripper_r_joint',
    # b'gripper_l_joint'
    joint_key : str = "state/joint/joint_angle_rad" # for getting gripper information # (n, 16) dtype('float64')
    traj_start_idx : int = 0
    gripper_width :  int = 0.0226
    scale_left_gripper : float = 1
    scale_right_gripper : float = 1

    def __init__(
        self, 
        dataset_config : DatasetConfig,
        shared_config : SharedConfig,
        logging_config : LoggingConfig,
        vision_transform : transforms.Compose,
        split : str = "train",
        debug : bool = False,
    ):
        self.seq_length = shared_config.seq_length
        self.num_pred_steps = shared_config.num_pred_steps
        self.dataset_root = dataset_config.dataset_root
        self.subsample_steps = dataset_config.subsample_steps
        self.num_cameras = shared_config.num_cameras
        assert os.path.exists(self.dataset_root), f"Dataset root {self.dataset_root} does not exist"
        self.vision_transform = vision_transform
        
        self.use_delta_action = shared_config.use_delta_action
        self.proprio_noise = dataset_config.proprio_noise

        # calculate length of dataset 
        self.h5_files = glob("**/*.h5", root_dir=self.dataset_root, recursive=True)
        self.h5_files = [os.path.join(self.dataset_root, h5_file) for h5_file in self.h5_files]

        # update camera keys 
        with h5py.File(self.h5_files[0], 'r') as f:
            self.camera_keys = [f'observation/{key}/image/camera_rgb' for key in f['observation'].keys()]

        assert len(self.camera_keys) >= self.num_cameras, f"Number of cameras {len(self.camera_keys)} is less than required {self.num_cameras}"
        self.camera_keys = self.camera_keys[:self.num_cameras]
        print("Using camera keys: ", self.camera_keys)

        self.file2length = {}
        for h5_file in self.h5_files:
            self.file2length[h5_file] = self.get_traj_length(h5_file) - 1 # subtract 1 as we need to predict the next action

        # load action statistics
        if dataset_config.action_statistics is not None:
            assert os.path.exists(dataset_config.action_statistics), f"Action statistics file {dataset_config.action_statistics} does not exist"
            with open(dataset_config.action_statistics, 'r') as f:
                action_stats = json.load(f)
            action_shape = action_stats["shape"]
            min_action = np.array(action_stats["min_action"]).reshape(action_shape)
            max_action = np.array(action_stats["max_action"]).reshape(action_shape)
        else:
            output_dir = logging_config.output_dir
            min_action, max_action = self.calculate_dataset_statistics(os.path.join(output_dir, "action_statistics.json"))
        self.stats = {
            "min" : torch.from_numpy(min_action), 
            "max" : torch.from_numpy(max_action),
        }

        # randomly shuffle the file2length dataset
        rng = np.random.default_rng(seed=shared_config.seed)
        keys = list(self.file2length.keys())
        rng.shuffle(keys)

        # train test split
        if split == "train":
            keys = keys[:int(len(keys) * dataset_config.train_split)]
        else:
            keys = keys[int(len(keys) * dataset_config.train_split):]
        self.file2length = {k : self.file2length[k] for k in keys}

        # index to start end 
        self.start_end = []
        self.debug = debug
        for h5_file, length in self.file2length.items():
            # for debugging, we visualize the entire trajecotry 
            if debug:
                self.start_end.append((h5_file, 0, length - self.subsample_steps))
            else:
                for i in range(length - (self.subsample_steps * (self.seq_length + self.num_pred_steps) - 1)):
                    self.start_end.append((h5_file, i, i + self.subsample_steps * (self.seq_length + self.num_pred_steps)))

        self.total_length = len(self.start_end)

        # vision augmentation
        if dataset_config.vision_aug:
            self.vision_aug = True
            self.contrast_range = [0.8, 1.2]
            self.brightness_range = [-0.1, 0.1]
            print("using numeric brightness and contrast augmentation")
            print("contrast range: ", self.contrast_range)
            print("brightness range: ", self.brightness_range)
        else:
            self.vision_aug = False

        # # precache the vr objects
        # self.cached_vr_objects = {}
        # print("caching h5 file objects")
        # for h5_file_path in tqdm(self.h5_files):
        #     mp4_paths = [h5_file_path.replace(".h5", "_left.mp4"), h5_file_path.replace(".h5", "_right.mp4")]
        #     for mp4_path in mp4_paths:
        #         self.cached_vr_objects[mp4_path] = VideoReader(mp4_path)

        # scale action 
        self.enable_scale_action = dataset_config.scale_action

    def calculate_dataset_statistics(
        self, 
        output_path : str = "config/action_statistics.json"
    ):
        # calculate the min and max of delta actions for left and right arm 
        global_min_action, global_max_action = None, None
        good_h5_files = []
        for h5_file in tqdm(self.h5_files):
            # with h5py.File(h5_file, 'r') as f:
            left_action, right_action = self.helper_load_action(h5_file, self.traj_start_idx, self.get_traj_length(h5_file) - self.subsample_steps)
            left_proprio, right_proprio = self.helper_load_proprio(h5_file, self.traj_start_idx, self.get_traj_length(h5_file) - self.subsample_steps, False)
            left_action = convert_multi_step_np(left_action, self.num_pred_steps)
            right_action = convert_multi_step_np(right_action, self.num_pred_steps)
            delta_left_action = convert_delta_action(left_action, left_proprio)
            delta_right_action = convert_delta_action(right_action, right_proprio)
            # if np.any(np.abs(delta_right_action[:, :, :3]) > 0.01): # TODO remove hack! the tiger pick dataset only contains good data
            #     continue
            # if np.any(np.abs(delta_left_action[:, 0, :3]) > 0.03): # velocity is too high, not good for normalization.
            #     continue
            good_h5_files.append(h5_file)
            if self.use_delta_action:
                action = np.concatenate([delta_left_action, delta_right_action], axis=-1)
            else:
                action = np.concatenate([left_action, right_action], axis=-1)

            min_action, max_action = action.min((0,1)), action.max((0,1)) # only calculate on the action dim
            if global_min_action is None:
                global_min_action = min_action
                global_max_action = max_action
            else:
                global_min_action = np.stack([global_min_action, min_action], axis=0).min(0)
                global_max_action = np.stack([global_max_action, max_action], axis=0).max(0)    
        self.h5_files = good_h5_files
        
        if np.all(global_max_action[..., 9] <= 0.01):
            self.scale_left_gripper = self.gripper_width / global_max_action[..., 9]
            global_max_action[..., 9] = self.gripper_width
        else:
            self.scale_left_gripper = 1
        
        if np.all(global_max_action[..., 19] <= 0.01):
            self.scale_right_gripper = self.gripper_width / global_max_action[..., 19]
            global_max_action[..., 19] = self.gripper_width
        else:
            self.scale_right_gripper = 1
        # save the statistics 
        stats = {
            "shape" : global_min_action.shape,
            "min_action": global_min_action.flatten().tolist(),
            "max_action": global_max_action.flatten().tolist()
        }
        with open(output_path, 'w') as f:
            json.dump(stats, f)
        print("Action statistics saved to ", output_path)
        return global_min_action, global_max_action
        

    def __len__(self):
        return self.total_length

    def get_h5_length(
        self, 
        h5_file : h5py.File,
    ): 
        return len(h5_file[self.action_key][:])

    def get_traj_length(
        self, 
        h5_file_path : str,
    ): 
        action_fp = h5_file_path.replace(".h5", "_proprio.zarr")
        actions = zarr.load(action_fp)
        return len(actions)
    
    @line_profiler.profile
    def __getitem__(self, idx : int):
        h5_file, start, end = self.start_end[idx]
        left_action, right_action = self.helper_load_action(h5_file, start, end)
        left_proprio, right_proprio = self.helper_load_proprio(h5_file, start, end)

        left_action = convert_multi_step_np(left_action, self.num_pred_steps)
        right_action = convert_multi_step_np(right_action, self.num_pred_steps)

        proprio = np.concatenate([left_proprio, right_proprio], axis=-1) 
        proprio = torch.from_numpy(proprio)

        # get delta actions 
        if self.use_delta_action:
            left_action = convert_delta_action(left_action, left_proprio)
            right_action = convert_delta_action(right_action, right_proprio)
        left_action = torch.from_numpy(left_action)
        right_action = torch.from_numpy(right_action)

        # concatenate actions 
        actions = torch.concatenate([left_action, right_action], dim=-1)

        # option for not scaling 
        if self.enable_scale_action:
            actions = scale_action(actions, self.stats, type="diffusion") 

        # remove nans 
        actions = torch.nan_to_num(actions)
            
        # get camera 
        camera = self.helper_load_camera(h5_file, start, end)

        # trim on the time dimension
        if not self.debug:
            actions = actions[:self.seq_length]
            proprio = proprio[:self.seq_length]
            camera = camera[:self.seq_length]
        return {
            "action" : actions.float(), # seq_length, num_pred_steps, 20
            "proprio" : proprio.float(), # seq_length, 20
            "observation" : camera.float(), # seq_length, 1, 3, 224, 224
        }
        
    def helper_load_action(self, h5_file_path : str, start : int, end : int):
        
        # get action path, joint path, and retrieved indices
        indices = np.arange(start + self.subsample_steps, end + self.subsample_steps, self.subsample_steps)
        action_fp = h5_file_path.replace(".h5", "_proprio.zarr")
        joint_fp = h5_file_path.replace(".h5", "_joint.zarr")

        # get actions
        actions = zarr.load(action_fp)[indices]
        left, right = actions[:, :7], actions[:, 7:]
        
        # get gripper
        joint_data = zarr.load(joint_fp)[indices]
        left_g = joint_data[:, -1][:, None]
        right_g = joint_data[:, -2][:, None]

        left = np.concatenate([
            left[:, 4:],
            quat_to_rot_6d(left[:, :4]), 
            left_g * self.scale_left_gripper
        ], axis=1)
        right = np.concatenate([
            right[:, 4:],
            quat_to_rot_6d(right[:, :4]), 
            right_g * self.scale_right_gripper
        ], axis=1)
        return left, right

    def randomize(self, transform : np.ndarray):
        # randomize the transform (N, 7) -> (N, 7)
        # each transform is wxyz, xyz
        t, rot = transform[:, 4:7], transform[:, :4]
        t += np.random.uniform(0, self.proprio_noise, t.shape)
        rot = quat_to_euler(rot)
        rot += np.random.uniform(0, self.proprio_noise, rot.shape)
        rot = euler_to_quat(rot)
        rt = np.concatenate([rot, t], axis=1)
        return rt

    def helper_load_proprio(self, h5_file_path : str, start : int, end : int, noisy : bool = True):
        indices = np.arange(start, end, self.subsample_steps)
        proprio_fp = h5_file_path.replace(".h5", "_proprio.zarr")
        joint_fp = h5_file_path.replace(".h5", "_joint.zarr")
        
        # get proprio data
        proprio = zarr.load(proprio_fp)[indices]
        left, right = proprio[:, :7], proprio[:, 7:]

        # get gripper data
        joint_data = zarr.load(joint_fp)[indices]
        left_g, right_g = joint_data[:, -1][:, None], joint_data[:, -2][:, None]
        
        # add proprio noise 
        if noisy:
            left = self.randomize(left)
            right = self.randomize(right)

        left = np.concatenate([
            left[:, 4:],
            quat_to_rot_6d(left[:, :4]), 
            left_g * self.scale_left_gripper
        ], axis=1)
        right = np.concatenate([
            right[:, 4:],
            quat_to_rot_6d(right[:, :4]), 
            right_g * self.scale_right_gripper
        ], axis=1)
        return left, right

    @line_profiler.profile
    def helper_load_camera(self, h5_file_path : str, start : int, end : int):
        mp4_paths = [h5_file_path.replace(".h5", "_left.mp4"), h5_file_path.replace(".h5", "_right.mp4")]
        indices = np.arange(start, end - self.num_pred_steps * self.subsample_steps, self.subsample_steps)
        camera_observations = []
        
        for camera_name, mp4_path in zip(self.camera_keys, mp4_paths):
            with iio.imopen(mp4_path, "r", plugin="pyav") as file:
                camera = [file.read(index=ind, constant_framerate=True) for ind in indices]
            camera = np.stack(camera)
            # vr = VideoReader(mp4_path)
            # vr = self.cached_vr_objects[mp4_path]
            # camera = vr.get_batch(indices)
            # camera = camera.asnumpy()
            if camera.dtype == np.uint8:
                norm = 255.
            else:
                norm = 1.
            subsequence = torch.from_numpy(camera / norm)

            if self.vision_aug:
                contrast = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
                brightness = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
                subsequence = contrast * subsequence + brightness
            
            camera_observations.append(subsequence)

        subsequence = torch.stack(camera_observations) # num_camera, T, H, W, C
        subsequence = subsequence.permute(1, 0, 4, 2, 3).contiguous()  # T, num_camera, C, H, W

        T, num_camera, C, H, W = subsequence.shape
        subsequence = subsequence.view(T * num_camera, C, H, W) # num_camera * T, C, H, W
        subsequence = self.vision_transform(subsequence).float()
        subsequence = subsequence.view(T, num_camera, *subsequence.shape[1:])
        return subsequence