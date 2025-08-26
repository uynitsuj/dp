import json
import os
from glob import glob
from typing import Iterator, List, Optional

import h5py

# import line_profiler
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoProcessor

from dp.util.args import DatasetConfig, LoggingConfig, SharedConfig

from .utils import convert_multi_step_np, scale_action


class CollateFunction:
    def __init__(
        self, 
        sequence_length : int, 
        tokenizer : Optional[AutoProcessor] = None, 
        max_token_length : int = 108,
        pred_left_only : bool = False,
        pred_right_only : bool = False,
    ):
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length    
        self.pred_left_only = pred_left_only
        self.pred_right_only = pred_right_only

    def __call__(self, batch):
        return collate_fn_lambda(
            batch, 
            sequence_length=self.sequence_length, 
            tokenizer=self.tokenizer, 
            max_token_length=self.max_token_length,
            pred_left_only=self.pred_left_only,
            pred_right_only=self.pred_right_only,
        )

class VideoSampler(Sampler):
    """
    Sampler for the sequence dataset with epoch-based shuffling.
    For each instance, it will be of size batch_size * sequence_length,
    then the collate_fn will reshape it to batch_size, sequence_length, ...
    """
    def __init__(
        self, 
        data: Dataset, 
        batch_size: int, 
        sequence_length: int, 
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.num_chunks = len(self.data) // (self.batch_size * self.sequence_length * self.num_replicas)
        self.total_length = self.num_chunks * self.batch_size * self.sequence_length

        self.start_idx = self.rank * self.total_length
        self.end_idx = self.start_idx + self.total_length

    def __len__(self) -> int:
        return self.num_chunks
    
    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        """
        Generate indices for each epoch with proper shuffling.
        Uses both epoch and seed for reproducible but different shuffling per epoch.
        """
        # Create a generator with seed that combines epoch and base seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Generate indices for this process
        indices = torch.arange(start=self.start_idx, end=self.end_idx)
        
        # Shuffle indices while maintaining sequence grouping
        num_batches = len(indices) // (self.batch_size * self.sequence_length)
        indices = indices.view(num_batches, self.batch_size * self.sequence_length)
        
        # Shuffle the batches
        perm = torch.randperm(num_batches, generator=g)
        indices = indices[perm].view(-1)
        
        # Yield batches of indices
        for batch in torch.chunk(indices, self.num_chunks):
            yield batch.tolist()

def collate_fn_lambda(
    batch: List[dict], 
    sequence_length: int, 
    tokenizer: Optional[AutoProcessor] = None,
    max_token_length: int = 108,
    pred_left_only: bool = False,
    pred_right_only: bool = False,
) -> dict:
    """
    Collate function for the sequence dataset.
    The EOS token is included as part of the valid sequence to predict.
    
    Returns:
        dict containing:
            - All original keys from batch
            - If tokenizer is provided:
                - "action": padded discrete actions (B, max_token_length)
                - "action_mask": boolean mask indicating valid tokens including EOS (B, max_token_length)
    """
    # get the keys 
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        data = [sample[key] for sample in batch]
        data = torch.stack(data)
        data = data.view(-1, sequence_length, *data.shape[1:])
        collated[key] = data

    if tokenizer is not None:
        import pdb; pdb.set_trace() # TODO: verify if using
        # Get discrete actions for the last timestep
        raw_action = collated["action"][:, -1]

        if pred_left_only:
            raw_action = raw_action[..., :10] # bruteforce it for now 
        elif pred_right_only:
            raw_action = raw_action[..., 10:] # bruteforce it for now 

        discrete_action = tokenizer(raw_action)  # List[List[int]]
        
        # Create padded action tensor and mask tensor
        eos_idx = tokenizer.vocab_size  # this is the eos token
        batch_size = len(discrete_action)
        padded_action = torch.ones(batch_size, max_token_length, dtype=torch.long) * eos_idx
        action_mask = torch.zeros(batch_size, max_token_length, dtype=torch.bool)
        
        # Fill in actions and masks
        for i, action in enumerate(discrete_action):
            action_length = min(len(action), max_token_length) # threshold the action length
            # Fill in the actual tokens
            padded_action[i, :action_length] = torch.tensor(action[:action_length])
            # Mask includes both the sequence and the EOS token
            action_mask[i, :action_length + 1] = True
        
        collated["action"] = padded_action
        collated["action_mask"] = action_mask

    return collated

class SequenceDataset(Dataset):
    traj_start_idx : int = 0

    def __init__(
        self, 
        dataset_config : DatasetConfig,
        shared_config : SharedConfig,
        logging_config : LoggingConfig,
        vision_transform : transforms.Compose,
        split : str = "train",
        debug : bool = False,
    ):
        # TODO: squash some shared config stuff or move into dataset config
        self.seq_length = shared_config.seq_length
        self.num_pred_steps = shared_config.num_pred_steps
        self.dataset_root = dataset_config.dataset_root
        self.subsample_steps = dataset_config.subsample_steps
        self.num_cameras = shared_config.num_cameras
        self.camera_keys = shared_config.camera_keys
        assert os.path.exists(self.dataset_root), f"Dataset root {self.dataset_root} does not exist"
        self.vision_transform = vision_transform
        self.data_transforms = dataset_config.data_transforms

        # calculate length of dataset 
        common_path = glob("**/episode.h5", root_dir=self.dataset_root, recursive=True)
        self.common_path = [os.path.join(self.dataset_root, p.replace("episode.h5", "")) for p in common_path]

        if dataset_config.data_subsample_num_traj > 0:
            self.common_path = self.common_path[:int(dataset_config.data_subsample_num_traj / dataset_config.train_split)]
        elif dataset_config.data_subsample_ratio > 0:
            self.common_path = self.common_path[:int(dataset_config.data_subsample_ratio * len(self.common_path) / dataset_config.train_split)]
        assert not (dataset_config.data_subsample_num_traj > 0 and dataset_config.data_subsample_ratio > 0), "Both data_subsample_num_traj and data_subsample_ratio are set, please only set one"
        print("Number of selected trajectories: ", len(self.common_path))
        
        self.file2length = {}
        for file in self.common_path:
            self.file2length[file] = self.get_traj_length(file) - 1 # subtract 1 as we need to predict the next action

        # load action statistics
        if dataset_config.action_statistics is not None:
            assert os.path.exists(dataset_config.action_statistics), f"Action statistics file {dataset_config.action_statistics} does not exist"
            with open(dataset_config.action_statistics, 'r') as f:
                action_stats = json.load(f)
            action_shape = action_stats["shape"]
            min_action = np.array(action_stats["min_action"]).reshape(action_shape)
            max_action = np.array(action_stats["max_action"]).reshape(action_shape)
            min_proprio = np.array(action_stats["min_proprio"]).reshape(action_shape)
            max_proprio = np.array(action_stats["max_proprio"]).reshape(action_shape)
            mean_action = np.array(action_stats["mean_action"]).reshape(action_shape)
            mean_proprio = np.array(action_stats["mean_proprio"]).reshape(action_shape)
        else:
            output_dir = logging_config.output_dir
            min_action, max_action, min_proprio, max_proprio, mean_action, mean_proprio = self.calculate_dataset_statistics(os.path.join(output_dir, "action_statistics.json"))
        
        self.stats = {
            "min" : torch.from_numpy(min_action), 
            "max" : torch.from_numpy(max_action),
        }
        self.stats_proprio = {
            "min" : torch.from_numpy(min_proprio), 
            "max" : torch.from_numpy(max_proprio),
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
        for file_path, length in self.file2length.items():
            for i in range(length - (self.subsample_steps * (self.seq_length + self.num_pred_steps) - 1)):
                self.start_end.append((file_path, i, i + self.subsample_steps * (self.seq_length + self.num_pred_steps)))

        rng = np.random.default_rng(seed=shared_config.seed)
        rng.shuffle(self.start_end)

        # modify start end such that it is flattened in sequence length dimension
        new_start_end = []
        for file_path, start, end in self.start_end:
            for i in range(start, end - self.subsample_steps * self.num_pred_steps, self.subsample_steps):
                new_start_end.append((file_path, i, i + self.subsample_steps * self.num_pred_steps))
        self.start_end = new_start_end

        self.total_length = len(self.start_end)

        # vision augmentation
        if dataset_config.vision_aug:
            self.vision_aug = True # TODO find out if this is being used externally, since it's not used in the class
            self.contrast_range = [0.8, 1.2]
            self.brightness_range = [-0.1, 0.1]
            print("using numeric brightness and contrast augmentation")
            print("contrast range: ", self.contrast_range)
            print("brightness range: ", self.brightness_range)
        else:
            self.vision_aug = False

        self.enable_scale_action = dataset_config.scale_action
        self.enable_scale_proprio = dataset_config.scale_proprio


    def calculate_dataset_statistics(
        self, 
        output_path : str = "config/action_statistics.json"
    ):
        # calculate the min and max of delta actions for left and right arm 
        global_min_action, global_max_action = None, None
        global_min_proprio, global_max_proprio = None, None
        good_files = []
        for file in tqdm(self.common_path):
            action, proprio = self.helper_load_episode_data(file, self.traj_start_idx, self.get_traj_length(file) - self.subsample_steps)
            action = convert_multi_step_np(action, self.num_pred_steps)

            data = {
            "action" : action,
            "proprio" : proprio,
            }

            # apply data input transforms here, also make sure appropriate inverse transform is applied at model output at wherever the inference wrapper calls the model (relative, inter-proprio relative, etc)
            for transform in self.data_transforms.inputs:
                data = transform(data)
            
            action = data["action"] # [B, T, action_dim]
            proprio = data["proprio"] # [B, proprio_dim]

            good_files.append(file)

            min_action, max_action = action.min((0,1)), action.max((0,1)) # only calculate on the action dim TODO: also try per-timestep dim normalization
            min_proprio, max_proprio = proprio.min((0)), proprio.max((0))
            mean_action = action.mean((0,1))
            mean_proprio = proprio.mean((0))

            if global_min_action is None:
                global_min_action = min_action
                global_max_action = max_action
            else:
                global_min_action = np.stack([global_min_action, min_action], axis=0).min(0)
                global_max_action = np.stack([global_max_action, max_action], axis=0).max(0)    

            if global_min_proprio is None:
                global_min_proprio = min_proprio
                global_max_proprio = max_proprio
            else:
                global_min_proprio = np.stack([global_min_proprio, min_proprio], axis=0).min(0)
                global_max_proprio = np.stack([global_max_proprio, max_proprio], axis=0).max(0)
        
        self.common_path = good_files
        
        # if np.all(global_max_action[..., 9] <= 0.01):
        #     self.scale_left_gripper = self.gripper_width / global_max_action[..., 9]
        #     global_max_action[..., 9] = self.gripper_width
        # else:
        #     self.scale_left_gripper = 1
        
        # if np.all(global_max_action[..., 19] <= 0.01):
        #     self.scale_right_gripper = self.gripper_width / global_max_action[..., 19]
        #     global_max_action[..., 19] = self.gripper_width
        # else:
        #     self.scale_right_gripper = 1
        # save the statistics 

        stats = {
            "shape" : global_min_action.shape,
            "min_action": global_min_action.flatten().tolist(),
            "max_action": global_max_action.flatten().tolist(),
            "mean_action": mean_action.flatten().tolist(),
            "min_proprio": global_min_proprio.flatten().tolist(),
            "max_proprio": global_max_proprio.flatten().tolist(),
            "mean_proprio": mean_proprio.flatten().tolist(),
        }

        with open(output_path, 'w') as f:
            json.dump(stats, f)
        print("Action statistics saved to ", output_path)
        return global_min_action, global_max_action, global_min_proprio, global_max_proprio, mean_action, mean_proprio

    def __len__(self):
        return self.total_length

    def get_traj_length(
        self, 
        file_path : str,
    ): 
        action_fp = file_path + "episode.h5" 
        actions = h5py.File(action_fp, "r")["action"]
        return len(actions)
    
    # @line_profiler.profile
    def __getitem__(self, idx : int):
        file_path, start, end = self.start_end[idx]
        action, proprio = self.helper_load_episode_data(file_path, start, end)
        
        proprio = torch.from_numpy(proprio[0]) # [proprio_dim]
        # get rid of the time dimension since there's just one step

        action = torch.from_numpy(action)
        action = action.squeeze(0) # [T, action_dim]

        data = {
            "action" : action.unsqueeze(0),
            "proprio" : proprio.unsqueeze(0),
        }

        # apply data input transforms here, also make sure appropriate inverse transform is applied at model output at wherever the inference wrapper calls the model (relative, inter-proprio relative, etc)
        for transform in self.data_transforms.inputs:
            data = transform(data)
        
        action = data["action"].squeeze(0) # [T, action_dim]
        proprio = data["proprio"].squeeze(0) # [proprio_dim]


        # import viser
        # import viser.transforms as vtf
        # from dp.util.matrix_utils import rot_6d_to_quat, quat_to_rot_6d
        # viser_server = viser.ViserServer()
        # # action format is [left_6d_rot, left_ee_ik_target_handle_position, left_gripper_pos, right_6d_rot, right_ee_ik_target_handle_position, right_gripper_pos]
        # state = data["proprio"][0].cpu().numpy() # index 0 batch and tstep (current)
        # # import pdb; pdb.set_trace()

        # left_t0_in_right = vtf.SE3.from_rotation_and_translation(
        #     vtf.SO3(wxyz=rot_6d_to_quat(np.asarray(state[:6]))[0]), np.asarray(state[6:9])
        # )

        # right_t0_in_world = vtf.SE3.from_rotation_and_translation(
        #     vtf.SO3(wxyz=rot_6d_to_quat(np.asarray(state[10:16]))[0]), np.asarray(state[16:19])
        # )

        # left_in_world = right_t0_in_world @ left_t0_in_right
        # left_in_world_6d_rot = quat_to_rot_6d(left_in_world.wxyz_xyz[..., :4][None, ...])[0]

        # state[:6] = torch.from_numpy(left_in_world_6d_rot)
        # state[6:9] = torch.from_numpy(left_in_world.wxyz_xyz[..., -3:])

        # # if self.action_dim == 29:
        # top_t0_in_right = vtf.SE3.from_rotation_and_translation(
        #     vtf.SO3(wxyz=rot_6d_to_quat(np.asarray(state[20:26]))[0]), np.asarray(state[26:29])
        # )
        # top_in_world = right_t0_in_world @ top_t0_in_right
        # top_in_world_6d_rot = quat_to_rot_6d(top_in_world.wxyz_xyz[..., :4][None, ...])[0]
        # state[20:26] = torch.from_numpy(top_in_world_6d_rot)
        # state[26:29] = torch.from_numpy(top_in_world.wxyz_xyz[..., -3:])

        # viser_server.scene.add_frame(f"left_t0_in_world", position=left_in_world.wxyz_xyz[-3:], wxyz=left_in_world.wxyz_xyz[:4])
        # viser_server.scene.add_frame(f"right_t0_in_world", position=right_t0_in_world.wxyz_xyz[-3:], wxyz=right_t0_in_world.wxyz_xyz[:4])
        # viser_server.scene.add_frame(f"right_t0_state_in_world", position=right_t0_in_world.wxyz_xyz[-3:], wxyz=right_t0_in_world.wxyz_xyz[:4])
        # viser_server.scene.add_frame(f"right_t0_in_world/left_in_right", position=left_t0_in_right.wxyz_xyz[-3:], wxyz=left_t0_in_right.wxyz_xyz[:4])

        # import pdb; pdb.set_trace()

        # option for not scaling 
        if self.enable_scale_action:
            actions = scale_action(action, self.stats, type="diffusion") 
        
        if self.enable_scale_proprio:
            proprio = scale_action(proprio, self.stats_proprio, type="diffusion") 
        # remove nans 
        if np.isnan(action).any():
            print("Warning: Num. NaNs in action: ", np.isnan(action).sum())
            actions = torch.nan_to_num(actions)
            print("NaNs in action data zeroed out")
            
        # get camera 
        camera = self.helper_load_camera(self.camera_keys, file_path, start, end)

        return {
            "action" : actions.float(), # num_pred_steps, 29
            "proprio" : proprio.float(), # 29
            "observation" : camera.float(), # num_camera, 3, 224, 224
        }
        
    def helper_load_episode_data(self, file_path : str, start : int, end : int):
        
        # get action path, joint path, and retrieved indices
        indices = np.arange(start + self.subsample_steps, end + self.subsample_steps, self.subsample_steps)
        episode_fp = file_path + "episode.h5"
        episode_data = h5py.File(episode_fp, "r")

        # get actions and proprio 
        actions = episode_data["action"][indices]
        proprio = episode_data["proprio"][indices]
        return actions, proprio

    # @line_profiler.profile
    def helper_load_camera(self, camera_keys : List[str], file_path : str, start : int, end : int):
        # image_left_path = file_path + f"_left/{start:04d}.jpg"
        # image_right_path =  file_path + f"_right/{start:04d}.jpg"
        # find all folders with jpgs in them
        image_paths = []
        for camera_key in camera_keys:
            image_paths.extend(glob(file_path + f"{camera_key}/{start:06d}.jpg", recursive=True))
        # import pdb; pdb.set_trace()
        # print(file_path + f"/{camera_key}/{start:04d}.jpg")
        # print(image_paths)
        camera_observations = []
        for image_path in image_paths:
            image = Image.open(image_path)
            # import pdb; pdb.set_trace()
            image = self.vision_transform(image)
            camera_observations.append(image)

        subsequence = torch.stack(camera_observations) # num_camera, 3, H, W
        return subsequence
