"""
python test/test_dataset.py --dataset-cfg.dataset-root /home/mfu/dataset/dp_gs 
"""
import os
import timm
import torch
import tyro 
import tqdm
import time
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np 

from dp_gs.dataset.dataset import SequenceDataset
from dp_gs.util.args import ExperimentConfig
import timm 
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dp_gs.dataset.utils import default_vision_transform

def undo_vision_transform(obs : torch.Tensor, mean : tuple, std : tuple):
    """
    Undo the vision transform applied to the observations.
    torch tensor has shape T, num_cam, 3, H, W
    return np.ndarray with shape T, num_cam * H, W, 3 at np.uint8
    """
    # undo normalization
    mean, std = torch.tensor(mean), torch.tensor(std)
    obs = obs.permute(0, 1, 3, 4, 2)
    obs = obs * std + mean
    obs = obs.numpy()
    obs = np.clip(obs * 255, 0, 255).astype(np.uint8)
    obs = np.concatenate([obs[:, i] for i in range(obs.shape[1])], axis=1)
    return obs

def main(args : ExperimentConfig):
    number_of_samples = 100
    out_dir = "test_outputs/test_dataset_output"
    args.logging_cfg.output_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)

    vision_transform = default_vision_transform()
    print("vision transform: ", vision_transform)
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    # making train val split
    dataset_train = SequenceDataset(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        logging_config=args.logging_cfg,
        vision_transform=vision_transform,
        split="train",
        debug=False,
    )
    dataset_val = SequenceDataset(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        vision_transform=vision_transform,
        logging_config=args.logging_cfg,
        split="val"
    )
    
    print("Length of the training set: ", len(dataset_train))
    print("Length of the validation set: ", len(dataset_val))

    # random sample 5 data points from the training set
    random_indices = np.random.choice(len(dataset_train), 5, replace=True)
    conseq_indices = len(dataset_train) - np.arange(5) - 1

    indices = np.append(random_indices, conseq_indices)
    retrieval_time = []

    for i in tqdm.tqdm(indices):
        print("getting index: ", i)
        start = time.time()
        data = dataset_train.__getitem__(i)
        retrieval_time.append(time.time() - start)
        obs = data['observation']
        obs = undo_vision_transform(obs, mean, std)
        media.write_video(f"{out_dir}/{i}_obs.mp4", obs, fps=10)

        # plot proprio and action 
        proprio = data['proprio'].numpy() # (only plotting the first proprio step) T, K 
        action = data['action'].numpy()[:, 0] # (only plotting the first action step) T, M

        # plot proprio with each dimension as subplot
        T, K = proprio.shape
        fig, axs = plt.subplots(K, 1, figsize=(10, 2 * K))
        for k in range(K):
            axs[k].plot(proprio[:, k])
            axs[k].set_title(f"proprio_{k}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{i}_proprio.png")   
        plt.clf()

        # plot action with each dimension as subplot
        T, M = action.shape
        fig, axs = plt.subplots(M, 1, figsize=(10, 2 * M))
        for m in range(M):
            axs[m].plot(action[:, m])
            axs[m].set_title(f"action_{m}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{i}_action.png")
        plt.clf()
    
    print(f"average retrieval time: {np.mean(retrieval_time):.4f} seconds with std {np.std(retrieval_time):.4f}")

if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)