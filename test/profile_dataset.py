"""
python test/profile_dataset.py --dataset-cfg.dataset-root /home/mfu/dataset/dp_gs/transfer_tiger_241204 --shared-cfg.seq-length 4 --shared-cfg.num-pred-steps 16 --dataset-cfg.subsample-steps 4
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
import cProfile
from torch.utils.data import DataLoader 

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

    # TODO: add dataloader 
    
    print("Length of the training set: ", len(dataset_train))
    print("Length of the validation set: ", len(dataset_val))

    # random sample 5 data points from the training set
    random_indices = np.random.choice(len(dataset_train), 5, replace=True)
    conseq_indices = len(dataset_train) - np.arange(5) - 1

    # # dataset profiling
    # indices = np.append(random_indices, conseq_indices)
    # retrieval_time = []
    # for i in tqdm.tqdm(indices):
    #     start = time.time()
    #     data = dataset_train.__getitem__(i)
    #     import pdb; pdb.set_trace()
    #     retrieval_time.append(time.time() - start)
    # print(f"average retrieval time: {np.mean(retrieval_time):.4f} seconds with std {np.std(retrieval_time):.4f}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("construct data loader")
    dataloader = DataLoader(
        dataset_train,
        batch_size=32,
        num_workers=20,
        shuffle=True,
        timeout=2e6,
        pin_memory=False,
        persistent_workers=False,
    )
    print("Done!")
    retrieval_time = []
    start = time.time()
    for ind, nbatch in enumerate(tqdm.tqdm(dataloader)):
        retrieval_time.append(time.time() - start)
        # nbatch = {k: v.to(device, non_blocking=True) for k, v in nbatch.items()}
        start = time.time()
        if ind == 10:
            break

    print(f"average retrieval time: {np.mean(retrieval_time):.4f} seconds with std {np.std(retrieval_time):.4f}")

if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)