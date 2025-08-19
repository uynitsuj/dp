"""
python test/test_dataset.py --dataset-cfg.dataset-root /home/mfu/dataset/dp_gs 
"""
import os
import torch
import tyro 
import tqdm
import time
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np 

from dp_gs.dataset.image_dataset import SequenceDataset, VideoSampler, CollateFunction
from dp_gs.dataset.image_dataset_sim import SequenceDataset as SimSequenceDataset
from dp_gs.util.args import ExperimentConfig
import timm 
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dp_gs.dataset.utils import default_vision_transform
from timm.data.loader import MultiEpochsDataLoader
from dp_gs.dataset.utils import unscale_action

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

    tokenizer = None
    # tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

    # let's first set the batch size to 2
    # args.shared_cfg.batch_size = 2  

    # making train val split
    dataset_train = SimSequenceDataset(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        logging_config=args.logging_cfg,
        vision_transform=vision_transform,
        split="train",
        debug=False,
    )

    sampler = VideoSampler(
        dataset_train, 
        batch_size=args.shared_cfg.batch_size, 
        sequence_length=args.shared_cfg.seq_length
    )

    collate_fn = CollateFunction(args.shared_cfg.seq_length, tokenizer=tokenizer)
    dataloader = MultiEpochsDataLoader(
        dataset_train,
        batch_sampler=sampler,
        num_workers=args.trainer_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
    )
    

    print("Length of the training set: ", len(dataset_train))
    
    nbatch = next(iter(dataset_train))
    proprio = nbatch['proprio'].numpy()
    action = nbatch['action']
    action = unscale_action(action, stat=dataset_train.stats, type='diffusion').numpy()
    fig, axs = plt.subplots(3, 1)
    for k in range(3):
        axs[k].plot(proprio[k],'o', label='proprio', color='green', )
        axs[k].plot(range(1, action.shape[0]+1),action[:, k], label='action', color='red')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/proprio_action.png")   
    plt.clf()
    

    start_time = time.time()
    retrieval_time = []
    for i, nbatch in enumerate(tqdm.tqdm(dataloader)):
        if i != 0:
            retrieval_time.append(time.time() - start_time)
        obs = nbatch['observation'][0]
        obs = undo_vision_transform(obs, mean, std)
        media.write_video(f"{out_dir}/{i}_obs.mp4", obs, fps=10)

        # plot proprio and action 
        proprio = nbatch['proprio'][0].numpy() # (only plotting the first proprio step) T, K 
        action = nbatch['action'][0, -1] # (only plotting the first action step) T, M
        action = unscale_action(action, stat=dataset_train.stats, type='diffusion').numpy()
        

        # plot proprio with each dimension as subplot
        T, K = proprio.shape
        fig, axs = plt.subplots(3, 1)
        for k in range(3):
            axs[k].plot(proprio[:, k], label='proprio', color='green')
            axs[k].plot(range(T, T+action.shape[0]), action[:, k], label='action', color='red')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{i}_proprio_action.png")   
        plt.clf()

        if i == 10:
            break
    
    print(f"average retrieval time: {np.mean(retrieval_time):.4f} seconds with std {np.std(retrieval_time):.4f}")

if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)