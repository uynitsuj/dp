# CUDA_VISIBLE_DEVICES=0 python script/inference.py --model_ckpt_folder /shared/projects/icrl/dp_outputs/250208_1027 --ckpt_id 50
import torch
from timm.data.loader import MultiEpochsDataLoader
from dp_gs.policy.model import DiffusionPolicy
from dp_gs.util.args import InferenceConfig, ExperimentConfig
from dp_gs.dataset.dataset import SequenceDataset
import tyro
from dp_gs.dataset.utils import default_vision_transform as transforms_noaug_train
from pathlib import Path
import yaml
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import matplotlib.pyplot as plt
import os
import json
from dp_gs.dataset.utils import unscale_action
import numpy as np
from dp_gs.policy.diffusion_wrapper import DiffusionWrapper
from dp_gs.dataset.image_dataset import SequenceDataset, VideoSampler, CollateFunction
from dp_gs.dataset.image_dataset_sim import SequenceDataset as SimSequenceDataset
from dp_gs.policy.model import Dinov2DiscretePolicy
from transformers import AutoProcessor
    
if __name__ == '__main__':
    # parsing args 
    args = tyro.cli(InferenceConfig)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_yaml_path = os.path.join(args.model_ckpt_folder, 'run.yaml')
    train_args : ExperimentConfig = yaml.load(Path(train_yaml_path).read_text(), Loader=yaml.Loader)

    # vision transform
    if train_args.shared_cfg.s2:
        resolution = train_args.shared_cfg.image_size * 2
    else:
        resolution = train_args.shared_cfg.image_size

    vision_transform = transforms_noaug_train(resolution=resolution) # default img_size: Union[int, Tuple[int, int]] = 224,

    train_args.dataset_cfg.action_statistics = os.path.join(args.model_ckpt_folder, 'action_statistics.json')
    train_args.shared_cfg.batch_size = 1 

    dataset_train = SimSequenceDataset(
        dataset_config=train_args.dataset_cfg,
        shared_config=train_args.shared_cfg,
        logging_config=train_args.logging_cfg,
        vision_transform=vision_transform,
        split="train",
        debug=False,
    )

    sampler = VideoSampler(
        dataset_train, 
        batch_size=train_args.shared_cfg.batch_size, 
        sequence_length=train_args.shared_cfg.seq_length
    )

    # Initialize tokenizer if using discrete policy
    tokenizer = None
    if train_args.model_cfg.policy_type == "discrete":
        tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

    collate_fn = CollateFunction(
        train_args.shared_cfg.seq_length,
        tokenizer=tokenizer,
        max_token_length=train_args.shared_cfg.num_tokens if tokenizer else None, 
        pred_left_only=train_args.model_cfg.policy_cfg.pred_left_only,
        pred_right_only=train_args.model_cfg.policy_cfg.pred_right_only
    )

    dataloader = MultiEpochsDataLoader(
        dataset_train,
        batch_sampler=sampler,
        num_workers=train_args.trainer_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
    )

    inferencer = DiffusionWrapper(args.model_ckpt_folder, args.ckpt_id, device=device)
    for idx, nbatch in enumerate(dataloader):
        # data normalized in dataset
        # device transfer
        nbatch = {k: v.to(device) for k, v in nbatch.items()}
        pred_action = inferencer(nbatch, True)
        if inferencer.model.pred_left_only:
            pred_action = pred_action[..., :10]
        elif inferencer.model.pred_right_only:
            pred_action = pred_action[..., 10:]
        proprio = nbatch['proprio'].detach().to('cpu').numpy()
        T = proprio.shape[1]
        if isinstance(inferencer.model, Dinov2DiscretePolicy):
            gt_action = nbatch['action']
            # Use action_mask to get valid tokens for each sequence
            valid_actions = []
            for seq_actions, seq_mask in zip(gt_action, nbatch['action_mask']):
                # Get only the valid tokens including EOS using the mask
                valid_seq = seq_actions[seq_mask].tolist()
                valid_actions.append(valid_seq[:-1]) # remove EOS
            gt_action = tokenizer.decode(valid_actions, time_horizon=train_args.shared_cfg.num_pred_steps, action_dim=inferencer.model.action_dim)
            gt_action = torch.tensor(gt_action, device=device)
        else:
            gt_action = nbatch['action'][:,-1]
        if inferencer.model.pred_left_only or inferencer.model.pred_right_only:
            gt_action = torch.concatenate([gt_action, gt_action], dim=-1)
        gt_action = unscale_action(gt_action, stat=inferencer.stats, type='diffusion').detach().to('cpu').numpy()
        if inferencer.model.pred_left_only:
            gt_action = gt_action[..., :10]
        elif inferencer.model.pred_right_only:
            gt_action = gt_action[..., 10:]
        #subplot of 20
        if inferencer.model.pred_left_only or inferencer.model.pred_right_only:
            num_subplots = 10 
            fig, axes = plt.subplots(2, 5)
        else:
            num_subplots = 20
            fig, axes = plt.subplots(4, 5)
        for i in range(num_subplots):
            ax = axes[i//5,i%5]
            ax.plot(range(T), proprio[0, :, i], label='proprio', color='green')
            ax.plot(range(T, T+pred_action[0, :, i].shape[0]),pred_action[0, :, i], label='pred', color='red')
            ax.plot(range(T, T+gt_action[0, :, i].shape[0]),gt_action[0, :, i], label='gt', color='blue')
            
        # plt.plot(pred_action[0, :, 0], label='pred', color='red')
        # plt.plot(gt_action[0, :, 0], label='gt', color='blue')
        # plt.ylim(-1, 1)
        plt.legend()
        fig.savefig(f'pred_vs_gt_{idx}.png')
        if idx > 10:
            break