# CUDA_VISIBLE_DEVICES=0 python script/inference.py --model_ckpt_folder /shared/projects/icrl/dp_outputs/250208_1027 --ckpt_id 50
from dataclasses import replace
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# from dp.dataset.dataset import SequenceDataset
import tyro
import yaml
from timm.data.loader import MultiEpochsDataLoader
from transformers import AutoProcessor

from dp.dataset.image_dataset_v2 import CollateFunction, SequenceDataset, VideoSampler
from dp.dataset.utils import default_vision_transform as transforms_noaug_train
from dp.dataset.utils import unscale_action
from dp.policy.diffusion_wrapper import DiffusionWrapper
from dp.policy.model import Dinov2DiscretePolicy
from dp.util.args import InferenceConfig, DatasetConfig
from dp.util.config import TrainConfig


def _load_action_stats(stats_path: Optional[str]) -> Optional[Dict]:
    """Load action min/max (or q01/q99) for y-limits if available."""
    if not stats_path or not os.path.exists(stats_path):
        return None
    try:
        with open(stats_path, "r") as f:
            data = json.load(f)
        # Accept either {"actions": {...}} or {"norm_stats": {"actions": {...}}}
        actions = None
        if "actions" in data:  # {"actions": {"min": [...], "max": [...]} }
            actions = data["actions"]
        elif "norm_stats" in data and "actions" in data["norm_stats"]:
            actions = data["norm_stats"]["actions"]
        if actions is None:
            return None

        if "min" in actions and "max" in actions:
            return {"min": actions["min"], "max": actions["max"]}
        if "q01" in actions and "q99" in actions:
            return {"min": actions["q01"], "max": actions["q99"]}
        return None
    except Exception:
        return None


def _compute_grid(n_plots: int, max_cols: int = 5) -> Tuple[int, int]:
    cols = min(max_cols, max(1, n_plots))
    rows = math.ceil(n_plots / cols)
    return rows, cols


def _plot_action_comparison(
    save_path: str,
    proprio: np.ndarray,          # (B, T, D)
    pred_action: np.ndarray,      # (B, H, D)
    gt_action: np.ndarray,        # (B, H, D)
    action_stats: Optional[Dict], # {"min": [...], "max": [...]}
    action_names: Optional[List[str]] = None
):
    """
    Plot (per-dim) time-series: history (proprio, 0..T-1), then horizon pred vs gt (T..T+H-1).
    """
    B, T, D = proprio.shape
    _, H, Dp = pred_action.shape
    _, Hg, Dg = gt_action.shape
    D = min(D, Dp, Dg)  # align on overlapping dims
    if action_names is None:
        action_names = [f"A{i}" for i in range(D)]
    else:
        action_names = action_names[:D]

    rows, cols = _compute_grid(D, max_cols=5)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    time_hist = np.arange(T)
    time_fut = np.arange(T, T + H)

    for i in range(D):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        # History
        ax.plot(time_hist, proprio[0, :, i], label="proprio", linewidth=1.5)
        # Pred & GT over horizon
        ax.plot(time_fut, pred_action[0, :, i], linestyle="--", label="pred", linewidth=1.5)
        ax.plot(time_fut, gt_action[0, :, i], linestyle="-.", label="gt", linewidth=1.5)

        ax.set_title(action_names[i])
        ax.set_xlabel("t")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)

        # y-limits if stats provided
        if action_stats is not None and "min" in action_stats and "max" in action_stats:
            if i < len(action_stats["min"]) and i < len(action_stats["max"]):
                y_min = float(action_stats["min"][i])
                y_max = float(action_stats["max"][i])
                # add margin
                span = max(1e-6, (y_max - y_min))
                pad = 0.05 * span
                ax.set_ylim(y_min - pad, y_max + pad)

    # One legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_error_heatmap(
    save_path: str,
    pred_action: np.ndarray,  # (B, H, D)
    gt_action: np.ndarray,    # (B, H, D)
    action_names: Optional[List[str]] = None
):
    """
    Heatmap of |pred - gt| with dims on Y and horizon on X.
    Uses batch mean (B) if B>1; with your script B=1 so it's a single-episode error map.
    """
    assert pred_action.shape[:2] == gt_action.shape[:2], "Pred/GT horizon mismatch"
    _, H, Dp = pred_action.shape
    _, Hg, Dg = gt_action.shape
    D = min(Dp, Dg)

    # absolute error per (b, t, d)
    err = np.abs(pred_action[:, :, :D] - gt_action[:, :, :D])  # (B, H, D)
    err_mean = err.mean(axis=0).T  # (D, H) for heatmap

    if action_names is None:
        action_names = [f"A{i}" for i in range(D)]
    else:
        action_names = action_names[:D]

    fig, ax = plt.subplots(figsize=(max(8, H * 0.4), max(6, D * 0.25)))
    im = ax.imshow(err_mean, aspect="auto", origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |error|")

    ax.set_xlabel("Horizon step")
    ax.set_ylabel("Action dim")
    ax.set_title("Prediction Error Heatmap")

    ax.set_yticks(np.arange(D))
    ax.set_yticklabels(action_names, fontsize=8)
    ax.set_xticks(np.arange(0, H, max(1, H // 10)))

    ax.grid(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@dataclasses.dataclass
class InferenceConfig:
    # path to model checkpoint
    model_ckpt_folder: str = "/home/justinyu/nfs_us/justinyu/dp/dp_xmi_rby_soup_can_intergripper_29D_20250819_174130"
    ckpt_id : int = 165

def main(inference_config: InferenceConfig):
    # parsing args 
    args = inference_config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_yaml_path = os.path.join(args.model_ckpt_folder, 'run.yaml')
    train_args : TrainConfig = yaml.load(Path(train_yaml_path).read_text(), Loader=yaml.Loader)

    # vision transform
    if train_args.shared_cfg.s2:
        resolution = train_args.shared_cfg.image_size * 2
    else:
        resolution = train_args.shared_cfg.image_size

    vision_transform = transforms_noaug_train(resolution=resolution) # default img_size: Union[int, Tuple[int, int]] = 224,


    train_args = replace(train_args, dataset_cfg=replace(train_args.dataset_cfg, action_statistics=os.path.join(args.model_ckpt_folder, 'action_statistics.json')), shared_cfg=replace(train_args.shared_cfg, batch_size=1))


    if type(train_args.dataset_cfg) is DatasetConfig and train_args.dataset_cfg.dataset_root is None:
        raise ValueError("Dataset root directory is not set")
    else:
        train_args = replace(train_args, dataset_cfg=train_args.dataset_cfg.create())

    dataset_train = SequenceDataset(
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
        # # data normalized in dataset
        # # device transfer
        nbatch = {k: v.to(device) for k, v in nbatch.items()}
        pred_action = inferencer(nbatch, True)


        proprio = nbatch['proprio'].detach().to('cpu').numpy()
        T = proprio.shape[1]
        if isinstance(inferencer.model, Dinov2DiscretePolicy):
            gt_action = nbatch['action']
            # Use action_mask to get valid tokens for each sequence
            valid_actions = []
            for seq_actions, seq_mask in zip(gt_action, nbatch['action_mask'], strict=False):
                # Get only the valid tokens including EOS using the mask
                valid_seq = seq_actions[seq_mask].tolist()
                valid_actions.append(valid_seq[:-1]) # remove EOS
            gt_action = tokenizer.decode(valid_actions, time_horizon=train_args.shared_cfg.num_pred_steps, action_dim=inferencer.model.action_dim)
            gt_action = torch.tensor(gt_action, device=device)
        else:
            gt_action = nbatch['action'][:,-1]

        gt_action = unscale_action(gt_action, stat=inferencer.stats, type='diffusion').detach().to('cpu').numpy()

        # -------------------- plotting (refined) --------------------
        # Shapes expected:
        #   proprio:    (B, T, D)
        #   pred_action:(B, H, D)
        #   gt_action:  (B, H, D)

        # Optional y-limits via action statistics (min/max or q01/q99)
        stats_path = getattr(train_args.dataset_cfg, "action_statistics", None)
        action_stats = _load_action_stats(stats_path)

        # Optional action names (defaults to A0..A{D-1})
        action_dim = min(proprio.shape[-1], pred_action.shape[-1], gt_action.shape[-1])
        action_names = [f"A{i}" for i in range(action_dim)]

        # 1) Per-dimension comparison plot (history + pred vs gt)
        comp_path = f"pred_vs_gt_{idx}.png"
        _plot_action_comparison(
            save_path=comp_path,
            proprio=proprio,
            pred_action=pred_action,
            gt_action=gt_action,
            action_stats=action_stats,
            action_names=action_names,
        )

        # 2) Error heatmap over (horizon x dims)
        heatmap_path = f"error_heatmap_{idx}.png"
        _plot_error_heatmap(
            save_path=heatmap_path,
            pred_action=pred_action,
            gt_action=gt_action,
            action_names=action_names,
        )

        print(f"[saved] {comp_path}  |  {heatmap_path}")

        if idx > 10:
            break

if __name__ == "__main__":
    tyro.cli(main)