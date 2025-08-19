# import os
# import timm
import torch
# import tyro 
# import tqdm
# import time
# import mediapy as media
# import matplotlib.pyplot as plt
import numpy as np 

# from dp_gs.dataset.dataset import SequenceDataset
# from dp_gs.util.args import ExperimentConfig
# import timm 
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import transforms_noaug_train

def undo_vision_transform(obs : torch.Tensor, mean : tuple = IMAGENET_DEFAULT_MEAN, std : tuple = IMAGENET_DEFAULT_STD):
    """
    Undo the vision transform applied to the observations.
    torch tensor has shape H, 3, H, W
    return np.ndarray with shape H, W, 3 at np.uint8
    """
    # undo normalization
    obs = obs.cpu()
    mean, std = torch.tensor(mean), torch.tensor(std)
    obs = obs.permute(0, 2, 3, 1)
    obs = obs * std + mean
    obs = obs.numpy()
    obs = np.clip(obs * 255, 0, 255).astype(np.uint8)
    return obs