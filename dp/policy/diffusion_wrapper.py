import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import AutoProcessor

# from timm.data.transforms_factory import transforms_noaug_train
from dp.dataset.utils import default_vision_transform as transforms_noaug_train  # scale to 224 224 first 
from dp.dataset.utils import unscale_action
from dp.policy.model import DiffusionPolicy, Dinov2DiscretePolicy, SimplePolicy
from dp.util.args import ExperimentConfig

# def load_state_dict_flexible(model, state_dict):
#     """
#     Load state dict while handling both DDP and non-DDP scenarios.
#     """
#     # Print model and state dict keys for debugging
#     print("Model state_dict keys:", model.state_dict().keys())
#     print("Loaded state_dict keys:", state_dict.keys())
    
#     # Check if state_dict has 'module.' prefix
#     is_parallel = any(key.startswith('module.') for key in state_dict.keys())
    
#     try:
#         # If model is DDP but state_dict is not parallel
#         if hasattr(model, 'module') and not is_parallel:
#             missing, unexpected = model.module.load_state_dict(state_dict, strict=False)
        
#         # If model is not DDP but state_dict is parallel
#         elif not hasattr(model, 'module') and is_parallel:
#             prefix = 'module.'
#             new_state_dict = {key.removeprefix(prefix): value 
#                              for key, value in state_dict.items()}
#             missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
#         # If both are parallel or both are not parallel
#         else:
#             missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
#         print(f"Missing keys: {missing}")
#         print(f"Unexpected keys: {unexpected}")
        
#         return model
        
#     except Exception as e:
#         print(f"Error loading state dict: {str(e)}")
#         raise

def load_state_dict_flexible(model, state_dict):
    """
    Load state dict while handling both DDP and non-DDP scenarios.
    """
    if "model" in state_dict:
        state_dict = state_dict["model"]
    # Check if state_dict has 'module.' prefix
    is_parallel = any(key.startswith('module.') for key in state_dict.keys())
    
    # If model is DDP but state_dict is not parallel
    if hasattr(model, 'module') and not is_parallel:
        model.module.load_state_dict(state_dict)
    
    # If model is not DDP but state_dict is parallel
    elif not hasattr(model, 'module') and is_parallel:
        prefix = 'module.'
        new_state_dict = {key.removeprefix(prefix): value 
                         for key, value in state_dict.items()}
        model.load_state_dict(new_state_dict)
    
    # If both are parallel or both are not parallel
    else:
        model.load_state_dict(state_dict)
        
    return model

class DiffusionWrapper():
    def __init__(self, model_ckpt_folder, ckpt_id, device, denoising_step=100) -> None:
        train_yaml_path = os.path.join(model_ckpt_folder, 'run.yaml')
        model_ckpt_name = os.path.join(model_ckpt_folder, f'checkpoint_{ckpt_id}.pt')
        
        action_stats = json.load(open(os.path.join(model_ckpt_folder, 'action_statistics.json')))
        action_shape = action_stats["shape"]
        min_action = np.array(action_stats["min_action"]).reshape(action_shape)
        max_action = np.array(action_stats["max_action"]).reshape(action_shape)
        self.stats = {
            "min" : torch.from_numpy(min_action),
            "max" : torch.from_numpy(max_action),
        }
        
        args : ExperimentConfig = yaml.load(Path(train_yaml_path).read_text(), Loader=yaml.Loader)
        self.device = device
        if args.model_cfg.policy_type == "diffusion": 
            policy = DiffusionPolicy
            self.policy_type = "diffusion"
        elif args.model_cfg.policy_type == "simple":
            policy = SimplePolicy
            self.policy_type = "simple"
        elif args.model_cfg.policy_type == "discrete":
            policy = Dinov2DiscretePolicy
            self.policy_type = "discrete"
            self.tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
            extra_kwargs = {
                'vocab_size': self.tokenizer.vocab_size + 1,  # eos token
                'num_tokens': args.shared_cfg.num_tokens
            }
            
        self.model = policy(
            model_cfg=args.model_cfg,
            shared_cfg=args.shared_cfg,
            **(extra_kwargs if args.model_cfg.policy_type == "discrete" else {})
        )
        
        try:
            self.model = load_state_dict_flexible(self.model, torch.load(model_ckpt_name))
        except FileNotFoundError:
            print(f"File {model_ckpt_name} not found")
            raise
        self.model = self.model.to(device)
        self.model.eval()
        
        # vision transform
        if args.shared_cfg.s2:
            resolution = args.shared_cfg.image_size * 2
        else:
            resolution = args.shared_cfg.image_size
        self.resolution = resolution
        self.vision_transform = transforms_noaug_train(resolution=resolution)
        # self.validate_model_state()  # Add validation after loading
        if self.policy_type == "diffusion":
            self.inference_step = 10
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps = self.model.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type= self.model.prediction_type)
            
            # self.inference_step = self.model.num_diffusion_iters
            # self.noise_scheduler = DDPMScheduler(
            #     num_train_timesteps = self.model.num_diffusion_iters,
            #     beta_schedule='squaredcos_cap_v2',
            #     clip_sample=True,
            #     prediction_type= self.model.prediction_type)
            # self.noise_scheduler.set_timesteps(self.inference_step)
    
    def validate_model_state(self):
        """
        Validate model state after loading checkpoint
        """
        # Check if model parameters contain NaN values
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"WARNING: NaN values found in {name}")
            
        # Test forward pass with dummy data
        self.model.eval()
        with torch.no_grad():
            try:
                dummy_batch = {
                    "observation": torch.randn(1, self.model.obs_horizon, 
                                            self.model.num_cameras, 3, self.resolution, self.resolution).to(self.device),
                    "proprio": torch.randn(1, self.model.obs_horizon, 20).to(self.device)
                }
                _ = self.model.forward_inference(dummy_batch, self.vision_transform)
                print("Forward pass successful with dummy data")
            except Exception as e:
                print(f"Forward pass failed: {e!s}")
                raise

    def _process_images(self, nimage: torch.Tensor) -> torch.Tensor:
        """
        Process images through vision transform and reshape.
        
        Args:
            nimage: Input images of shape (B, T, num_cameras, C, H, W)
            
        Returns:
            Processed images of shape (B, T, num_cameras, C, H, W)
        """
        B, T = nimage.shape[:2]
        # Reshape for vision transform
        nimage = nimage.reshape(B*T*self.model.num_cameras, *nimage.shape[3:])
        if nimage.shape[1] != 3:
            nimage = nimage.permute(0, 3, 1, 2)  # T, C, H, W
        nimage = self.vision_transform(nimage).float()
        # Reshape back to original format
        nimage = nimage.reshape(B, T, self.model.num_cameras, *nimage.shape[1:])
        return nimage

    def _get_vision_features(self, nimage: torch.Tensor) -> torch.Tensor:
        """
        Extract vision features using the appropriate encoder.
        
        Args:
            nimage: Input images of shape (B, T, num_cameras, C, H, W)
            
        Returns:
            Vision features of shape (B, T, num_cameras * vision_feature_dim)
        """
        B = nimage.shape[0]
        
        if self.model.vision_model_type == "dino":
            # Process images through DINOv2
            flat_imgs = nimage.flatten(start_dim=0, end_dim=2)  # (B*T*num_cameras, C, H, W)
            with torch.no_grad():
                dino_out = self.model.vision_encoder.forward_features(flat_imgs)  # (B*T*num_cameras, num_patches, embed_dim)
            
            # Reshape to (B, T, num_cameras, num_patches, embed_dim)
            dino_out = dino_out.view(B, -1, self.model.num_cameras, dino_out.shape[1], self.model.vision_feature_dim)
            
            if self.model.use_attention_pooling:
                # Batch process attention pooling for all time steps and cameras at once
                dino_flat = dino_out.view(B * self.model.obs_horizon * self.model.num_cameras, -1, self.model.vision_feature_dim)
                
                # Apply attention pooling in batch
                pooled, _ = self.model.attention_pool(
                    dino_flat,  # (B*T*num_cameras, num_patches, embed_dim)
                    dino_flat,  # (B*T*num_cameras, num_patches, embed_dim)
                    dino_flat   # (B*T*num_cameras, num_patches, embed_dim)
                )
                
                # Take the first token (CLS token) as the pooled representation
                pooled = pooled[:, 0].view(B, self.model.obs_horizon, self.model.num_cameras, self.model.vision_feature_dim)
                image_features = pooled.reshape(B, -1, self.model.num_cameras * self.model.vision_feature_dim)
            else:
                # Traditional approach: take the first token (CLS token) as the feature
                image_features = dino_out[:, :, :, 0, :]  # (B, T, num_cameras, embed_dim)
                image_features = image_features.reshape(B, -1, self.model.num_cameras * self.model.vision_feature_dim)
        else:
            # Traditional ResNet approach
            image_features = self.model.vision_encoder(nimage.flatten(end_dim=2))
            image_features = image_features.reshape(*nimage.shape[:3],-1).view(*nimage.shape[:2],-1)
        
        return image_features

    def forward_diffusion(self, nbatch, denormalize=True):
        """
        Forward pass for diffusion policy models.
        
        Args:
            nbatch: Batch dictionary containing:
                - observation: Images of shape (B, T, num_cameras, C, H, W)
                - proprio: Proprioceptive data of shape (B, T, D)
            denormalize: Whether to denormalize the output actions
            
        Returns:
            Predicted actions of shape (B, action_horizon, action_dim)
        """
        # create a deep copy of nbatch 
        nbatch = {k: v.clone() for k, v in nbatch.items()}
        for key in nbatch:
            # if nbatch[key] is a tensor move to device
            if isinstance(nbatch[key], torch.Tensor):
                nbatch[key] = nbatch[key].to(self.device)
        with torch.inference_mode():
            # Use model's forward_inference method
            naction = self.model.forward_inference(nbatch, self.vision_transform)
            
            # Handle denormalization and left/right prediction
            if denormalize:
                if self.model.pred_left_only or self.model.pred_right_only:
                    naction = torch.concatenate([naction, naction], dim=-1)
                naction = unscale_action(naction, stat=self.stats, type='diffusion')
            
            # Handle left/right prediction
            if self.model.pred_left_only:
                proprio_right = nbatch["proprio"][:, -1:, self.model.action_dim:]
                naction[:, :, self.model.action_dim:] = proprio_right
            
            if self.model.pred_right_only:
                proprio_left = nbatch["proprio"][:, -1:, :self.model.action_dim]
                naction[:, :, :self.model.action_dim] = proprio_left
            
            naction = naction.detach().to('cpu').numpy()
            return naction

    def forward_simple(self, nbatch, denormalize=True):
        """
        Forward pass for simple policy models.
        
        Args:
            nbatch: Batch dictionary containing:
                - observation: Images of shape (B, T, num_cameras, C, H, W)
                - proprio: Proprioceptive data of shape (B, T, D)
            denormalize: Whether to denormalize the output actions
            
        Returns:
            Predicted actions of shape (B, action_horizon, action_dim)
        """
        with torch.inference_mode():
            # Use model's forward_inference method
            naction = self.model.forward_inference(nbatch, self.vision_transform)
            
            # Handle denormalization and left/right prediction
            if denormalize:
                if self.model.pred_left_only or self.model.pred_right_only:
                    naction = torch.concatenate([naction, naction], dim=-1)
                naction = unscale_action(naction, stat=self.stats, type='diffusion')
            
            # Handle left/right prediction
            if self.model.pred_left_only:
                proprio_right = nbatch["proprio"][:, -1:, self.model.action_dim:]
                naction[:, :, self.model.action_dim:] = proprio_right
            
            if self.model.pred_right_only:
                proprio_left = nbatch["proprio"][:, -1:, :self.model.action_dim]
                naction[:, :, :self.model.action_dim] = proprio_left
            
            naction = naction.detach().to('cpu').numpy()
            return naction

    def forward_discrete(self, nbatch, denormalize=True, return_raw_pred=False):
        """
        Forward pass for discrete policy models.
        
        Args:
            nbatch: Batch dictionary containing:
                - observation: Images of shape (B, T, num_cameras, C, H, W)
                - proprio: Proprioceptive data of shape (B, T, D)
            denormalize: Whether to denormalize the output actions
            return_raw_pred: Whether to return raw predictions
            
        Returns:
            Predicted actions of shape (B, action_horizon, action_dim)
        """
        with torch.inference_mode():
            # Use model's forward_inference method
            predictions = self.model.forward_inference(nbatch, self.vision_transform)
            
            # Remove EOS token and decode
            # Find first occurrence of EOS token (vocab_size) or take full sequence if no EOS
            # Create mask where 1s indicate valid tokens before first EOS, 0s after
            eos_positions = (predictions == self.tokenizer.vocab_size).float() # B x seq_len boolean mask for EOS tokens
            # Find first EOS token position for each sequence
            first_eos = (predictions == self.tokenizer.vocab_size).float().argmax(dim=1)
            # Create mask that is 1 for all positions before first EOS
            valid_tokens = torch.arange(predictions.shape[1], device=predictions.device)[None, :] < first_eos[:, None]
            valid_tokens = eos_positions == 0 # Mask of valid tokens (before first EOS)
            
            # Apply mask to predictions
            masked_predictions = predictions * valid_tokens
            
            # Decode tokens to actions
            decoded_actions = []
            bad_predictions = 0
            bad_pred_idx = []
            for seq in masked_predictions:
                # Filter out padding and convert to list
                valid_seq = seq[seq > 0].tolist()
                # Decode single sequence
                decoded = self.tokenizer.decode(
                    [valid_seq], 
                    time_horizon=self.model.action_horizon, 
                    action_dim=self.model.action_dim
                )
                # Check if decoded is all zeros
                if np.all(decoded == 0):
                    bad_pred_idx.append(bad_predictions)
                    bad_predictions += 1
                decoded_actions.append(decoded)
            print(f"Number of bad predictions (all zeros): {bad_predictions} out of {len(masked_predictions)}")
            
            # Stack all decoded actions
            naction = np.concatenate(decoded_actions, axis=0)
            
            if denormalize:
                if self.model.pred_left_only or self.model.pred_right_only:
                    naction = np.concatenate([naction, naction], axis=-1)
                naction = unscale_action(torch.from_numpy(naction), stat=self.stats, type='diffusion').numpy()

            proprio_data = nbatch["proprio"].detach().cpu().numpy()
            for idx in bad_pred_idx:
                # replace with gt proprio
                naction[idx] = proprio_data[idx, -1:, :]
            
            # handle pred left only 
            if self.model.pred_left_only:
                proprio_right = proprio_data[:, -1:, self.model.action_dim:]
                naction[:, :, self.model.action_dim:] = proprio_right
            
            if self.model.pred_right_only:
                proprio_left = proprio_data[:, -1:, :self.model.action_dim]
                naction[:, :, :self.model.action_dim] = proprio_left

            if return_raw_pred:
                return naction, predictions
            else:
                return naction

    def __call__(self, nbatch, denormalize=True):
        if self.policy_type == "diffusion":
            return self.forward_diffusion(nbatch, denormalize)
        elif self.policy_type == "simple":
            return self.forward_simple(nbatch, denormalize)
        elif self.policy_type == "discrete":
            return self.forward_discrete(nbatch, denormalize)
