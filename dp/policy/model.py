from typing import Callable

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from peft import LoraConfig, get_peft_model
from s2wrapper import forward as multiscale_forward
from torch import nn

from dp.policy.action_head import DiscreteActionDecoder
from dp.policy.transformer_for_diffusion import TransformerForDiffusion
from dp.policy.scale_transformer_diffusion import ScaleTransformerDiffusionPolicy
from dp.policy.utils import ConditionalUnet1D
from dp.util.args import ModelConfig, SharedConfig
from dp.policy.vision.dinov3 import Dinov3ImageBranch

def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def format_number(num):
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    return str(num)

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet

# Just remove the forward method's flattening operation
class S2ResNet(nn.Module):
    def __init__(self, original_resnet):
        super().__init__()
        # self.features = nn.Sequential(*list(get_resnet("resnet18").children())[:-2])
        modules = list(original_resnet.children())[:-2]
        self.features = nn.Sequential(*modules)
        
    def forward(self, x):
        features = self.features(x)
        # Reshape to (B, 512, 49)
        B, C, H, W = features.shape
        features = features.reshape(B, C, H*W)
        # Transpose to get (B, 49, 512)
        features = features.transpose(1, 2)
        return features

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

class MLPActionRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(MLPActionRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        return action
    
class SimplePolicy(nn.Module):
    hidden_dim = 256
    action_dim = 20
    lowdim_obs_dim = 20
    vision_feature_dim = 512
    def __init__(
        self, 
        model_cfg : ModelConfig,
        shared_cfg : SharedConfig, 
    ):
        super(SimplePolicy, self).__init__()
        
        # Determine which vision model to use
        self.vision_model_type = model_cfg.policy_cfg.simple_vision_model
        self.use_attention_pooling = model_cfg.policy_cfg.simple_attention_pooling
        self.image_resolution = shared_cfg.image_size

        # Create vision encoder based on selected model type
        if self.vision_model_type == "dino":
            # Use DINOv2 as the vision encoder
            self.timm_vision_encoder = model_cfg.policy_cfg.timm_vision_encoder
            if self.timm_vision_encoder is None:
                self.timm_vision_encoder = "vit_base_patch14_dinov2.lvd142m"
                print("defaulting to dinov2 backbone: ", self.timm_vision_encoder)
            
            self.vision_encoder = timm.create_model(
                self.timm_vision_encoder,
                pretrained=True,
                img_size=(self.image_resolution, self.image_resolution),
                num_classes=0,
                global_pool="",  # No global pooling, we'll handle it ourselves
            )
            
            # Set vision feature dimension based on DINOv2's embed dimension
            self.vision_feature_dim = self.vision_encoder.embed_dim
            
            # Freeze vision encoder parameters
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
                
            # Apply LoRA if specified
            self.lora_rank_vision_encoder = model_cfg.policy_cfg.lora_rank_vision_encoder
            if self.lora_rank_vision_encoder > 0:
                lora_config = LoraConfig(
                    r=self.lora_rank_vision_encoder,
                    lora_alpha=self.lora_rank_vision_encoder,
                    target_modules=["qkv"]
                )
                self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)
        else:
            # Use ResNet as the vision encoder (default)
            self.vision_encoder = get_resnet('resnet18')
            self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
            self.vision_feature_dim = 512  # ResNet18 output dimension
        
        # Set other parameters
        self.num_cameras = shared_cfg.num_cameras
        self.obs_horizon = shared_cfg.seq_length
        self.action_horizon = shared_cfg.num_pred_steps
        self.only_vision = model_cfg.policy_cfg.only_vision
        self.gripper_loss_w = model_cfg.policy_cfg.gripper_loss_w
        
        # Calculate observation dimension
        if self.only_vision:
            obs_dim = self.vision_feature_dim * self.num_cameras
        else:
            obs_dim = self.vision_feature_dim * self.num_cameras + self.lowdim_obs_dim
        
        self.global_cond_dim = obs_dim * self.obs_horizon
        
        # Handle left/right prediction
        self.pred_left_only = model_cfg.policy_cfg.pred_left_only
        self.pred_right_only = model_cfg.policy_cfg.pred_right_only
        if self.pred_left_only or self.pred_right_only:
            self.action_dim = self.action_dim // 2
            print("pred_left_only or pred_right_only is enabled, changed action dim to ", self.action_dim)
        
        # Add position embeddings for cameras and timesteps
        self.camera_embeddings = nn.Parameter(
            torch.randn(self.num_cameras)[:, None]
        )
        self.temporal_embeddings = nn.Parameter(
            torch.randn(self.obs_horizon)[:, None, None]
        )
        
        # Create action regressor
        if self.use_attention_pooling:
            # Create learnable query tokens for attention pooling
            self.query_tokens = nn.Parameter(
                torch.randn(1, 1, self.vision_feature_dim)  # Single query token
            )
            
            # Use attention pooling instead of concatenation
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=self.vision_feature_dim,
                num_heads=8,
                batch_first=True
            )
            
            self.obs_state_proj = nn.Linear(obs_dim, self.vision_feature_dim)

            # Add a second attention layer for temporal relationships
            self.temporal_query = nn.Parameter(
                torch.randn(1, 1, self.vision_feature_dim)  # Single query token for temporal attention
            )
            
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=self.vision_feature_dim,
                num_heads=8,
                batch_first=True
            )
            
            # Final MLP for action prediction
            # When using attention pooling, input dimension is vision_feature_dim
            self.mlp = MLPActionRegressor(
                self.vision_feature_dim,  # Input dimension is now just vision_feature_dim
                self.hidden_dim, 
                self.action_dim * self.action_horizon
            )
        else:
            # Use traditional concatenation approach
            self.mlp = MLPActionRegressor(
                self.global_cond_dim, 
                self.hidden_dim, 
                self.action_dim * self.action_horizon
            )

    def forward(self, nbatch):
        nimage = nbatch["observation"][:, :, :self.num_cameras] # pick the first image # B, T, 1, C, H ,W
        nagent_pos = nbatch["proprio"] # pick the current proprio # B, D
        # nbatch['action'] has shape (B, T, num_pred_steps, action_dim)
        if self.pred_right_only:
            naction = nbatch['action'][:, -1, ..., self.action_dim:]
        else:
            naction = nbatch['action'][:, -1, ..., :self.action_dim] # B, N, D # enforce pred left only if action dim is changed
        
        obs_features = self._get_observation_features(nimage, nagent_pos)
        
        if self.use_attention_pooling:
            # add mlp here 
            obs_cond = self.obs_state_proj(obs_features)
            # apply temporal attention
            temporal_q = self.temporal_query.expand(nimage.shape[0], 1, -1)

            obs_cond = self.temporal_attention(
                query=temporal_q,
                key=obs_cond,
                value=obs_cond
            )[0].squeeze(1) # (B, embed_dim)
        else:
            # Flatten for MLP input
            obs_cond = obs_features.flatten(start_dim=1)

        # Predict action using MLP
        pred_action = self.mlp(obs_cond)
        pred_action = pred_action.view(nimage.shape[0], self.action_horizon, self.action_dim)
        
        # calculate loss 
        loss = nn.functional.l1_loss(pred_action, naction, reduction="none")
        # if self.pred_left_only or self.pred_right_only:
        #     loss[:, :, -1] *= self.gripper_loss_w
        # else:
        #     loss[:, :, -1] *= self.gripper_loss_w
        #     loss[:, :, self.action_dim // 2 - 1] *= self.gripper_loss_w
        loss = loss.mean()
        return loss
        
    def forward_inference(self, nbatch, vision_transform=None):
        """
        Inference method for the simple policy.
        
        Args:
            nbatch: Batch dictionary containing:
                - observation: Images of shape (B, T, num_cameras, C, H, W)
                - proprio: Proprioceptive data of shape (B, T, D)
            vision_transform: Optional function to transform images
            
        Returns:
            Predicted actions of shape (B, action_horizon, action_dim)
        """
        # Process batch
        nimage, nagent_pos, B = self._process_batch(nbatch, vision_transform)
        
        # Get observation features
        obs_features = self._get_observation_features(nimage, nagent_pos)
        
        if self.use_attention_pooling:
            # add mlp here 
            obs_cond = self.obs_state_proj(obs_features)
            # apply temporal attention
            temporal_q = self.temporal_query.expand(B, 1, -1)

            obs_cond = self.temporal_attention(
                query=temporal_q,
                key=obs_cond,
                value=obs_cond
            )[0].squeeze(1) # (B, embed_dim)
        else:
            # Flatten for MLP input
            obs_cond = obs_features.flatten(start_dim=1)
        
        # Predict action using MLP
        pred_action = self.mlp(obs_cond)
        pred_action = pred_action.view(B, self.action_horizon, self.action_dim)
        
        return pred_action
        
    def __repr__(self):
        """
        Return a string representation of the SimplePolicy model.
        """
        width = 70  # Slightly narrower for better display in notebooks/console
        border = "=" * width
        section_border = "-" * width
        
        # Collect all information
        model_info = [
            ("Model Architecture", [
                ("Vision Encoder", self.vision_model_type),
                ("Attention Pooling", self.use_attention_pooling),
                ("Num Cameras", self.num_cameras),
                ("Seq Length", self.obs_horizon),
                ("Action Horizon", self.action_horizon)
            ]),
            ("Dimensions", [
                ("Proprio Input", self.lowdim_obs_dim),
                ("Action Dim", self.action_dim),
                ("Vision Feature Dim", self.vision_feature_dim),
            ]),
            ("Parameters", [
                ("Total", format_number(count_parameters(self))),
                ("Vision Encoder", format_number(count_parameters(self.vision_encoder))),
                ("Vision Encoder Lora", format_number(count_parameters(self.vision_encoder, trainable_only=True))),
                ("Policy", format_number(count_parameters(self.mlp))),
                ("Trainable", format_number(sum(p.numel() for p in self.parameters() if p.requires_grad)))
            ])
        ]

        # Build the string representation
        lines = [
            border,
            f"{'Simple Policy Model':^{width}}",
            border,
            ""
        ]

        for section_title, items in model_info:
            lines.append(f"{section_title:^{width}}")
            lines.append(section_border)
            
            # Find maximum length for alignment
            max_key_length = max(len(k) for k, _ in items)
            
            for key, value in items:
                # Add each line with proper spacing
                lines.append(f"{key:<{max_key_length}} : {value:>}")
            lines.append("")

        lines.append(border)

        # Join all lines with newlines
        return "\n".join(lines)

    def _process_images(self, nimage: torch.Tensor, vision_transform) -> torch.Tensor:
        """
        Process images through vision transform and reshape.
        
        Args:
            nimage: Input images of shape (B, T, num_cameras, C, H, W)
            vision_transform: Function to transform images
            
        Returns:
            Processed images of shape (B, T, num_cameras, C, H, W)
        """
        B, T = nimage.shape[:2]
        # Reshape for vision transform
        nimage = nimage.reshape(B*T*self.num_cameras, *nimage.shape[3:])
        if nimage.shape[1] != 3:
            nimage = nimage.permute(0, 3, 1, 2)  # T, C, H, W
        nimage = vision_transform(nimage).float()
        # Reshape back to original format
        nimage = nimage.reshape(B, T, self.num_cameras, *nimage.shape[1:])
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

        # Process images through vision encoder
        if self.vision_model_type == "dino":
            # Process images through DINOv2
            flat_imgs = nimage.flatten(start_dim=0, end_dim=2)  # (B*T*num_cameras, C, H, W)
            with torch.no_grad():
                dino_out = self.vision_encoder.forward_features(flat_imgs)  # (B*T*num_cameras, num_patches, embed_dim)
            
            # Reshape to (B, T, num_cameras, num_patches, embed_dim)
            dino_out = dino_out.view(B, -1, self.num_cameras, dino_out.shape[1], self.vision_feature_dim)
            
            if self.use_attention_pooling:
                # add camera embedding and temporal embedding
                dino_out = dino_out + self.camera_embeddings[..., None] + self.temporal_embeddings[..., None]
                
                # First attention: across patches within each image
                # Reshape for patch attention
                dino_flat = dino_out.view(B * self.obs_horizon * self.num_cameras, -1, self.vision_feature_dim)
                
                # Expand query token for batch size
                query = self.query_tokens.expand(B * self.obs_horizon * self.num_cameras, 1, -1)
                
                # Apply patch attention with learnable query
                pooled_patches, _ = self.attention_pool(
                    query=query,  # (B*T*num_cameras, 1, embed_dim)
                    key=dino_flat,  # (B*T*num_cameras, num_patches, embed_dim)
                    value=dino_flat  # (B*T*num_cameras, num_patches, embed_dim)
                )
                
                # Reshape back to (B, T, num_cameras, embed_dim)
                pooled_patches = pooled_patches.view(B, self.obs_horizon, self.num_cameras, self.vision_feature_dim)

                # reshape to (B, T, num_cameras * embed_dim)
                image_features = pooled_patches.reshape(B, self.obs_horizon, self.num_cameras * self.vision_feature_dim)
            else:
                # Traditional approach: take the first token (CLS token) as the feature
                image_features = dino_out[:, :, :, 0, :]  # (B, T, num_cameras, embed_dim)
                # add camera embedding and temporal embedding
                image_features = image_features + self.camera_embeddings + self.temporal_embeddings
                image_features = image_features.reshape(B, -1, self.num_cameras * self.vision_feature_dim)
        else:
            # Traditional ResNet approach
            image_features = self.vision_encoder(nimage.flatten(end_dim=2))  # to incorporate num_camera dimension
            # convert first to (B, obs_horizon, self.num_cameras, D) 
            image_features = image_features.reshape(*nimage.shape[:3],-1)

            # add camera pe and temporal pe
            image_features = image_features + self.camera_embeddings + self.temporal_embeddings

            # flatten it to (B, obs_horizon, self.num_cameras * D)
            image_features = image_features.view(*nimage.shape[:2],-1)

        return image_features

    def _get_observation_features(self, nimage, nagent_pos):
        """
        Get observation features by combining vision and proprioceptive features.
        
        Args:
            nimage: Processed images
            nagent_pos: Proprioceptive data
            
        Returns:
            Observation features ready for model input
        """
        # Get vision features
        image_features = self._get_vision_features(nimage)
        
        # Combine vision and proprio features
        if not self.only_vision:
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        else:
            obs_features = image_features
        
        return obs_features

    def _process_batch(self, nbatch, vision_transform=None):
        """
        Process a batch of data for inference.
        
        Args:
            nbatch: Batch dictionary containing:
                - observation: Images of shape (B, T, num_cameras, C, H, W)
                - proprio: Proprioceptive data of shape (B, T, D)
            vision_transform: Optional function to transform images
            
        Returns:
            Tuple of (processed_images, proprioceptive_data, batch_size)
        """
        # Process images
        nimage = nbatch["observation"][:, :self.obs_horizon, :self.num_cameras]
        if vision_transform is not None:
            nimage = self._process_images(nimage, vision_transform)
        
        # Get proprioceptive data
        nagent_pos = nbatch["proprio"][:, :self.obs_horizon]
        batch_size = nagent_pos.shape[0]
        
        return nimage, nagent_pos, batch_size

class DiffusionPolicy(nn.Module):
    action_dim = 29
    lowdim_obs_dim = 29
    vision_feature_dim = 512

    def __init__(
        self, 
        model_cfg : ModelConfig,
        shared_cfg : SharedConfig, 
    ): 
        super().__init__()
        self.obs_horizon = shared_cfg.seq_length
        self.action_horizon = shared_cfg.num_pred_steps
        self.pred_left_only = model_cfg.policy_cfg.pred_left_only
        self.pred_right_only = model_cfg.policy_cfg.pred_right_only
        self.num_cameras = shared_cfg.num_cameras
        self.camera_keys = shared_cfg.camera_keys
        self.s2 = shared_cfg.s2 # https://github.com/bfshi/scaling_on_scales
        self.timm_vision_encoder = model_cfg.policy_cfg.timm_vision_encoder
        self.vision_encoder_pretrained_type = model_cfg.policy_cfg.vision_encoder_pretrained_type
        self.lora_rank_vision_encoder = model_cfg.policy_cfg.lora_rank_vision_encoder
        self.diffusion_model_type = model_cfg.policy_cfg.diffusion_model_type
        self.gripper_loss_w = model_cfg.policy_cfg.gripper_loss_w
        self.objective = model_cfg.policy_cfg.diffusion_objective
        # print("gripper_loss_w: ", self.gripper_loss_w)

        if self.timm_vision_encoder is not None:
            assert not self.s2, "Scaling on scale is not supported for timm vision encoder for now."
            self.vision_encoder = timm.create_model(
                self.timm_vision_encoder, # vit_base_patch14_reg4_dinov2.lvd142m for register tokens
                pretrained=True,
                num_classes=0,
                global_pool="token",
            )
            # freeze except the final attention pool and fc_norm
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_feature_dim = self.vision_encoder.embed_dim
        
            if self.lora_rank_vision_encoder > 0:
                # lora the vision encoder 
                lora_config = LoraConfig(
                    r=self.lora_rank_vision_encoder,
                    lora_alpha=self.lora_rank_vision_encoder,
                    target_modules=["qkv"]
                )
                self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)

        elif self.vision_encoder_pretrained_type == "dinov3":
            assert not self.s2, "Scaling on scale is not supported for dinov3 for now."
            self.vision_encoder = Dinov3ImageBranch(
                model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
                crop_size=(200, 200),
                target_size=(224, 224),
                freeze_backbone= self.lora_rank_vision_encoder > 0,
                eval_fixed_crop=False,
                normalize_images=False, # normalized outside this wrapper
            )
            if self.lora_rank_vision_encoder > 0:
                # lora the vision encoder 
                lora_config = LoraConfig(
                    r=self.lora_rank_vision_encoder,
                    lora_alpha=self.lora_rank_vision_encoder,
                    target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
                )
                
                self.vision_encoder.backbone = get_peft_model(self.vision_encoder.backbone, lora_config)
        else:
            # fallback toconstruct ResNet18 encoder
            # if you have multiple camera views, use seperate encoder weights for each view.
            self.vision_encoder = get_resnet('resnet18')

            # IMPORTANT!
            # replace all BatchNorm with GroupNorm to work with EMA
            # performance will tank if you forget to do this!
            self.vision_encoder = replace_bn_with_gn(self.vision_encoder)

        if self.s2:
            print("using scaling on scale. make sure dataset is 448 x 448")
            self.vision_encoder = S2ResNet(self.vision_encoder)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.vision_feature_dim = self.vision_feature_dim * 2
            # currently we use adaptive pooling to make it work with previous vision feature dimension

        # observation feature has 514 dims in total per step
        self.only_vision = model_cfg.policy_cfg.only_vision
        if self.only_vision:
            obs_dim_t = self.vision_feature_dim * self.num_cameras
        else:
            obs_dim_t = self.vision_feature_dim * self.num_cameras + self.lowdim_obs_dim

        # create network object
        self.global_cond_dim = obs_dim_t * self.obs_horizon

        if self.pred_left_only or self.pred_right_only:
            self.action_dim = self.action_dim // 2
            print("pred_left_only or pred_right_only is enabled, changed action dim to ", self.action_dim)

        # for this demo, we use DDPMScheduler with 100 diffusion iterations
        self.num_diffusion_iters = model_cfg.policy_cfg.num_train_diffusion_steps
        self.prediction_type='epsilon'
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type=self.prediction_type
        )

        # create diffusion model
        if self.diffusion_model_type == "unet":
            # UNet-based diffusion model
            print("using unet for diffusion")
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=self.action_dim,
                down_dims=model_cfg.policy_cfg.down_dims,
                global_cond_dim=self.global_cond_dim
            )
        elif self.diffusion_model_type == "transformer":
            print("using transformer for diffusion")
            # Transformer-based diffusion model
            self.noise_pred_net = TransformerForDiffusion(
                input_dim=self.action_dim,
                output_dim=self.action_dim,
                horizon=self.action_horizon,
                n_obs_steps=self.obs_horizon,
                cond_dim=obs_dim_t,
                n_layer=model_cfg.policy_cfg.transformer_n_layer,
                n_head=model_cfg.policy_cfg.transformer_n_head,
                n_emb=model_cfg.policy_cfg.transformer_n_emb,
                p_drop_emb=model_cfg.policy_cfg.transformer_p_drop_emb,
                p_drop_attn=model_cfg.policy_cfg.transformer_p_drop_attn,
                causal_attn=model_cfg.policy_cfg.transformer_causal_attn,
                time_as_cond=model_cfg.policy_cfg.transformer_time_as_cond,
                obs_as_cond=model_cfg.policy_cfg.transformer_obs_as_cond,
                n_cond_layers=model_cfg.policy_cfg.transformer_n_cond_layers
            )
        elif self.diffusion_model_type == "scale_DP":
            print("using scale_DP for diffusion")
            self.noise_pred_net = ScaleTransformerDiffusionPolicy(
                input_dim=self.action_dim,
                output_dim=self.action_dim,
                horizon=self.action_horizon,
                n_obs_steps=self.obs_horizon,
                cond_dim=obs_dim_t,
                depth=model_cfg.policy_cfg.transformer_n_layer,
                n_head=model_cfg.policy_cfg.transformer_n_head,
                n_emb=model_cfg.policy_cfg.transformer_n_emb,
                p_drop_emb=model_cfg.policy_cfg.transformer_p_drop_emb,
                causal_attn=model_cfg.policy_cfg.transformer_causal_attn,
            )

    def _process_images(self, nimage: torch.Tensor, vision_transform) -> torch.Tensor:
        """
        Process images through vision transform and reshape.
        
        Args:
            nimage: Input images of shape (B, T, num_cameras, C, H, W)
            vision_transform: Function to transform images
            
        Returns:
            Processed images of shape (B, T, num_cameras, C, H, W)
        """
        B, T = nimage.shape[:2]
        # Reshape for vision transform
        nimage = nimage.reshape(B*T*self.num_cameras, *nimage.shape[3:])
        if nimage.shape[1] != 3:
            nimage = nimage.permute(0, 3, 1, 2)  # T, C, H, W
        nimage = vision_transform(nimage).float()
        # Reshape back to original format
        nimage = nimage.reshape(B, T, self.num_cameras, *nimage.shape[1:])
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
        
        if self.s2:
            # S2 approach
            image_features = multiscale_forward(self.vision_encoder, nimage.flatten(end_dim=2), scales=[0.5, 1], max_split_size=224)
            batch_size, T, _ = image_features.shape
            res = int(np.sqrt(T))
            image_features = image_features.transpose(1, 2).view(batch_size, self.vision_feature_dim, res, res)
            image_features = self.pool(image_features).squeeze()
            image_features = image_features.reshape(*nimage.shape[:3],-1).reshape(*nimage.shape[:2],-1)
        else:
            # Traditional ResNet approach
            image_features = self.vision_encoder(nimage.flatten(end_dim=2))
            image_features = image_features.reshape(*nimage.shape[:3],-1).view(*nimage.shape[:2],-1)
        
        return image_features

    def _process_batch(self, nbatch, vision_transform=None):
        """
        Process a batch of data for inference.
        
        Args:
            nbatch: Batch dictionary containing:
                - observation: Images of shape (B, T, num_cameras, C, H, W)
                - proprio: Proprioceptive data of shape (B, T, D)
            vision_transform: Optional function to transform images
            
        Returns:
            Tuple of (processed_images, proprioceptive_data, batch_size)
        """
        # Process images
        nimage = nbatch["observation"][:, :self.obs_horizon, :self.num_cameras]
        if vision_transform is not None:
            nimage = self._process_images(nimage, vision_transform)
        
        # Get proprioceptive data
        nagent_pos = nbatch["proprio"][:, :self.obs_horizon]
        batch_size = nagent_pos.shape[0]
        
        return nimage, nagent_pos, batch_size

    def _get_observation_features(self, nimage, nagent_pos):
        """
        Get observation features by combining vision and proprioceptive features.
        
        Args:
            nimage: Processed images
            nagent_pos: Proprioceptive data
            
        Returns:
            Observation features ready for model input
        """
        # Get vision features
        image_features = self._get_vision_features(nimage)
        
        # Combine vision and proprio features
        if not self.only_vision:
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        else:
            obs_features = image_features
        
        return obs_features

    def _build_xt_and_target_flow_rect(self, x):
        """
        Rectified Flow: x_t = (1-t) z + t x, target v* = x - z
        x: (B, T, D)
        returns x_t, v_target, t
        """
        B = x.shape[0]
        z = torch.randn_like(x)
        # sample continuous t ~ U(0,1); broadcast to (B, 1, 1) so it works for (B,T,D)
        t = torch.rand(B, 1, 1, device=x.device)
        x_t = (1.0 - t) * z + t * x
        v_target = x - z
        return x_t, v_target, t

    def _build_xt_and_target_flow_gauss(self, x):
        """
        Gaussian-path Flow Matching (optional): reuse your scheduler but compute derivatives.
        Requires continuous t in [0,1]. We'll map t->[0, num_train_timesteps-1] for embeddings,
        but derive alpha, sigma, and their derivatives analytically (implement your schedule here).
        """
        B = x.shape[0]
        z = torch.randn_like(x)
        t = torch.rand(B, 1, 1, device=x.device)

        # Implement your schedule analytically here if you want Gaussian FM.
        # Example placeholders (MUST be replaced with your true alpha(t), sigma(t) and derivatives):
        # alpha_bar = ...
        # alpha = torch.sqrt(alpha_bar)
        # sigma = torch.sqrt(1 - alpha_bar)
        # alpha_dot = 0.5 * alpha_bar_dot / alpha
        # sigma_dot = -0.5 * alpha_bar_dot / sigma

        raise NotImplementedError("flow_gauss path: implement alpha(t),sigma(t), and derivatives or use flow_rect.")
    
    @torch.no_grad()
    def _fm_sample_heun(self, obs_features, B, steps=16):
        """
        Heun's method (2nd order) to integrate dx/dt = v_theta(x,t,c) from t=0->1.
        obs_features: (B, obs_horizon, obs_dim_t)   # same as you pass during training
        Returns: x at t=1, i.e., predicted actions (B, T, D)
        """
        # init x(0) ~ N(0,I)
        x = torch.randn((B, self.action_horizon, self.action_dim), device=obs_features.device)

        # time grid
        t_grid = torch.linspace(0., 1., steps + 1, device=x.device)
        dt = t_grid[1] - t_grid[0]

        for i in range(steps):
            ti = t_grid[i].expand(B)                        # (B,)
            tip1 = t_grid[i+1].expand(B)                    # (B,)

            # discretize for time embeddings (same trick as in training)
            ti_idx = (ti * (self.noise_scheduler.config.num_train_timesteps - 1)).long()
            tip1_idx = (tip1 * (self.noise_scheduler.config.num_train_timesteps - 1)).long()

            if self.diffusion_model_type == "unet":
                k1 = self.noise_pred_net(x, ti_idx, global_cond=obs_features.flatten(start_dim=1))   # (B,T,D)
                x_euler = x + dt * k1
                k2 = self.noise_pred_net(x_euler, tip1_idx, global_cond=obs_features.flatten(start_dim=1))
            else:
                k1 = self.noise_pred_net(x, ti_idx, cond=obs_features)   # (B,T,D)
                x_euler = x + dt * k1
                k2 = self.noise_pred_net(x_euler, tip1_idx, cond=obs_features)

            x = x + 0.5 * dt * (k1 + k2)

        return x

    def forward_loss(self, nbatch):
        nimage = nbatch["observation"][:, :, :self.num_cameras] # pick the first image # B, T, self.num_cameras, C, H ,W
        nagent_pos = nbatch["proprio"] # pick the current proprio # B, T, D
        if self.pred_right_only:
            naction = nbatch['action'][:, -1, ..., self.action_dim:]
        else:
            naction = nbatch['action'][:, -1, ..., :self.action_dim] # B, N, D # dataset returns action for all T steps. We only need the last one
        B = nagent_pos.shape[0]

        # encoder vision features
        if self.s2: 
            image_features = multiscale_forward(self.vision_encoder, nimage.flatten(end_dim=2), scales=[0.5, 1], max_split_size=224) # torch.Size([32, 49, 1024])
            batch_size, T, _ = image_features.shape
            res = int(np.sqrt(T))
            image_features = image_features.transpose(1, 2).view(batch_size, self.vision_feature_dim, res, res)
            image_features = self.pool(image_features).squeeze()
        else:
            # pre flatten shape [128, 1, 3, 3, 224, 224] resnet example 128 batch size
            # post flatten shape [384, 3, 224, 224] resnet example 128 batch size
            image_features = self.vision_encoder(nimage.flatten(end_dim=2)) # to incorporate num_camera dimension

            # image_features.shape [384, 512] resnet example 128 batch size
        # convert first to (B, obs_horizon, self.num_cameras, D) then flatten to (B, obs_horizon, self.num_cameras * D)
        image_features = image_features.reshape(*nimage.shape[:3],-1).reshape(*nimage.shape[:2],-1)

        # concatenate vision feature and low-dim obs
        if not self.only_vision:
            # print("image_features.shape", image_features.shape)
            # print("nagent_pos.shape", nagent_pos.shape)
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        else:
            obs_features = image_features
        obs_cond = obs_features.flatten(start_dim=1)
        # naction = naction.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        if self.objective == "diffusion":
            # ------- ORIGINAL DIFFUSION -------
            noise = torch.randn(naction.shape, device=naction.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (B,), device=naction.device
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

            if self.diffusion_model_type == "unet":
                pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
            else:
                pred = self.noise_pred_net(noisy_actions, timesteps, cond=obs_features)

            target = noise  # diffusion target

        else:
            # ------- FLOW MATCHING (Rectified Flow by default) -------
            if self.objective == "flow_rect":
                x_t, v_target, t_cont = self._build_xt_and_target_flow_rect(naction)  # (B, T, D), (B, T, D), (B,1,1)
            elif self.objective == "flow_gauss":
                x_t, v_target, t_cont = self._build_xt_and_target_flow_gauss(naction)
            else:
                raise ValueError(f"Unknown objective: {self.objective}")

            # Your backbones expect a 'timesteps' input (int). We can discretize for embeddings:
            # (this only affects the time embedding; the target remains correct)
            t_idx = (t_cont.squeeze(-1).squeeze(-1) * (self.noise_scheduler.config.num_train_timesteps - 1)).long()

            if self.diffusion_model_type == "unet":
                pred = self.noise_pred_net(x_t, t_idx, global_cond=obs_cond)   # predict velocity
            else:
                pred = self.noise_pred_net(x_t, t_idx, cond=obs_features)      # predict velocity

            target = v_target  # flow-matching target

        # ------- shared loss weighting (grippers etc.) -------
        loss = nn.functional.mse_loss(pred, target, reduction='none')
        if self.pred_left_only or self.pred_right_only:
            loss[..., -1] *= self.gripper_loss_w
        else:
            loss[..., self.action_dim // 2 - 1] *= self.gripper_loss_w
            loss[..., -1] *= self.gripper_loss_w
        loss = loss.mean()

        return loss

    def forward(self, nbatch):
        # if self.training: # this is commented out to circumvent the validation loss calculation
        return self.forward_loss(nbatch)
        # else:
            # raise NotImplementedError("Inference not implemented yet")

    # def forward_inference(self, nbatch, vision_transform=None):
    #     nimage, nagent_pos, B = self._process_batch(nbatch, vision_transform)
    #     obs_features = self._get_observation_features(nimage, nagent_pos)

    #     if self.objective == "diffusion":
    #         # ---- original diffusion sampling (unchanged) ----
    #         obs_cond = obs_features.flatten(start_dim=1)
    #         if self.diffusion_model_type == "unet":
    #             noisy_actions = torch.randn((B, self.action_horizon, self.action_dim), device=obs_cond.device)
    #             for t in self.noise_scheduler.timesteps:
    #                 noise_pred = self.noise_pred_net(noisy_actions, t, global_cond=obs_cond)
    #                 noisy_actions = self.noise_scheduler.step(noise_pred, t, noisy_actions).prev_sample
    #             pred_action = noisy_actions
    #         else:
    #             noisy_actions = torch.randn((B, self.action_horizon, self.action_dim), device=obs_cond.device)
    #             for t in self.noise_scheduler.timesteps:
    #                 noise_pred = self.noise_pred_net(noisy_actions, t, cond=obs_features)
    #                 noisy_actions = self.noise_scheduler.step(noise_pred, t, noisy_actions).prev_sample
    #             pred_action = noisy_actions
    #         return pred_action

    #     else:
    #         # ---- Flow Matching ODE sampling ----
    #         # For UNet we passed flattened global cond during training; mirror that here.
    #         # The helper handles both UNet and Transformer internally.
    #         pred_action = self._fm_sample_heun(obs_features, B, steps=getattr(self, "fm_nfe", 16))
    #         return pred_action

    def forward_inference(self, nbatch, vision_transform=None):
        """
        Inference method for the diffusion policy.
        
        Args:
            nbatch: Batch dictionary containing:
                - observation: Images of shape (B, T, num_cameras, C, H, W)
                - proprio: Proprioceptive data of shape (B, T, D)
            vision_transform: Optional function to transform images
            
        Returns:
            Predicted actions of shape (B, action_horizon, action_dim)
        """
        # Process batch
        import time
        t0 = time.perf_counter()
        nimage, nagent_pos, B = self._process_batch(nbatch, vision_transform)
        t1 = time.perf_counter()
        print("process_batch time", t1 - t0)
        
        t0 = time.perf_counter()
        # Get observation features
        obs_features = self._get_observation_features(nimage, nagent_pos)
        t1 = time.perf_counter()
        print("get_observation_features time", t1 - t0)
        
        t0 = time.perf_counter()
        # For transformer, we need to pass the observation features directly
        obs_cond = obs_features.flatten(start_dim=1)
        t1 = time.perf_counter()
        print("flatten obs_features time", t1 - t0)
        
        t0 = time.perf_counter()
        # Initialize with random noise
        if self.diffusion_model_type == "unet":
            # For UNet, we need to reshape to (B, T, D)
            noisy_actions = torch.randn(
                (B, self.action_horizon, self.action_dim), 
                device=obs_cond.device
            )
            
            # Denoise step by step
            for t in self.noise_scheduler.timesteps:
                # Predict noise residual
                noise_pred = self.noise_pred_net(noisy_actions, t, global_cond=obs_cond)
                
                # Compute previous noisy sample x_t -> x_t-1
                noisy_actions = self.noise_scheduler.step(
                    noise_pred, t, noisy_actions
                ).prev_sample
            
            # Final prediction
            pred_action = noisy_actions
        else:
            # For transformer, we need to reshape to (B, T, D)
            noisy_actions = torch.randn(
                (B, self.action_horizon, self.action_dim), 
                device=obs_cond.device
            )
            
            # Denoise step by step
            for t in self.noise_scheduler.timesteps:
                # Predict noise residual
                t0 = time.perf_counter()
                noise_pred = self.noise_pred_net(noisy_actions, t, cond=obs_features)
                t1 = time.perf_counter()
                print("one step noise_pred time", t1 - t0)
                # Compute previous noisy sample x_t -> x_t-1
                noisy_actions = self.noise_scheduler.step(
                    noise_pred, t, noisy_actions
                ).prev_sample
            
            # Final prediction
            pred_action = noisy_actions
        t1 = time.perf_counter()
        print("denoise time", t1 - t0)
        print(len(self.noise_scheduler.timesteps))

        return pred_action

    def __repr__(self):
        return f"DiffusionPolicy(action_dim={self.action_dim}, obs_horizon={self.obs_horizon}, action_horizon={self.action_horizon}, diffusion_model_type={self.diffusion_model_type})"
    
class Dinov2DiscretePolicy(nn.Module):
    lowdim_obs_dim : int = 20 
    action_dim : int = 20
    def __init__(
        self,
        model_cfg: ModelConfig,
        shared_cfg: SharedConfig,
        *, # This forces the following arguments to be keyword-only
        vocab_size: int = 1024,
        num_tokens: int = 50
    ):
        super().__init__()
        self.obs_horizon = shared_cfg.seq_length
        self.pred_left_only = model_cfg.policy_cfg.pred_left_only
        self.pred_right_only = model_cfg.policy_cfg.pred_right_only
        self.num_cameras = shared_cfg.num_cameras
        
        if self.pred_left_only or self.pred_right_only:
            self.action_dim = self.action_dim // 2
        else:
            self.action_dim = self.action_dim
        self.action_horizon = shared_cfg.num_pred_steps

        # Create DINOv2 backbone
        self.timm_vision_encoder = model_cfg.policy_cfg.timm_vision_encoder
        self.lora_rank_vision_encoder = model_cfg.policy_cfg.lora_rank_vision_encoder
        if self.timm_vision_encoder is None:
            self.timm_vision_encoder = "vit_base_patch14_reg4_dinov2.lvd142m"
            print("defaulting to dinov2 backbone: ", self.timm_vision_encoder)

        self.vision_encoder = timm.create_model(
            self.timm_vision_encoder, # vit_base_patch14_reg4_dinov2.lvd142m for register tokens
            pretrained=True,
            num_classes=0,
            global_pool="",
        )

        # freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        if self.lora_rank_vision_encoder > 0:
            # lora the vision encoder 
            lora_config = LoraConfig(
                r=self.lora_rank_vision_encoder,
                lora_alpha=self.lora_rank_vision_encoder,
                target_modules=["qkv"]
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)
            
        self.embed_dim = self.vision_encoder.embed_dim + self.lowdim_obs_dim
        
        # Calculate input feature dimension
        self.num_patches = self.vision_encoder.patch_embed.num_patches + 1
        # TODO check if this includes registedr tokens
        
        # Project proprio features to same dimension as vision features
        self.proprio_proj = nn.Linear(self.lowdim_obs_dim, self.embed_dim)
        
        # Discrete action decoder
        self.decoder = DiscreteActionDecoder(
            decoder_config=model_cfg.action_decoder_cfg,
            encoder_hidden_size=self.embed_dim,  # All features are projected to embed_dim
            vocab_size=vocab_size,
            num_tokens=num_tokens,
        )
        print(self)

    def _get_encoder_features(self, nimage : torch.Tensor, nagent_pos: torch.Tensor):
        B = nagent_pos.shape[0]
        # Process images through DINO-v2
        flat_imgs = nimage.flatten(start_dim=0, end_dim=2)  # (B*T*num_cameras, C, H, W)
        if self.lora_rank_vision_encoder > 0:
            dino_out = self.vision_encoder.forward_features(flat_imgs) # (B*T*num_cameras, self.num_patches, dino.embed_dim)
        else:
            # freeze dino 
            with torch.no_grad():
                dino_out = self.vision_encoder.forward_features(flat_imgs) # (B*T*num_cameras, self.num_patches, dino.embed_dim)

        dino_out = dino_out.view(B, -1, self.num_cameras, self.num_patches, self.vision_encoder.embed_dim) # (B, T, num_cameras, self.num_patches, dino.embed_dim)

        # append nagent_pos to dino_out
        nagent_pos = nagent_pos.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.num_cameras, self.num_patches, -1) # (B, T, num_cameras, D)
        dino_out = torch.cat([dino_out, nagent_pos], dim=-1) # (B, T, num_cameras, num_patches, embed_dim)

        # reshape dino_out to B, T*num_cameras*num_patches, embed_dim+D
        encoder_features = dino_out.view(B, -1, self.embed_dim) # (B, T*num_cameras*num_patches, embed_dim+D)
        return encoder_features

    def forward(self, nbatch, return_acc : bool = True):
        """
        Process full sequence of images and proprio data, then predict action
        """
        nimage = nbatch["observation"][:, :, :self.num_cameras]  # (B, T, num_cameras, C, H, W)
        nagent_pos = nbatch["proprio"]  # (B, T, D)
        ntarget = nbatch['action']  # (B, num_tokens), this is taken care off by the collate function
        target_mask = nbatch['action_mask']  # (B, num_tokens)

        # Get encoder features
        encoder_features = self._get_encoder_features(nimage, nagent_pos)

        # Calculate loss
        return self.decoder.loss(encoder_features, ntarget, target_mask, return_acc)

    def forward_inference(self, nbatch, vision_transform=None):
        """
        Inference method for the discrete policy.
        
        Args:
            nbatch: Batch dictionary containing:
                - observation: Images of shape (B, T, num_cameras, C, H, W)
                - proprio: Proprioceptive data of shape (B, T, D)
            vision_transform: Optional function to transform images
            
        Returns:
            Predicted token sequences
        """
        # Process batch
        nimage, nagent_pos, _ = self._process_batch(nbatch, vision_transform)
        
        # Get encoder features
        encoder_features = self._get_encoder_features(nimage, nagent_pos)
        
        # Get predictions
        predictions = self.decoder.pred(encoder_features)
        return predictions
    
    def __repr__(self):
        """
        Return a string representation of the Dinov2DiscretePolicy model.
        """
        width = 70  # Slightly narrower for better display in notebooks/console
        border = "=" * width
        section_border = "-" * width
        
        # Collect all information
        model_info = [
            ("Model Architecture", [
                ("Vision Encoder", self.timm_vision_encoder),
                ("Num Cameras", self.num_cameras),
                ("Seq Length", self.obs_horizon),
                ("Action Horizon", self.action_horizon)
            ]),
            ("Dimensions", [
                ("Proprio Input", self.lowdim_obs_dim),
                ("Action Dim", self.action_dim),
                ("Vision Pool Out (per camera)", self.vision_encoder.embed_dim),
            ]),
            ("Parameters", [
                ("Total", format_number(count_parameters(self))),
                ("Vision Encoder", format_number(count_parameters(self.vision_encoder))),
                ("Vision Encoder Lora", format_number(count_parameters(self.vision_encoder, trainable_only=True))),
                ("Policy", format_number(count_parameters(self.decoder))),
                ("Trainable", format_number(sum(p.numel() for p in self.parameters() if p.requires_grad)))
            ])
        ]

        # Build the string representation
        lines = [
            border,
            f"{'Dinov2 Discrete Policy Model':^{width}}",
            border,
            ""
        ]

        for section_title, items in model_info:
            lines.append(f"{section_title:^{width}}")
            lines.append(section_border)
            
            # Find maximum length for alignment
            max_key_length = max(len(k) for k, _ in items)
            
            for key, value in items:
                # Add each line with proper spacing
                lines.append(f"{key:<{max_key_length}} : {value:>}")
            lines.append("")

        lines.append(border)

        # Join all lines with newlines
        return "\n".join(lines)
        
    