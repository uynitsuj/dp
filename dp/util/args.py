import dataclasses
from typing import List, Literal, Optional, Tuple

from dp.util import transforms as _transforms


@dataclasses.dataclass(frozen=True)
class DatasetConfig: 
    # Dataset root path 
    dataset_root : Optional[str] = None

    # Data transforms
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
        
    # subsample_data [0.0, 1.0]
    data_subsample_ratio : float = -1

    # subsample number of  trajectories 
    data_subsample_num_traj : int = -1

    # scale action by statistics 
    scale_action : bool = True

    # Dataset config path 
    dataset_json : Optional[str] = None

    # action statistics json path 
    action_statistics : Optional[str] = None

    # train split 
    train_split : float = 0.95

    # add vision data augmentation
    vision_aug : bool = True

    # subsample steps 
    subsample_steps : int = 1

@dataclasses.dataclass
class VisionEncoderConfig: 
    # vision encoder type (or path to checkpoint)
    vision_encoder : str = "vit_base_patch16_224.mae"

    # Whether to use a randomly initialized vision encoder instead of pretrained weights
    vision_nonpretrained : bool = False 

    # Whether to unfreeze the vision encoder
    vision_unfreeze_all : bool = False 

    # Whether to use LoRA in the vision encoder 
    vision_lora : bool = False

    # Rank of LoRA layers
    vision_lora_rank : int = 8 

    # Number of blocks unfrozen in the vision encoder 
    vision_unfreeze_last_n : int = 0 

@dataclasses.dataclass
class PolicyConfig: 
    # down_dims
    down_dims : Tuple[int, ...] = (256, 512, 1024)

    # pred left only 
    pred_left_only : bool = False
    
    # pred right only 
    pred_right_only : bool = False

    # use timm vision encoder 
    timm_vision_encoder : Optional[str] = None  # "vit_base_patch14_dinov2.lvd142m"

    # use other vision encoder 
    vision_encoder_pretrained_type : Optional[str] = "dinov3" # options: "dinov3"

    # lora vision encoder 
    lora_rank_vision_encoder : int = 2

    # only vision as input 
    only_vision : bool = False

    # path to LLaMA pretrained checkpoint
    llama_ckpt_dir : str = "/home/mfu/checkpoints/llama-2/llama-2-7b"

    # uses different linear layers for attention pooling 
    multikv_attn_pool : bool = False

    # Number of heads for adapter
    adapter_num_heads : int = 8 

    # Adapter MLP ratio 
    adapter_mlp_ratio : float = 4.0 

    # Weight for action loss 
    loss_w_action : float = 1.0 
    
    # add camera positional embeddings
    camera_pos_emb : bool = False 

    # add modality positional embeddings
    modality_pos_emb : bool = False

    # Separate encoder adapter for different cameras
    separate_camera_adapter : bool = True 

    # Rank of LoRA layers for Llama 
    lora_rank : int = 4 

    # Layer indices to apply LoRA
    lora_layer_idxs : Optional[Tuple[int, ...]] = None 

    # Phase of training 
    phase : Literal["pretrain", "finetune"] = "pretrain"

    # path to checkpoint from pretrain stage
    pretrained_path : Optional[str] = None

    # predict only action 
    pred_action_only : bool = True

    # remove proprio from input 
    remove_proprio : bool = False

    # no loss on prompt
    no_prompt_loss : bool = True

    # Prediction head, can be one of "mlp", "gmm", "diffusion"
    decoder_pred_head : Literal["mlp", "gmm", "diffusion", "discrete"] = "diffusion"

    # Allowed: "diffusion" (current), "flow_rect" (Rectified Flow), "flow_gauss" (Gaussian-path FM)
    diffusion_objective : Literal["diffusion", "flow_rect", "flow_gauss"] = "flow_rect"

    # use kl div loss
    kl_div_loss : bool = False

    # loss scaler 
    scale_loss : float = 1.0
    
    # load llama
    load_llama : bool = True
    
    # step weight
    step_weight : float = 1.0

    # train llama from scratch
    scratch_llama_config : Optional[str] = None

    # training diffusion steps:
    num_train_diffusion_steps : Optional[int] = 100

    # inference diffusion steps
    num_inference_diffusion_steps : Optional[int] = 100 

    # enable or disable action as input to the transformer 
    action_as_input : bool = True

    # compile or not 
    compile : bool = False
    
    # diffusion model type: "unet" or "transformer"
    diffusion_model_type : Literal["unet", "transformer", "scale_DP"] = "scale_DP"
    
    # transformer specific parameters
    transformer_n_layer : int = 28
    transformer_n_head : int = 16
    transformer_n_emb : int = 1024
    transformer_p_drop_emb : float = 0.1
    transformer_p_drop_attn : float = 0.2
    transformer_causal_attn : bool = True
    transformer_time_as_cond : bool = True
    transformer_obs_as_cond : bool = True
    transformer_n_cond_layers : int = 0
    
    # vision model type for simple policy: "resnet" or "dino"
    simple_vision_model : Literal["resnet", "dino"] = "resnet"
    
    # use attention pooling for simple policy
    simple_attention_pooling : bool = False

    # gripper loss weight (simple policy)
    gripper_loss_w : float = 1.0 
    
@dataclasses.dataclass
class PreferenceLearningConfig: 
    # enable preference learning 
    enable : bool = False

    # reference policy path 
    ref_policy_path : Optional[str] = None

    # beta (dpo / ipo)
    beta : float = 0.1

    # sigma (fixed spread for gaussian for laplace distribution)
    sigma : float = 0.01

    # label smoothing
    label_smoothing : float = 0.0

    # DPO or IPO 
    opt_algo : Literal["dpo", "ipo"] = "dpo"

    # preference loss type 
    loss_type : Literal["l1", "l2"] = "l1"

    # reference free preference learning (do not use reference policy)
    reference_free : bool = False

    # weights applied on positive trajectory diff (icrt loss)
    w_regress_loss : float = 1.0

@dataclasses.dataclass
class ActionDecoderConfig:
    # hidden size
    hidden_size : int = 1024

    # number of attention heads
    num_attention_heads : int = 16

    # intermediate size (transformer expansion)
    intermediate_size : int = 4096
    
    # number of transformer blocks
    num_hidden_layers : int = 12

    # max position embeddings
    max_position_embeddings : int = 512

    # attention dropout probability
    attention_probs_dropout_prob : int = 0.0

    # dropout probability
    hidden_dropout_prob : int = 0.0

@dataclasses.dataclass
class ModelConfig: 
    # Policy (llama + adapter) configuration
    policy_cfg : PolicyConfig = dataclasses.field(default_factory=PolicyConfig)

    # vision encoder config 
    vision_encoder_cfg : VisionEncoderConfig = dataclasses.field(default_factory=VisionEncoderConfig)

    # action decoder config (experimental)
    action_decoder_cfg : ActionDecoderConfig = dataclasses.field(default_factory=ActionDecoderConfig)

    # preference learning config 
    pref_cfg : PreferenceLearningConfig = dataclasses.field(default_factory=PreferenceLearningConfig)

    # policy type 
    policy_type : Literal["simple", "diffusion", "discrete"] = "diffusion"

@dataclasses.dataclass
class OptimizerConfig: 
    # weight decay (default: 0.01) 
    weight_decay : float = 0.01

    # learning rate (absolute lr)
    lr : Optional[float] = 1e-4

    # base learning rate: absolute_lr = base_lr * total_batch_size / 256 
    blr : float = 1e-3 

    # lower lr bound for cyclic schedulers that hit 0
    min_lr : float = 0.0 

    # epochs to warmup LR
    warmup_epochs : float = 40

@dataclasses.dataclass
class TrainerConfig:
    # number of epochs 
    epochs : int = 300

    # Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
    accum_iter : int = 1

    # pin memory for dataloader
    pin_memory : bool = True

    # number of workers for dataloader 
    num_workers : int = 20 
    # num_workers : int = 0 # for debugging

    # number of augmentation workers for DALI
    num_augmentation_workers : int = 20
    
    # whether to augment with gpu
    augment_with_gpu : bool = False
    
@dataclasses.dataclass
class SharedConfig: # TODO: migrate some of these to train config and squash some args
    # Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus
    batch_size : int = 128

    # number of frames in a sequence 
    seq_length : int = 1

    # seed for random number generators
    seed : int = 0
    
    # start epoch 
    start_epoch : int = 0

    # frequency of saving checkpoint 
    save_every : int = 5

    # resume from checkpoint 
    resume : Optional[str] = None 

    # Number of stages for progressive training 
    num_stages : int = 1

    # split epoch into k different epochs 
    split_epoch : int = 1 

    # Number of cameras 
    num_cameras : int = 3

    # Number of predicted action steps 
    num_pred_steps : int = 40

    # camera keys 
    camera_keys : List[str] = dataclasses.field(default_factory=lambda: ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"])

    # scale action with calculated action statistics (json file)
    scale_action : Optional[str] = None

    # use scaling on scale
    s2 : bool = False

    # image size (Dino: need to change to 518)
    image_size : int = 224

    # number of tokens to predict 
    num_tokens : int = 108

@dataclasses.dataclass
class LoggingConfig:
    # path where to save, empty for no saving
    output_dir: str = "./output"

    # path where to save tensorboard logs 
    log_dir : Optional[str] = None 

    # log name (for wandb)
    log_name : Optional[str] = None

# @dataclasses.dataclass
# class ExperimentConfig: 
#     # Dataset configuration
#     dataset_cfg: DatasetConfig

#     # Model configuration
#     model_cfg: ModelConfig

#     # Optimizer configuration
#     optimizer_cfg: OptimizerConfig

#     # Shared configuration
#     shared_cfg: SharedConfig

#     # Logging configuration 
#     logging_cfg: LoggingConfig

#     # trainer configuration
#     trainer_cfg: TrainerConfig

#     # train or eval 
#     train : bool = True

#     # number of distributed processes (required by torch distributed)
#     world_size: int = 1

#     # local rank of the process (required by torch distributed)
#     local_rank: int = -1

#     # distributed training on the optimizer (required by torch distributed)
#     dist_on_itp: bool = False

#     # distributed training url (required by torch distributed)
#     dist_url: str = 'env://'

#     # device to use for training / testing (required by torch distributed)
#     device : str = "cuda"

#     # load config. instead of using command line arguments, load from a config file
#     load_config: Optional[str] = None

@dataclasses.dataclass
class InferenceConfig:
    # path to model checkpoint
    model_ckpt_folder: str = "/home/justinyu/nfs_us/justinyu/dp/250819_0305"
    ckpt_id : int = 205

# if __name__ == "__main__": 
#     args = tyro.cli(ExperimentConfig)
#     dict_args = dataclasses.asdict(args)