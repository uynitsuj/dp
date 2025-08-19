from typing import Union
from dp_gs.util.args import ModelConfig, SharedConfig, VisionEncoderConfig, PolicyConfig, PreferenceLearningConfig
from dp_gs.models.backbones.encoders import VisionEncoder, VisionEncoderCNN
from dp_gs.models.policy.icrt import ICRT
from dp_gs.models.policy.picrt import PICRT

def vision_encoder_cond(
    vision_encoder_cfg : VisionEncoderConfig
) -> Union[str, int, bool]:
    if vision_encoder_cfg.vision_unfreeze_all:
        vision_finetune = "all"

    elif vision_encoder_cfg.vision_unfreeze_last_n > 0:
        vision_finetune = vision_encoder_cfg.vision_unfreeze_last_n
    
    elif vision_encoder_cfg.vision_lora:
        vision_finetune = "lora"
    
    else:
        vision_finetune = False
    
    return vision_finetune

def vision_encoder_constructor(
    vision_encoder_cfg : VisionEncoderConfig,
) -> Union[VisionEncoder, VisionEncoderCNN]: 
    """Instantiate a vision encoder based on the vision_encoder_cfg 

    Return the vision encoder instance
    """
    vision_finetune = vision_encoder_cond(vision_encoder_cfg)
    
    vision_pretrained = not vision_encoder_cfg.vision_nonpretrained

    vision_encoder_name = vision_encoder_cfg.vision_encoder
    print(f"using vision encoder {vision_encoder_name}")
    vision_encoder_cls = VisionEncoderCNN if "resnet" in vision_encoder_name.lower() else VisionEncoder
    vision_encoder = vision_encoder_cls(
        vision_encoder_name, pretrained=vision_pretrained, 
        global_pool="", finetune=vision_finetune, lora_rank=vision_encoder_cfg.vision_lora_rank,
    )

    return vision_encoder

def policy_constructor(
    policy_cfg : PolicyConfig, 
    shared_config : SharedConfig,
    vision_encoder : Union[VisionEncoder, VisionEncoderCNN], 
    train : bool = True, 
    preference_training : bool = False,
) -> ICRT: 
    """
    Instantiate a policy based on the policy_cfg
    """
    # proprio and action dim 
    proprio_dim = 10 if shared_config.rot_6d else 8
    action_dim = 11 if shared_config.rot_6d else 8 

    max_batch_size = shared_config.batch_size
    if preference_training:
        max_batch_size = 2 * max_batch_size

    model = ICRT(
        llama_ckpt_dir=policy_cfg.llama_ckpt_dir, 
        vision_encoder=vision_encoder,
        phase=policy_cfg.phase, 
        num_cameras=shared_config.num_cameras, 
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        adapter_mlp_ratio=policy_cfg.adapter_mlp_ratio, 
        adapter_num_heads=policy_cfg.adapter_num_heads,
        multikv_attn_pool=policy_cfg.multikv_attn_pool,
        loss_w_action=policy_cfg.loss_w_action,    
        lora_rank=policy_cfg.lora_rank,
        camera_pos_emb=policy_cfg.camera_pos_emb,
        modality_pos_emb=policy_cfg.modality_pos_emb,
        separate_camera_adapter=policy_cfg.separate_camera_adapter, 
        seq_length=shared_config.seq_length,
        rot_6d=shared_config.rot_6d,
        train=train,
        max_batch_size=max_batch_size,
        num_pred_steps=shared_config.num_pred_steps,
        pred_action_only=policy_cfg.pred_action_only,
        remove_proprio=policy_cfg.remove_proprio,
        no_prompt_loss=policy_cfg.no_prompt_loss,
        decoder_pred_head=policy_cfg.decoder_pred_head,
        use_delta_action=shared_config.use_delta_action,
        kl_div_loss=policy_cfg.kl_div_loss,
        scale_loss=policy_cfg.scale_loss,
        load_llama=policy_cfg.load_llama,
        step_weight=policy_cfg.step_weight,
        scratch_llama_config=policy_cfg.scratch_llama_config,
        num_train_diffusion_steps=policy_cfg.num_train_diffusion_steps,
        num_inference_diffusion_steps=policy_cfg.num_inference_diffusion_steps,
        scale_action=shared_config.scale_action,
        action_as_input=policy_cfg.action_as_input,
        compile=policy_cfg.compile,
    )
    return model 

def model_constructor(
    model_config : ModelConfig, 
    shared_config : SharedConfig,
    train : bool = True, 
) -> Union[ICRT, PICRT]:
    if model_config.pref_cfg.enable:
        return dpo_model_constructor(model_config, shared_config, train=train)
    else:
        return vanilla_model_constructor(model_config, shared_config, train=train)

def vanilla_model_constructor(
    model_config : ModelConfig, 
    shared_config : SharedConfig,
    train : bool = True, 
) -> ICRT:
    vision_encoder = vision_encoder_constructor(model_config.vision_encoder_cfg)
    policy = policy_constructor(model_config.policy_cfg, shared_config, vision_encoder, train=train)
    return policy

def preference_policy_constructor(
    pref_learning_cfg : PreferenceLearningConfig, 
    policy : ICRT, 
    reference_policy : ICRT,
): 
    model = PICRT(
        policy, 
        reference_policy, 
        reference_policy_path=pref_learning_cfg.ref_policy_path, 
        beta=pref_learning_cfg.beta,
        sigma=pref_learning_cfg.sigma,
        label_smoothing=pref_learning_cfg.label_smoothing,
        ipo=pref_learning_cfg.opt_algo == "ipo",
        reference_free=pref_learning_cfg.reference_free,
        loss_type=pref_learning_cfg.loss_type,
        w_regress_loss=pref_learning_cfg.w_regress_loss,
    )
    return model

def dpo_model_constructor(
    model_config : ModelConfig, 
    shared_config : SharedConfig,
    train : bool = True, 
) -> PICRT:
    assert model_config.pref_cfg.ref_policy_path is not None, "Reference policy path must be provided for preference learning"
    vision_finetune = vision_encoder_cond(model_config.vision_encoder_cfg)
    vision_encoder = vision_encoder_constructor(model_config.vision_encoder_cfg)
    if vision_finetune:
        vision_encoder_ref = vision_encoder_constructor(model_config.vision_encoder_cfg)
    else:
        vision_encoder_ref = vision_encoder
    policy = policy_constructor(model_config.policy_cfg, shared_config, vision_encoder, train=train, preference_training=True)
    reference_model = policy_constructor(model_config.policy_cfg, shared_config, vision_encoder_ref, train=train, preference_training=True)
    preference_policy = preference_policy_constructor(model_config.pref_cfg, policy, reference_model)
    return preference_policy