import torch
import tyro
from tqdm import trange
import time
from dp_gs.policy.model import SimplePolicy
from dp_gs.util.args import ExperimentConfig

def create_dummy_input(batch_size, seq_length, num_cameras, image_size, action_dim, action_horizon):
    """
    Create dummy input data for testing the SimplePolicy.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        num_cameras: Number of cameras
        image_size: Image size (height, width)
        action_dim: Action dimension
        action_horizon: Action horizon
        
    Returns:
        Dictionary containing dummy input data
    """
    # Create dummy image data (B, T, num_cameras, C, H, W)
    images = torch.randn(batch_size, seq_length, num_cameras, 3, image_size[0], image_size[1])
    
    # Create dummy proprioceptive data (B, T, D)
    proprio_dim = 20  # Standard proprioceptive dimension
    proprio = torch.randn(batch_size, seq_length, proprio_dim)
    
    # Create dummy action data (B, T, D)
    actions = torch.randn(batch_size, seq_length, action_horizon, action_dim)
    
    return {
        "observation": images,
        "proprio": proprio,
        "action": actions
    }

def test_simple_policy(config : ExperimentConfig):
    """
    Test the SimplePolicy with different vision encoders and attention pooling options.
    
    Args:
        config: ExperimentConfig object
    """
    # Set up test parameters
    batch_size = 2
    seq_length = config.shared_cfg.seq_length
    num_cameras = config.shared_cfg.num_cameras
    image_size = (config.shared_cfg.image_size, config.shared_cfg.image_size)
    action_dim = 20
    action_horizon = config.shared_cfg.num_pred_steps
    
    # Test combinations
    # vision_models = ["dino"]
    vision_models = ["resnet", "dino"]
    # attention_pooling_options = [True]
    attention_pooling_options = [False, True]
    
    for vision_model in vision_models:
        for use_attention_pooling in attention_pooling_options:
            print(f"\nTesting SimplePolicy with vision_model={vision_model}, attention_pooling={use_attention_pooling}")
            
            # Update config
            config.model_cfg.policy_cfg.simple_vision_model = vision_model
            config.model_cfg.policy_cfg.simple_attention_pooling = use_attention_pooling
            
            # Create model
            model = SimplePolicy(config.model_cfg, config.shared_cfg)
            print(model)
            
            # Create dummy input
            dummy_input = create_dummy_input(
                batch_size, seq_length, num_cameras, image_size, action_dim, action_horizon
            )
            
            # Test forward pass
            print("Testing forward pass...")
            loss = model(dummy_input)
            assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
            assert loss.dim() == 0, "Loss should be a scalar"
            assert not torch.isnan(loss), "Loss should not be NaN"
            print(f"Loss: {loss.item():.4f}")
            
            # Test backward pass
            print("Testing backward pass...")
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"Parameter {name} should have gradients"
                    assert not torch.isnan(param.grad).any(), f"Gradients for {name} should not be NaN"
            print("Backward pass successful")
            
            # Test inference
            print("Testing inference...")
            model.eval()
            with torch.no_grad():
                pred_action = model.forward_inference(dummy_input)
                ratio = 2 if config.model_cfg.policy_cfg.pred_right_only or config.model_cfg.policy_cfg.pred_left_only else 1
                expected_shape = (batch_size, action_horizon, action_dim // ratio)
                assert tuple(pred_action.shape) == expected_shape, \
                    f"Expected shape {expected_shape}, got {tuple(pred_action.shape)}"
            print("Inference successful")
            
            # Test inference speed
            print("Testing inference speed...")
            num_samples = 100
            batch_sizes = [1, 4]
            
            for bs in batch_sizes:
                dummy_input = create_dummy_input(
                    bs, seq_length, num_cameras, image_size, action_dim, action_horizon
                )
                
                # Warm up
                for _ in range(5):
                    with torch.no_grad():
                        _ = model.forward_inference(dummy_input)
                
                # Measure inference time
                start_time = time.time()
                for _ in trange(num_samples // bs, desc=f"Batch size {bs}"):
                    with torch.no_grad():
                        _ = model.forward_inference(dummy_input)
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                samples_per_second = num_samples / elapsed_time
                print(f"Batch size {bs}: {samples_per_second:.2f} samples/second")
            
            print(f"Test completed for vision_model={vision_model}, attention_pooling={use_attention_pooling}")

if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig) 
    test_simple_policy(args)