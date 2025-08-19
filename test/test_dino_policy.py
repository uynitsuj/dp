import torch
from dp_gs.policy.model import Dinov2DiscretePolicy
from dp_gs.util.args import ExperimentConfig
from tqdm import trange
import tyro
import time

def create_dummy_input(batch_size, seq_length, num_cameras, image_size, proprio_dim, vocab_size, max_tokens, device):
    """
    Create dummy input data with variable length sequences and masks
    """
    # Create dummy input data
    dummy_images = torch.randn(batch_size, seq_length, num_cameras, 3, image_size, image_size).float().to(device)
    dummy_proprio = torch.randn(batch_size, seq_length, proprio_dim).float().to(device)
    
    # Create variable length sequences
    dummy_target = []
    dummy_mask = []
    
    for i in range(batch_size):
        # Random sequence length between 14 and max_tokens-1 (leaving room for EOS)
        seq_len = torch.randint(14, max_tokens-1, (1,)).item()
        
        # Create sequence of random tokens
        seq = torch.randint(0, vocab_size-2, (seq_len,))  # vocab_size-1 to leave room for EOS
        
        # Add EOS token
        seq = torch.cat([seq, torch.tensor([vocab_size - 1])])
        
        # Pad to max_tokens
        padded_seq = torch.full((max_tokens,), vocab_size - 1, dtype=torch.long)
        padded_seq[:len(seq)] = seq
        
        # Create mask
        mask = torch.zeros(max_tokens, dtype=torch.bool)
        mask[:len(seq)] = True  # True for actual tokens including EOS
        
        dummy_target.append(padded_seq)
        dummy_mask.append(mask)
    
    # Stack and move to device
    dummy_target = torch.stack(dummy_target).to(device)
    dummy_mask = torch.stack(dummy_mask).to(device)
    
    # Create batch dictionary
    batch = {
        "observation": dummy_images,
        "proprio": dummy_proprio,
        "action": dummy_target,
        "action_mask": dummy_mask
    }
    return batch

def test_dino_policy(exp_config: ExperimentConfig):
    # Test configurations
    batch_size = 32
    seq_length = 4
    num_cameras = 2
    image_size = 518
    num_tokens = 100
    vocab_size = 2048
    proprio_dim = 20
    device = "cuda:1"
    
    batch = create_dummy_input(batch_size, seq_length, num_cameras, image_size, proprio_dim, vocab_size, num_tokens, device)
    
    # Initialize model
    model = Dinov2DiscretePolicy(
        model_cfg=exp_config.model_cfg,
        shared_cfg=exp_config.shared_cfg,
        vocab_size=vocab_size,
        num_tokens=num_tokens
    ).to(device)
    model.num_cameras = num_cameras  # Set num_cameras attribute
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1_000_000:.2f}M")
    print(f"Number of trainable parameters: {num_trainable_params/1_000_000:.2f}M")
    
    # Test forward pass and loss calculation
    out = model.forward(batch, return_acc=True)

    # unpack 
    loss, acc = out["loss"], out["acc"]
    
    # Basic checks
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    
    # Test backward pass
    loss.backward()
    
    print("All tests passed!")
    print(f"Loss value: {loss.item():.4f}")
    print(f"Accuracy: {acc:.4f}")   

    # let's evaluate the average inference speed of the model 
    model = model.eval()
    model = torch.compile(model)
    num_samples = 100
    start = time.time()
    batch = create_dummy_input(1, seq_length, num_cameras, image_size, proprio_dim, vocab_size, num_tokens, device)
    with torch.no_grad():
        for _ in trange(num_samples):
            iter_start = time.time()
            model.forward_inference(batch)
            print("Inference speed: ", time.time()-iter_start)
    end = time.time()
    print(f"Average inference speed: {(end-start)/num_samples:.4f} seconds")

if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    test_dino_policy(args)
