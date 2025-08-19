import torch
from dp_gs.policy.action_head import DiscreteActionDecoder
from dp_gs.util.args import ActionDecoderConfig, ExperimentConfig
from dataclasses import dataclass
import tyro 

def test_discrete_action_decoder(decoder_config : ActionDecoderConfig):
    # Test configurations
    batch_size = 4
    seq_length = 4
    feature_dim = 512
    num_tokens = 50
    vocab_size = 1024
    
    # Create dummy input
    x = torch.randn(batch_size, seq_length, feature_dim)
    target = torch.randint(0, vocab_size, (batch_size, seq_length, num_tokens))
    
    # Initialize model
    model = DiscreteActionDecoder(
        decoder_config=decoder_config,
        in_feature_dim=feature_dim,
        vocab_size=vocab_size,
        num_tokens=num_tokens
    )
    
    # print number of parameters of this model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params/1_000_000:.2f}M")

    # Test forward pass
    logits = model.forward(x)
    assert logits.shape == (batch_size, seq_length, num_tokens, vocab_size), \
        f"Expected shape {(batch_size, seq_length, num_tokens, vocab_size)}, got {logits.shape}"
    
    # Test loss calculation
    loss = model.loss(x, target)
    loss.backward()
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss should not be NaN"
    
    # Test prediction
    predictions = model.pred(x)
    assert predictions.shape == (batch_size, seq_length, num_tokens), \
        f"Expected shape {(batch_size, seq_length, num_tokens)}, got {predictions.shape}"
    assert torch.all(predictions >= 0) and torch.all(predictions < vocab_size), \
        "Predictions should be within vocabulary range"

if __name__ == "__main__":
    args = tyro.cli(ActionDecoderConfig)
    test_discrete_action_decoder(args)
    print("All tests passed!")
