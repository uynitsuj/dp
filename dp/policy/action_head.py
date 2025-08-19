import abc
from typing import Dict, Union, Optional, Literal 
import torch
import torch.nn as nn
import torch.distributions as D
from timm.layers import Mlp
from torch.nn import functional as F
from dp_gs.util.args import ActionDecoderConfig
from .transformer import TransformerBlock, CrossAttentionBlock

class PredHead(abc.ABC, nn.Module):
    """
    Abstract class for prediction head
    """
    @abc.abstractmethod
    def forward(self, x : torch.Tensor) -> Union[torch.Tensor, D.Distribution]:
        """
        Forward pass of the prediction head
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            Union[torch.Tensor, D.Distribution], (B, output_dim) or D.Distribution
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pred(self, x : torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the prediction head
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the prediction head
        Args:
            x: torch.Tensor, (B, input_dim)
            y: torch.Tensor, (B, output_dim)
        Returns:
            torch.Tensor, scalar
        """
        raise NotImplementedError
    
class DiscreteActionDecoder(PredHead):
    def __init__(
        self,
        decoder_config: ActionDecoderConfig,
        encoder_hidden_size: int,  # Renamed from in_feature_dim
        vocab_size: int = 1024,
        num_tokens: int = 50,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.hidden_size = decoder_config.hidden_size
        
        # Learnable masked tokens that will be decoded into actions
        self.masked_tokens = nn.Parameter(torch.randn(1, num_tokens, decoder_config.hidden_size))
        # initialize with he uniform 
        nn.init.kaiming_uniform_(self.masked_tokens)
        
        # Remove encoder projection since CrossAttention handles different dimensions
        
        # Cross attention blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_config, encoder_hidden_size)
            for _ in range(decoder_config.num_hidden_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(decoder_config.hidden_size, vocab_size)
        
        # Keep only final layer norm, remove encoder layer norm
        self.final_ln = nn.LayerNorm(decoder_config.hidden_size)

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: Tensor of shape (B, N, D) where:
                B is batch size
                N is number of encoder tokens (T * num_cameras * patches + T * proprio_dim)
                D is encoder feature dimension
        Returns:
            logits: Tensor of shape (B, num_tokens, vocab_size)
        """
        B = encoder_features.shape[0]
        
        # Expand masked tokens for batch
        decoder_states = self.masked_tokens.expand(B, -1, -1)  # (B, num_tokens, hidden_size)
        
        # Pass through cross attention blocks
        for block in self.blocks:
            decoder_states = block(
                decoder_states,
                encoder_features,
            )
        
        # Final layer norm and projection
        decoder_states = self.final_ln(decoder_states)
        logits = self.output_proj(decoder_states)  # (B, num_tokens, vocab_size)
        
        return logits

    def loss(
        self, 
        encoder_features: torch.Tensor, 
        targets: torch.Tensor, 
        target_mask: torch.Tensor,
        return_acc: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_features: Tensor of shape (B, N, D)
            targets: Tensor of shape (B, num_tokens) containing target token indices
            target_mask: Tensor of shape (B, num_tokens) containing True for valid positions
        """
        logits = self.forward(encoder_features)  # (B, num_tokens, vocab_size)
        
        # Reshape for loss calculation
        flat_logits = logits.view(-1, self.vocab_size)  # (B*num_tokens, vocab_size)
        flat_targets = targets.view(-1)  # (B*num_tokens)
        flat_mask = target_mask.view(-1)  # (B*num_tokens)
        
        # Calculate loss only on valid positions
        loss = F.cross_entropy(
            flat_logits[flat_mask],
            flat_targets[flat_mask],
            reduction='mean'
        )
        
        result = {'loss': loss}
        
        if return_acc:
            acc_dict = self.calculate_acc(
                logits,
                targets,
                target_mask,
                eos_token_id=self.vocab_size - 1
            )
            result.update(acc_dict)
            
        return result

    def calculate_acc(
        self, 
        logits : torch.Tensor,
        ground_truth_targets : torch.Tensor,
        target_mask : torch.Tensor, 
        eos_token_id : int,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate accuracy and EOS token accuracy
        Args:
            logits: Tensor of shape (B, num_action_tokens, vocab_size)
            ground_truth_targets: Tensor of shape (B, num_action_tokens)
            target_mask: Tensor of shape (B, num_action_tokens)
        Returns:
            Dict[str, torch.Tensor], {'acc': accuracy, 'eos_acc': EOS token accuracy}
        """
        result = {}
        argmax_pred = torch.argmax(logits, dim=-1)
        # Calculate accuracy only on valid positions
        correct = (argmax_pred == ground_truth_targets) & target_mask
        acc = correct.float().sum() / target_mask.float().sum()
        result['acc'] = acc

        num_action_tokens = target_mask.shape[1]
        # Calculate EOS token prediction accuracy
        # Get position of last True in target_mask for each batch
        last_true_positions = target_mask.int().argmin(dim=1) - 1 # (B, ) - 1 because we want the last True
        # make sure it is not -1 
        last_true_positions = torch.where(last_true_positions == -1, num_action_tokens - 1, last_true_positions)

        # Check if EOS token (vocab_size-1) is predicted at those positions
        eos_correct = argmax_pred[torch.arange(len(last_true_positions)), last_true_positions] == eos_token_id
        result['eos_acc'] = eos_correct.float().mean()
        return result

    def pred(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Returns predictions and stops at first EOS token during inference
        """
        logits = self.forward(encoder_features)
        predictions = torch.argmax(logits, dim=-1)  # (B, num_tokens)
        
        # Find the first EOS token in each sequence
        eos_positions = (predictions == self.vocab_size - 1).int().argmax(dim=1)
        
        # Create a mask that's True up to and including the first EOS token
        mask = torch.arange(predictions.size(1), device=predictions.device)[None, :] <= eos_positions[:, None]
        
        # Apply the mask - replace everything after EOS with EOS token
        predictions = torch.where(mask, predictions, self.vocab_size - 1)
        
        return predictions