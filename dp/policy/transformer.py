import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dp.util.args import ActionDecoderConfig

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension {dim} should be divisible by 2")
            
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Create position indices for the sequence length
        pos = torch.arange(max_seq_len).float()
        sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer("emb", emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.emb[:seq_len, :]  # [seq_len, dim]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, pos_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [batch, n_heads, seq_len, head_dim]
    # pos_emb: [seq_len, head_dim]
    seq_len = q.shape[2]
    head_dim = q.shape[3]
    
    # Ensure pos_emb has the right sequence length
    pos_emb = pos_emb[:seq_len, :]
    
    # Split into real and imaginary parts
    cos = pos_emb[..., :head_dim//2]  # [seq_len, head_dim//2]
    sin = pos_emb[..., head_dim//2:]  # [seq_len, head_dim//2]
    
    # Reshape for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    
    # Split q and k into real and imaginary parts
    q1, q2 = q[..., :head_dim//2], q[..., head_dim//2:]
    k1, k2 = k[..., :head_dim//2], k[..., head_dim//2:]
    
    # Apply rotation
    q_out = torch.cat([
        q1 * cos - q2 * sin,
        q2 * cos + q1 * sin
    ], dim=-1)
    
    k_out = torch.cat([
        k1 * cos - k2 * sin,
        k2 * cos + k1 * sin
    ], dim=-1)
    
    return q_out, k_out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config : ActionDecoderConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} is not divisible by number of attention heads {config.num_attention_heads}"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.rope = RotaryPositionalEmbedding(self.attention_head_size, config.max_position_embeddings)
        
        self.dropout = config.attention_probs_dropout_prob
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Apply RoPE to query and key
        pos_emb = self.rope(hidden_states)  # [seq_len, head_dim]
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, pos_emb)

        # Convert attention mask to proper format if provided
        if attention_mask is not None:
            # attention_mask should be broadcastable to [batch, num_heads, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]

        # Use torch's scaled_dot_product_attention
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return self.out(context_layer)
    

class TransformerBlock(nn.Module):
    def __init__(self, config : ActionDecoderConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention_output = self.attention(
            self.layernorm1(hidden_states),
            attention_mask
        )
        hidden_states = hidden_states + self.dropout(attention_output)

        layer_output = self.layernorm2(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.activation(layer_output)
        layer_output = self.output(layer_output)
        
        return hidden_states + self.dropout(layer_output)

class CrossAttention(nn.Module):
    def __init__(
        self, 
        config: ActionDecoderConfig,
        encoder_hidden_size: int,  # Added parameter for encoder dimension
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} is not divisible by number of attention heads {config.num_attention_heads}"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Query comes from decoder hidden states (config.hidden_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # Key and Value come from encoder hidden states (encoder_hidden_size)
        self.key = nn.Linear(encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(encoder_hidden_size, self.all_head_size)
        
        # RoPE is applied only to query since cross-attention operates on different sequences
        self.rope = RotaryPositionalEmbedding(self.attention_head_size, config.max_position_embeddings)
        
        self.dropout = config.attention_probs_dropout_prob
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,  # Decoder hidden states
        encoder_hidden_states: torch.Tensor,  # Encoder hidden states
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Project decoder hidden states to query
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        # Project encoder hidden states to key and value
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))

        # Apply RoPE only to query (decoder positions)
        pos_emb = self.rope(hidden_states)  # [seq_len, head_dim]
        query_layer, _ = apply_rotary_pos_emb(query_layer, query_layer, pos_emb)  # Only apply to query

        # Convert encoder attention mask to proper format if provided
        if encoder_attention_mask is not None:
            # encoder_attention_mask should be broadcastable to [batch, num_heads, seq_len, encoder_seq_len]
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # Use torch's scaled_dot_product_attention
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=encoder_attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return self.out(context_layer)

class CrossAttentionBlock(nn.Module):
    def __init__(
        self, 
        config: ActionDecoderConfig,
        encoder_hidden_size: int,  # Added parameter
    ):
        super().__init__()
        # Self attention layer (operates on decoder hidden states)
        self.self_attention = MultiHeadAttention(config)
        # Cross attention layer
        self.cross_attention = CrossAttention(config, encoder_hidden_size)
        # Feed forward layer
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Layer norms
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.layernorm3 = nn.LayerNorm(config.hidden_size)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention
        self_attention_output = self.self_attention(
            self.layernorm1(hidden_states),
            attention_mask
        )
        hidden_states = hidden_states + self.dropout(self_attention_output)

        # Cross attention
        cross_attention_output = self.cross_attention(
            self.layernorm2(hidden_states),
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask
        )
        hidden_states = hidden_states + self.dropout(cross_attention_output)

        # Feed forward
        layer_output = self.layernorm3(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.activation(layer_output)
        layer_output = self.output(layer_output)
        
        return hidden_states + self.dropout(layer_output)
