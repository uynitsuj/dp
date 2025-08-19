from typing import Optional

import torch
from torch import nn


class ModelEMA:
    """
    Model Exponential Moving Average (EMA)
    
    Args:
        model: The model to create EMA for
        decay: EMA decay rate (default: 0.9999)
        device: Device to store EMA parameters on
    """
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None
    ):
        self.ema = {}
        self.ema_decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create EMA for each parameter
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema[name] = param.data.clone().detach().to(self.device)
    
    def update(self, model: nn.Module) -> None:
        """
        Update EMA parameters
        
        Args:
            model: The model to update EMA from
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.ema[name] = self.ema[name] * self.ema_decay + param.data * (1 - self.ema_decay)
    
    def apply_ema(self, model: nn.Module) -> None:
        """
        Apply EMA parameters to model
        
        Args:
            model: The model to apply EMA to
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.ema[name])
    
    def state_dict(self) -> dict:
        """
        Get EMA state dict
        
        Returns:
            Dictionary containing EMA parameters
        """
        return {name: param.clone().detach() for name, param in self.ema.items()}
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load EMA state dict
        
        Args:
            state_dict: Dictionary containing EMA parameters
        """
        for name, param in state_dict.items():
            if name in self.ema:
                self.ema[name].copy_(param) 