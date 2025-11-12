"""
Neural network architecture configurations for compression.

This module defines different network architectures that can be used
for neural network compression, allowing easy experimentation with
different layer sizes and configurations.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class AutoencoderArchitecture:
    """
    Configuration for autoencoder neural network architecture.
    
    This class defines the structure of the compression network,
    including layer sizes and activation functions.
    """
    
    # Input/output dimensions
    input_size: int
    
    # Hidden layer configurations
    encoder_layers: List[int]  # List of hidden layer sizes for encoder
    bottleneck_size: int       # Size of bottleneck (compression factor)
    decoder_layers: List[int]  # List of hidden layer sizes for decoder
    
    # Training parameters
    learning_rate: float = 0.01
    activation: str = "relu"   # "relu", "sigmoid", "tanh"
    
    # Optional parameters
    dropout_rate: Optional[float] = None
    batch_norm: bool = False
    
    def __post_init__(self):
        """Validate architecture configuration."""
        if self.input_size <= 0:
            raise ValueError("Input size must be positive")
        
        if self.bottleneck_size <= 0:
            raise ValueError("Bottleneck size must be positive")
        
        if not all(size > 0 for size in self.encoder_layers):
            raise ValueError("All encoder layer sizes must be positive")
        
        if not all(size > 0 for size in self.decoder_layers):
            raise ValueError("All decoder layer sizes must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.activation not in ["relu", "sigmoid", "tanh"]:
            raise ValueError("Activation must be one of: relu, sigmoid, tanh")
    
    def get_total_parameters(self) -> int:
        """Calculate total number of parameters in the network."""
        total = 0
        
        # Encoder layers
        prev_size = self.input_size
        for layer_size in self.encoder_layers:
            total += prev_size * layer_size + layer_size  # weights + biases
            prev_size = layer_size
        
        # Bottleneck layer
        total += prev_size * self.bottleneck_size + self.bottleneck_size
        
        # Decoder layers
        prev_size = self.bottleneck_size
        for layer_size in self.decoder_layers:
            total += prev_size * layer_size + layer_size
            prev_size = layer_size
        
        # Output layer
        total += prev_size * self.input_size + self.input_size
        
        return total
    
    def get_model_size_kb(self) -> float:
        """Calculate model size in KB (assuming float32 parameters)."""
        return (self.get_total_parameters() * 4) / 1024
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio compared to original data."""
        original_size = self.input_size * 4  # bytes for float32
        compressed_size = self.get_model_size_kb() * 1024  # bytes
        return original_size / compressed_size
    
    def is_efficient(self, target_size_kb: float = 10.0) -> bool:
        """Check if model size is under target threshold."""
        return self.get_model_size_kb() <= target_size_kb
    
    def get_layer_sizes(self) -> List[int]:
        """Get all layer sizes in order."""
        return [self.input_size] + self.encoder_layers + [self.bottleneck_size] + self.decoder_layers + [self.input_size]


# Predefined architectures
def get_tiny_architecture(input_size: int) -> AutoencoderArchitecture:
    """Get the tiny architecture used in the original implementation."""
    return AutoencoderArchitecture(
        input_size=input_size,
        encoder_layers=[16],
        bottleneck_size=4,
        decoder_layers=[16],
        learning_rate=0.1,
        activation="sigmoid"
    )


def get_perfect_architecture(input_size: int) -> AutoencoderArchitecture:
    """Get the perfect architecture with ReLU activation."""
    return AutoencoderArchitecture(
        input_size=input_size,
        encoder_layers=[16],
        bottleneck_size=4,
        decoder_layers=[16],
        learning_rate=0.01,
        activation="relu"
    )


def get_deep_architecture(input_size: int, 
                         hidden_layers: List[int] = [32, 16, 8],
                         bottleneck_size: int = 4) -> AutoencoderArchitecture:
    """Get a deeper architecture for more complex data."""
    return AutoencoderArchitecture(
        input_size=input_size,
        encoder_layers=hidden_layers,
        bottleneck_size=bottleneck_size,
        decoder_layers=list(reversed(hidden_layers)),
        learning_rate=0.001,
        activation="relu"
    )


def get_ultra_compressed_architecture(input_size: int) -> AutoencoderArchitecture:
    """Get ultra-compressed architecture with minimal bottleneck."""
    return AutoencoderArchitecture(
        input_size=input_size,
        encoder_layers=[8],
        bottleneck_size=2,
        decoder_layers=[8],
        learning_rate=0.05,
        activation="sigmoid"
    )
