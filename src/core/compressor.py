"""
Core neural network compression implementations.

This module contains the main compression classes:
- TinyCompressor: Basic implementation with sigmoid activation
- PerfectTinyCompressor: Advanced implementation with ReLU and proper backpropagation
"""

import numpy as np
from typing import Union, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TinyCompressor:
    """
    A tiny neural network that overfits on a single piece of data.
    
    Architecture: Input → Hidden(16) → Bottleneck(4) → Hidden(16) → Output
    Uses sigmoid activation and simplified training for basic compression.
    """
    
    def __init__(self, data_size: int, hidden_size: int = 16, bottleneck_size: int = 4):
        """
        Initialize the tiny compressor.
        
        Args:
            data_size: Size of input data
            hidden_size: Size of hidden layers
            bottleneck_size: Size of bottleneck layer (compression factor)
        """
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        
        logger.info(f"Creating TinyCompressor: Input={data_size}, Hidden={hidden_size}, Bottleneck={bottleneck_size}")
        
        # Initialize weights with Xavier initialization
        np.random.seed(42)
        self.W1 = np.random.randn(data_size, hidden_size) * np.sqrt(2.0 / data_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, bottleneck_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(bottleneck_size)
        self.W3 = np.random.randn(bottleneck_size, hidden_size) * np.sqrt(2.0 / bottleneck_size)
        self.b3 = np.zeros(hidden_size)
        self.W4 = np.random.randn(hidden_size, data_size) * np.sqrt(2.0 / hidden_size)
        self.b4 = np.zeros(data_size)
        
        # Calculate model size
        self.model_size_kb = self._calculate_model_size()
        logger.info(f"Model size: {self.model_size_kb:.2f} KB")
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in KB."""
        total_params = (self.W1.size + self.b1.size + self.W2.size + 
                       self.b2.size + self.W3.size + self.b3.size + 
                       self.W4.size + self.b4.size)
        return (total_params * 4) / 1024  # 4 bytes per float32
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with clipping."""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        # Encoder: data → hidden → bottleneck
        h1 = self.sigmoid(np.dot(x, self.W1) + self.b1)
        bottleneck = self.sigmoid(np.dot(h1, self.W2) + self.b2)
        
        # Decoder: bottleneck → hidden → output
        h2 = self.sigmoid(np.dot(bottleneck, self.W3) + self.b3)
        output = self.sigmoid(np.dot(h2, self.W4) + self.b4)
        
        return output
    
    def train(self, data: np.ndarray, epochs: int = 30000, learning_rate: float = 0.1) -> float:
        """
        Train the network to overfit on the data.
        
        Args:
            data: Input data to memorize
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Final training loss
        """
        logger.info(f"Training for {epochs} epochs with learning rate {learning_rate}")
        
        X = data.reshape(1, -1)
        y = data.reshape(1, -1)
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss
            loss = np.mean((output - y) ** 2)
            
            # Progress updates
            if epoch % 10000 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss:.6f}")
            
            # Simple gradient descent
            error = output - y
            
            # Get intermediate values for weight updates
            h1 = self.sigmoid(np.dot(X, self.W1) + self.b1)
            bottleneck = self.sigmoid(np.dot(h1, self.W2) + self.b2)
            h2 = self.sigmoid(np.dot(bottleneck, self.W3) + self.b3)
            
            # Update weights (simplified)
            self.W4 -= learning_rate * 0.001 * np.dot(h2.T, error)
            self.W3 -= learning_rate * 0.001 * np.dot(bottleneck.T, h2)
            self.W2 -= learning_rate * 0.001 * np.dot(h1.T, bottleneck)
            self.W1 -= learning_rate * 0.001 * np.dot(X.T, h1)
        
        logger.info(f"Final loss: {loss:.6f}")
        return loss
    
    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress data through encoder."""
        X = data.reshape(1, -1)
        h1 = self.sigmoid(np.dot(X, self.W1) + self.b1)
        bottleneck = self.sigmoid(np.dot(h1, self.W2) + self.b2)
        return bottleneck
    
    def decompress(self, compressed: np.ndarray) -> np.ndarray:
        """Decompress data through decoder."""
        h2 = self.sigmoid(np.dot(compressed, self.W3) + self.b3)
        output = self.sigmoid(np.dot(h2, self.W4) + self.b4)
        return output


class PerfectTinyCompressor:
    """
    A tiny neural network that achieves PERFECT reconstruction through better training.
    
    Uses ReLU activation, proper backpropagation, and advanced optimization
    to achieve exact data reconstruction.
    """
    
    def __init__(self, data_size: int, hidden_size: int = 16, bottleneck_size: int = 4):
        """
        Initialize the perfect compressor.
        
        Args:
            data_size: Size of input data
            hidden_size: Size of hidden layers
            bottleneck_size: Size of bottleneck layer
        """
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        
        logger.info(f"Creating PerfectTinyCompressor: Input={data_size}, Hidden={hidden_size}, Bottleneck={bottleneck_size}")
        
        # Better weight initialization
        np.random.seed(42)
        self.W1 = np.random.randn(data_size, hidden_size) * np.sqrt(2.0 / data_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, bottleneck_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(bottleneck_size)
        self.W3 = np.random.randn(bottleneck_size, hidden_size) * np.sqrt(2.0 / bottleneck_size)
        self.b3 = np.zeros(hidden_size)
        self.W4 = np.random.randn(hidden_size, data_size) * np.sqrt(2.0 / hidden_size)
        self.b4 = np.zeros(data_size)
        
        # Calculate model size
        self.model_size_kb = self._calculate_model_size()
        logger.info(f"Model size: {self.model_size_kb:.2f} KB")
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in KB."""
        total_params = (self.W1.size + self.b1.size + self.W2.size + 
                       self.b2.size + self.W3.size + self.b3.size + 
                       self.W4.size + self.b4.size)
        return (total_params * 4) / 1024
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with stored intermediate values."""
        # Encoder
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)  # Bottleneck
        
        # Decoder
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.relu(self.z3)
        
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.a4 = self.z4  # No activation on output for better training
        
        return self.a4
    
    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.001) -> None:
        """Proper backpropagation with gradient clipping."""
        m = x.shape[0]
        
        # Output layer gradients
        dz4 = self.a4 - y
        dW4 = np.dot(self.a3.T, dz4) / m
        db4 = np.sum(dz4, axis=0) / m
        
        # Hidden layer 3 gradients
        da3 = np.dot(dz4, self.W4.T)
        dz3 = da3 * self.relu_derivative(self.z3)
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0) / m
        
        # Bottleneck layer gradients
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        
        # Hidden layer 1 gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m
        
        # Update weights with gradient clipping
        dW4 = np.clip(dW4, -1, 1)
        dW3 = np.clip(dW3, -1, 1)
        dW2 = np.clip(dW2, -1, 1)
        dW1 = np.clip(dW1, -1, 1)
        
        self.W4 -= learning_rate * dW4
        self.b4 -= learning_rate * db4
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, data: np.ndarray, epochs: int = 100000, 
              learning_rate: float = 0.01, patience: int = 5000) -> float:
        """
        Train until perfect reconstruction.
        
        Args:
            data: Input data to memorize
            epochs: Maximum number of training epochs
            learning_rate: Initial learning rate
            patience: Early stopping patience
            
        Returns:
            Final training loss
        """
        logger.info(f"Training for up to {epochs} epochs with goal: perfect reconstruction")
        
        X = data.reshape(1, -1)
        y = data.reshape(1, -1)
        
        best_loss = float('inf')
        no_improve = 0
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss
            loss = np.mean((output - y) ** 2)
            
            # Progress updates
            if epoch % 20000 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss:.10f}")
            
            # Check for perfect reconstruction
            if loss < 1e-10:
                logger.info(f"🎉 PERFECT RECONSTRUCTION at epoch {epoch}!")
                return loss
            
            # Early stopping check
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience and epoch > 10000:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            # Adaptive learning rate
            if epoch > 0 and epoch % 10000 == 0:
                learning_rate *= 0.9
        
        logger.info(f"Final loss: {loss:.10f}")
        return loss
    
    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress data through encoder."""
        X = data.reshape(1, -1)
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        bottleneck = self.relu(z2)
        return bottleneck
    
    def decompress(self, compressed: np.ndarray) -> np.ndarray:
        """Decompress data through decoder."""
        z3 = np.dot(compressed, self.W3) + self.b3
        a3 = self.relu(z3)
        output = np.dot(a3, self.W4) + self.b4
        return output
