"""
Generative compressor that takes a simple input (like 1) and generates the entire data.

This is a single neural network that overfits to generate the exact data we want
when given a simple input like the number 1.
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GenerativeCompressor:
    """
    A neural network that takes a simple input (e.g., 1) and generates the entire data.
    
    Architecture: Input(1) → Hidden → Output(data_size)
    This allows us to store data as weights in a model that generates it.
    """
    
    def __init__(self, data_size: int, hidden_size: int = 4, target_size_kb: float = 1.0):
        """
        Initialize the generative compressor.
        
        Args:
            data_size: Size of output data to generate
            hidden_size: Size of hidden layer (adjusted to meet target size)
            target_size_kb: Target model size in KB
        """
        self.data_size = data_size
        self.target_size_kb = target_size_kb
        self.target_params = int(target_size_kb * 1024 / 4)  # float32 = 4 bytes
        
        # Calculate optimal hidden size to meet target
        # Model params = (1 * hidden + hidden) + (hidden * data_size + data_size)
        # = hidden + hidden + hidden*data_size + data_size
        # = 2*hidden + hidden*data_size + data_size
        # For 1KB target with data_size, solve for hidden:
        # 2*hidden + hidden*data_size + data_size <= target_params
        # hidden*(2 + data_size) <= target_params - data_size
        # hidden <= (target_params - data_size) / (2 + data_size)
        
        max_hidden = int((self.target_params - data_size) / (2 + data_size))
        if max_hidden < 1:
            max_hidden = 1
            logger.warning(f"Cannot fit {data_size} output in {target_size_kb}KB model. Using minimal hidden size.")
        
        # Use the calculated hidden size or the provided one, whichever is smaller
        self.hidden_size = min(hidden_size, max_hidden)
        
        logger.info(f"Creating GenerativeCompressor: Input=1, Hidden={self.hidden_size}, Output={data_size}")
        
        # Initialize weights
        np.random.seed(42)
        # Input(1) → Hidden
        self.W1 = np.random.randn(1, self.hidden_size) * 0.1
        self.b1 = np.zeros(self.hidden_size)
        
        # Hidden → Output(data_size)
        self.W2 = np.random.randn(self.hidden_size, data_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(data_size)
        
        # Calculate actual model size
        self.model_size_kb = self._calculate_model_size()
        logger.info(f"Model size: {self.model_size_kb:.2f} KB (target: {target_size_kb:.2f} KB)")
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in KB."""
        total_params = (self.W1.size + self.b1.size + self.W2.size + self.b2.size)
        return (total_params * 4) / 1024
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with clipping."""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def forward(self, input_val: float = 1.0) -> np.ndarray:
        """
        Forward pass: generate data from simple input.
        
        Args:
            input_val: Input value (default 1.0)
            
        Returns:
            Generated data array
        """
        x = np.array([[input_val]])
        # Input → Hidden
        h = self.sigmoid(np.dot(x, self.W1) + self.b1)
        # Hidden → Output
        output = self.sigmoid(np.dot(h, self.W2) + self.b2)
        return output.flatten()
    
    def train(self, target_data: np.ndarray, epochs: int = 100000, 
              learning_rate: float = 0.1, input_val: float = 1.0) -> float:
        """
        Train the network to generate target_data when given input_val.
        
        Args:
            target_data: The data we want the model to generate
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            input_val: Input value to use (default 1.0)
            
        Returns:
            Final training loss
        """
        logger.info(f"Training to generate {len(target_data)} values from input {input_val}")
        logger.info(f"Training for {epochs} epochs with learning rate {learning_rate}")
        
        X = np.array([[input_val]])
        y = target_data.reshape(1, -1)
        
        for epoch in range(epochs):
            # Forward pass
            h = self.sigmoid(np.dot(X, self.W1) + self.b1)
            output = self.sigmoid(np.dot(h, self.W2) + self.b2)
            
            # Calculate loss
            loss = np.mean((output - y) ** 2)
            
            # Progress updates
            if epoch % 10000 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss:.10f}")
            
            # Check for perfect reconstruction
            if loss < 1e-8:
                logger.info(f"🎉 Perfect generation at epoch {epoch}!")
                return loss
            
            # Backpropagation
            error = output - y
            
            # Output layer gradients
            d_output = error * output * (1 - output)  # sigmoid derivative
            d_h = d_output.dot(self.W2.T) * h * (1 - h)  # sigmoid derivative
            
            # Update weights
            self.W2 -= learning_rate * h.T.dot(d_output) / X.shape[0]
            self.b2 -= learning_rate * np.mean(d_output, axis=0)
            self.W1 -= learning_rate * X.T.dot(d_h) / X.shape[0]
            self.b1 -= learning_rate * np.mean(d_h, axis=0)
        
        logger.info(f"Final loss: {loss:.10f}")
        return loss
    
    def generate(self, input_val: float = 1.0) -> np.ndarray:
        """
        Generate data from the model.
        
        Args:
            input_val: Input value (default 1.0)
            
        Returns:
            Generated data array
        """
        return self.forward(input_val)
    
    def save_model_info(self) -> dict:
        """Get model information for saving."""
        return {
            'model_size_kb': self.model_size_kb,
            'hidden_size': self.hidden_size,
            'data_size': self.data_size,
            'weights': {
                'W1': self.W1.tolist(),
                'b1': self.b1.tolist(),
                'W2': self.W2.tolist(),
                'b2': self.b2.tolist()
            }
        }

