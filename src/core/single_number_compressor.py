"""
Single number compressor - converts data to one number, then uses NN to generate it.

This is the ultimate compression: entire file → single number → tiny NN generates that number.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class SingleNumberCompressor:
    """
    A tiny neural network that takes input 1 and outputs a single number
    representing the entire file.
    
    Architecture: Input(1) → Hidden → Output(1)
    This can be extremely small (under 1KB)!
    """
    
    def __init__(self, hidden_size: int = 4, target_size_kb: float = 1.0):
        """
        Initialize the single number compressor.
        
        Args:
            hidden_size: Size of hidden layer
            target_size_kb: Target model size in KB
        """
        self.hidden_size = hidden_size
        self.target_size_kb = target_size_kb
        
        # Architecture: Input(1) → Hidden → Output(1)
        # Parameters: (1 * hidden + hidden) + (hidden * 1 + 1)
        # = hidden + hidden + hidden + 1 = 3*hidden + 1
        
        logger.info(f"Creating SingleNumberCompressor: Input=1, Hidden={hidden_size}, Output=1")
        
        # Initialize weights
        np.random.seed(42)
        # Input(1) → Hidden
        self.W1 = np.random.randn(1, self.hidden_size) * 0.1
        self.b1 = np.zeros(self.hidden_size)
        
        # Hidden → Output(1)
        self.W2 = np.random.randn(self.hidden_size, 1) * 0.1
        self.b2 = np.zeros(1)
        
        # Calculate model size
        self.model_size_kb = self._calculate_model_size()
        logger.info(f"Model size: {self.model_size_kb:.2f} KB (target: {target_size_kb:.2f} KB)")
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in KB."""
        total_params = (self.W1.size + self.b1.size + self.W2.size + self.b2.size)
        return (total_params * 4) / 1024  # float32 = 4 bytes
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with clipping."""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def forward(self, input_val: float = 1.0) -> float:
        """
        Forward pass: generate the number from input.
        
        Args:
            input_val: Input value (default 1.0)
            
        Returns:
            Generated number (normalized)
        """
        x = np.array([[input_val]])
        # Input → Hidden
        h = self.sigmoid(np.dot(x, self.W1) + self.b1)
        # Hidden → Output
        output = self.sigmoid(np.dot(h, self.W2) + self.b2)
        return float(output[0, 0])
    
    def train(self, target_number_normalized: float, epochs: int = 100000, 
              learning_rate: float = 0.1, input_val: float = 1.0) -> float:
        """
        Train the network to generate target_number when given input_val.
        
        Args:
            target_number_normalized: The normalized number we want (0-1 range)
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            input_val: Input value to use (default 1.0)
            
        Returns:
            Final training loss
        """
        logger.info(f"Training to generate number {target_number_normalized:.10f} from input {input_val}")
        logger.info(f"Training for {epochs} epochs with learning rate {learning_rate}")
        
        X = np.array([[input_val]])
        y = np.array([[target_number_normalized]])
        
        best_loss = float('inf')
        patience = 10000
        no_improve = 0
        
        for epoch in range(epochs):
            # Forward pass
            h = self.sigmoid(np.dot(X, self.W1) + self.b1)
            output = self.sigmoid(np.dot(h, self.W2) + self.b2)
            
            # Calculate loss
            loss = np.mean((output - y) ** 2)
            
            # Progress updates
            if epoch % 10000 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss:.10f}, Output = {output[0,0]:.10f}, Target = {target_number_normalized:.10f}")
            
            # Check for perfect match
            if loss < 1e-12:
                logger.info(f"🎉 Perfect generation at epoch {epoch}!")
                return loss
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience and epoch > 20000:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Backpropagation
            error = output - y
            
            # Output layer gradients (sigmoid derivative)
            d_output = error * output * (1 - output)
            d_h = d_output.dot(self.W2.T) * h * (1 - h)
            
            # Gradient clipping
            d_output = np.clip(d_output, -1, 1)
            d_h = np.clip(d_h, -1, 1)
            
            # Update weights
            self.W2 -= learning_rate * h.T.dot(d_output) / X.shape[0]
            self.b2 -= learning_rate * np.mean(d_output, axis=0)
            self.W1 -= learning_rate * X.T.dot(d_h) / X.shape[0]
            self.b1 -= learning_rate * np.mean(d_h, axis=0)
            
            # Adaptive learning rate
            if epoch > 0 and epoch % 20000 == 0:
                learning_rate *= 0.9
        
        logger.info(f"Final loss: {loss:.10f}")
        logger.info(f"Final output: {output[0,0]:.10f}, Target: {target_number_normalized:.10f}")
        return loss
    
    def generate(self, input_val: float = 1.0) -> float:
        """
        Generate the number from the model.
        
        Args:
            input_val: Input value (default 1.0)
            
        Returns:
            Generated normalized number
        """
        return self.forward(input_val)

