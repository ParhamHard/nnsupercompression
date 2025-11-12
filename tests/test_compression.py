"""
Tests for neural network compression functionality.
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.compressor import TinyCompressor, PerfectTinyCompressor
from src.utils.text_utils import text_to_array, array_to_text, validate_text_reconstruction


class TestTextUtils(unittest.TestCase):
    """Test text utility functions."""
    
    def test_text_to_array(self):
        """Test text to array conversion."""
        text = "Hello"
        arr = text_to_array(text)
        
        self.assertEqual(arr.shape, (5,))
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.all(arr >= 0) and np.all(arr <= 1))
    
    def test_array_to_text(self):
        """Test array to text conversion."""
        text = "World"
        arr = text_to_array(text)
        reconstructed = array_to_text(arr)
        
        self.assertEqual(text, reconstructed)
    
    def test_validate_reconstruction(self):
        """Test reconstruction validation."""
        original = "Test"
        perfect = "Test"
        imperfect = "Tesx"
        
        perfect_validation = validate_text_reconstruction(original, perfect)
        imperfect_validation = validate_text_reconstruction(original, imperfect)
        
        self.assertTrue(perfect_validation["perfect_match"])
        self.assertFalse(imperfect_validation["perfect_match"])
        self.assertEqual(perfect_validation["char_accuracy"], 1.0)
        self.assertLess(imperfect_validation["char_accuracy"], 1.0)


class TestTinyCompressor(unittest.TestCase):
    """Test TinyCompressor functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_text = "Hello"
        self.data = text_to_array(self.test_text)
        self.compressor = TinyCompressor(len(self.data))
    
    def test_initialization(self):
        """Test compressor initialization."""
        self.assertEqual(self.compressor.data_size, len(self.data))
        self.assertEqual(self.compressor.hidden_size, 16)
        self.assertEqual(self.compressor.bottleneck_size, 4)
        self.assertGreater(self.compressor.model_size_kb, 0)
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        output = self.compressor.forward(self.data.reshape(1, -1))
        
        self.assertEqual(output.shape, (1, len(self.data)))
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))
    
    def test_compression_decompression(self):
        """Test compression and decompression."""
        compressed = self.compressor.compress(self.data)
        decompressed = self.compressor.decompress(compressed)
        
        self.assertEqual(compressed.shape, (1, self.compressor.bottleneck_size))
        self.assertEqual(decompressed.shape, (1, len(self.data)))
    
    def test_training(self):
        """Test network training."""
        initial_loss = np.mean((self.compressor.forward(self.data.reshape(1, -1)) - self.data.reshape(1, -1)) ** 2)
        
        # Train for a few epochs
        final_loss = self.compressor.train(self.data, epochs=100)
        
        self.assertLessEqual(final_loss, initial_loss)


class TestPerfectTinyCompressor(unittest.TestCase):
    """Test PerfectTinyCompressor functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_text = "Hi"
        self.data = text_to_array(self.test_text)
        self.compressor = PerfectTinyCompressor(len(self.data))
    
    def test_initialization(self):
        """Test compressor initialization."""
        self.assertEqual(self.compressor.data_size, len(self.data))
        self.assertEqual(self.compressor.hidden_size, 16)
        self.assertEqual(self.compressor.bottleneck_size, 4)
        self.assertGreater(self.compressor.model_size_kb, 0)
    
    def test_relu_activation(self):
        """Test ReLU activation function."""
        x = np.array([[-1, 0, 1]])
        activated = self.compressor.relu(x)
        
        self.assertEqual(activated[0, 0], 0)  # -1 -> 0
        self.assertEqual(activated[0, 1], 0)  # 0 -> 0
        self.assertEqual(activated[0, 2], 1)  # 1 -> 1
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        output = self.compressor.forward(self.data.reshape(1, -1))
        
        self.assertEqual(output.shape, (1, len(self.data)))
        # No activation on output layer, so values can be outside [0,1]
    
    def test_compression_decompression(self):
        """Test compression and decompression."""
        compressed = self.compressor.compress(self.data)
        decompressed = self.compressor.decompress(compressed)
        
        self.assertEqual(compressed.shape, (1, self.compressor.bottleneck_size))
        self.assertEqual(decompressed.shape, (1, len(self.data)))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete compression pipeline."""
    
    def test_complete_pipeline_tiny(self):
        """Test complete compression pipeline with TinyCompressor."""
        text = "Test compression"
        data = text_to_array(text)
        
        compressor = TinyCompressor(len(data))
        compressor.train(data, epochs=1000)
        
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)
        reconstructed = array_to_text(decompressed.flatten())
        
        validation = validate_text_reconstruction(text, reconstructed)
        
        self.assertGreater(validation["char_accuracy"], 0.5)  # Should have some accuracy
        self.assertLess(compressor.model_size_kb, 20)  # Should be reasonably small
    
    def test_complete_pipeline_perfect(self):
        """Test complete compression pipeline with PerfectTinyCompressor."""
        text = "Hi"  # Short text for perfect reconstruction
        data = text_to_array(text)
        
        compressor = PerfectTinyCompressor(len(data))
        compressor.train(data, epochs=5000)
        
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)
        reconstructed = array_to_text(decompressed.flatten())
        
        validation = validate_text_reconstruction(text, reconstructed)
        
        self.assertGreater(validation["char_accuracy"], 0.5)
        self.assertLess(compressor.model_size_kb, 20)


if __name__ == "__main__":
    unittest.main()
