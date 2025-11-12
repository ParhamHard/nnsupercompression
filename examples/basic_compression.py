#!/usr/bin/env python3
"""
Basic Neural Network Compression Example

This script demonstrates the basic usage of the neural network compression library.
It shows how to compress and decompress text using the TinyCompressor.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.compressor import TinyCompressor
from src.utils.text_utils import text_to_array, array_to_text, validate_text_reconstruction


def demonstrate_basic_compression():
    """Demonstrate basic compression functionality."""
    
    print("🎯 BASIC NEURAL NETWORK COMPRESSION")
    print("=" * 50)
    print()
    
    # Test text
    test_text = "Hello, this is a test string for compression!"
    print(f"📝 Text to compress: '{test_text}'")
    print(f"📏 Length: {len(test_text)} characters")
    print()
    
    # Convert to numerical array
    data = text_to_array(test_text)
    print(f"🔢 Converted to array: {data.shape}")
    print()
    
    # Create compressor
    compressor = TinyCompressor(len(data))
    print(f"🤖 Created compressor with {compressor.model_size_kb:.2f} KB model")
    print()
    
    # Train the network
    print("🎓 Training network...")
    final_loss = compressor.train(data, epochs=30000)
    print(f"✅ Training complete! Final loss: {final_loss:.6f}")
    print()
    
    # Test compression and decompression
    print("🧪 Testing compression...")
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    reconstructed_text = array_to_text(decompressed.flatten())
    
    print(f"   Original: '{test_text}'")
    print(f"   Reconstructed: '{reconstructed_text}'")
    
    # Validate reconstruction
    validation = validate_text_reconstruction(test_text, reconstructed_text)
    print(f"   Perfect match: {'✅ YES!' if validation['perfect_match'] else '❌ NO'}")
    print(f"   Character accuracy: {validation['char_accuracy']:.2%}")
    print()
    
    # Show results
    print("📊 COMPRESSION RESULTS:")
    print(f"   Original size: {len(data) * 4 / 1024:.2f} KB")
    print(f"   Model size: {compressor.model_size_kb:.2f} KB")
    print(f"   Compression ratio: {len(data) * 4 / (compressor.model_size_kb * 1024):.1f}x")
    print()
    
    if compressor.model_size_kb < 10:
        print("🎯 SUCCESS! Model is under 10KB target!")
    else:
        print("⚠️  Model is over 10KB target")


if __name__ == "__main__":
    demonstrate_basic_compression()
