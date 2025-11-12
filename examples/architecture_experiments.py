#!/usr/bin/env python3
"""
Architecture Experimentation Example

This script demonstrates how to experiment with different neural network
architectures for compression, showing the trade-offs between model size
and compression quality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.architectures import (
    get_tiny_architecture, 
    get_perfect_architecture,
    get_deep_architecture,
    get_ultra_compressed_architecture
)
from src.core.compressor import TinyCompressor, PerfectTinyCompressor
from src.utils.text_utils import text_to_array, array_to_text, validate_text_reconstruction


def compare_architectures():
    """Compare different neural network architectures."""
    
    print("🏗️  NEURAL NETWORK ARCHITECTURE COMPARISON")
    print("=" * 60)
    print()
    
    # Test text
    test_text = "Hello, this is a test string for architecture comparison!"
    print(f"📝 Test text: '{test_text}'")
    print(f"📏 Length: {len(test_text)} characters")
    print()
    
    # Convert to array
    data = text_to_array(test_text)
    
    # Define architectures to test
    architectures = [
        ("Tiny (Sigmoid)", get_tiny_architecture(len(data))),
        ("Perfect (ReLU)", get_perfect_architecture(len(data))),
        ("Deep (ReLU)", get_deep_architecture(len(data))),
        ("Ultra-Compressed", get_ultra_compressed_architecture(len(data)))
    ]
    
    results = []
    
    for name, arch in architectures:
        print(f"🔬 Testing {name} architecture:")
        print(f"   Encoder: {arch.encoder_layers}")
        print(f"   Bottleneck: {arch.bottleneck_size}")
        print(f"   Decoder: {arch.decoder_layers}")
        print(f"   Activation: {arch.activation}")
        print(f"   Parameters: {arch.get_total_parameters():,}")
        print(f"   Model size: {arch.get_model_size_kb():.2f} KB")
        print(f"   Compression ratio: {arch.get_compression_ratio():.1f}x")
        print(f"   Efficient (<10KB): {'✅' if arch.is_efficient() else '❌'}")
        print()
        
        # Test actual compression
        if arch.activation == "sigmoid":
            compressor = TinyCompressor(len(data), arch.hidden_size, arch.bottleneck_size)
        else:
            compressor = PerfectTinyCompressor(len(data), arch.hidden_size, arch.bottleneck_size)
        
        # Train with fewer epochs for comparison
        epochs = 10000 if "Deep" in name else 20000
        final_loss = compressor.train(data, epochs=epochs)
        
        # Test reconstruction
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)
        reconstructed = array_to_text(decompressed.flatten())
        
        validation = validate_text_reconstruction(test_text, reconstructed)
        
        results.append({
            "name": name,
            "architecture": arch,
            "final_loss": final_loss,
            "perfect_match": validation["perfect_match"],
            "char_accuracy": validation["char_accuracy"],
            "actual_model_size": compressor.model_size_kb
        })
        
        print(f"   Training complete:")
        print(f"     Final loss: {final_loss:.6f}")
        print(f"     Perfect match: {'✅' if validation['perfect_match'] else '❌'}")
        print(f"     Character accuracy: {validation['char_accuracy']:.2%}")
        print(f"     Actual model size: {compressor.model_size_kb:.2f} KB")
        print()
    
    # Summary
    print("📊 ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 60)
    print()
    
    for result in results:
        arch = result["architecture"]
        print(f"{result['name']:20} | "
              f"Size: {result['actual_model_size']:5.2f} KB | "
              f"Loss: {result['final_loss']:8.6f} | "
              f"Perfect: {'✅' if result['perfect_match'] else '❌'} | "
              f"Accuracy: {result['char_accuracy']:5.1%}")
    
    print()
    print("💡 INSIGHTS:")
    print("   - Smaller bottleneck = more compression but harder training")
    print("   - ReLU activation often leads to better convergence")
    print("   - Deeper networks can handle more complex data")
    print("   - Trade-off between model size and reconstruction quality")


if __name__ == "__main__":
    compare_architectures()
