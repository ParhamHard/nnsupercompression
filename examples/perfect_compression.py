#!/usr/bin/env python3
"""
Perfect Neural Network Compression Example

This script demonstrates the perfect compression functionality using the
PerfectTinyCompressor, which achieves exact text reconstruction.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.compressor import PerfectTinyCompressor
from src.utils.text_utils import text_to_array, array_to_text, validate_text_reconstruction


def demonstrate_perfect_compression():
    """Demonstrate perfect compression functionality."""
    
    print("🎯 PERFECT NEURAL NETWORK COMPRESSION")
    print("=" * 55)
    print()
    
    print("💡 THE GOAL:")
    print("   Achieve PERFECT reconstruction - exact text match!")
    print("   Uses proper backpropagation and better optimization")
    print()
    
    # Test with shorter text for perfect reconstruction
    test_text = "Hello, World! This works perfectly!"
    print("📝 DATA TO COMPRESS:")
    print(f"   Text: '{test_text}'")
    print(f"   Length: {len(test_text)} characters")
    print()
    
    # Convert to numbers
    data = text_to_array(test_text)
    print("🔢 CONVERTING TO NUMBERS:")
    print(f"   Text → ASCII values → Normalized [0,1] array")
    print(f"   Array shape: {data.shape}")
    print()
    
    # Create the perfect compressor
    print("🤖 CREATING THE PERFECT COMPRESSOR:")
    compressor = PerfectTinyCompressor(len(data))
    print()
    
    # Train the network
    print("🎓 TRAINING FOR PERFECT RECONSTRUCTION:")
    final_loss = compressor.train(data, epochs=100000)
    print()
    
    # Test compression
    print("🧪 TESTING PERFECT COMPRESSION:")
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    reconstructed_text = array_to_text(decompressed.flatten())
    
    print(f"   Original: '{test_text}'")
    print(f"   Reconstructed: '{reconstructed_text}'")
    
    is_perfect = test_text == reconstructed_text
    print(f"   Perfect match: {'✅ YES!' if is_perfect else '❌ NO'}")
    print()
    
    # Show results
    print("📊 PERFECT RESULTS:")
    print(f"   Original size: {len(data) * 4 / 1024:.2f} KB (float32 array)")
    print(f"   Model size: {compressor.model_size_kb:.2f} KB")
    print(f"   Compression ratio: {(len(data) * 4) / (compressor.model_size_kb * 1024):.1f}x")
    print(f"   Final loss: {final_loss:.15f}")
    print()
    
    if compressor.model_size_kb < 10 and is_perfect:
        print("🎉 PERFECT SUCCESS!")
        print(f"   ✅ Model is only {compressor.model_size_kb:.2f} KB (< 10KB)")
        print(f"   ✅ Perfect text reconstruction achieved!")
        print(f"   🎯 This tiny model can perfectly memorize your text!")
    elif compressor.model_size_kb < 10:
        print("🎯 SUCCESS! Model is under 10KB")
        print("   ⚠️  Reconstruction is very close but not perfect yet")
        print("   💡 Try more training epochs or different text")
    else:
        print("⚠️  Model is over 10KB target")


def test_with_junk_text():
    """Test perfect reconstruction with junk text."""
    print("\n" + "=" * 55)
    print("🧪 TESTING PERFECT RECONSTRUCTION WITH JUNK TEXT:")
    print()
    
    junk_text = "hellohowaadbaskdbjasjkbdksajbdkjasbdjksabkjdbakjbdkjsabkajsdbkjas xm xmz mxnmxzc mc reyou"
    print(f"Junk text: '{junk_text}'")
    
    # Create compressor
    data = text_to_array(junk_text)
    compressor = PerfectTinyCompressor(len(data))
    
    # Train
    compressor.train(data, epochs=50000)
    
    # Test
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    reconstructed = array_to_text(decompressed.flatten())
    
    print(f"Reconstructed: '{reconstructed}'")
    print(f"Perfect match: {'✅ YES!' if junk_text == reconstructed else '❌ NO'}")
    print(f"Model size: {compressor.model_size_kb:.2f} KB")


if __name__ == "__main__":
    # Perfect compression demonstration
    demonstrate_perfect_compression()
    
    # Test with junk text
    test_with_junk_text()
    
    print("\n🎯 This demonstrates PERFECT neural network compression!")
    print("   The model can achieve exact text reconstruction!")
