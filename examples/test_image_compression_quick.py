#!/usr/bin/env python3
"""
Quick test of image compression with reduced epochs for faster testing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.chunked_compressor import ChunkedOneKBCompressor
from src.utils.image_utils import (
    image_to_base64,
    base64_to_image,
    base64_to_array,
    array_to_base64
)
import numpy as np


def quick_test(image_path: str, max_chunks: int = 10, epochs: int = 5000):
    """
    Quick test with limited chunks and epochs.
    
    Args:
        image_path: Path to input image
        max_chunks: Maximum number of chunks to test (for speed)
        epochs: Number of epochs per chunk (reduced for testing)
    """
    print("🎯 QUICK IMAGE COMPRESSION TEST")
    print("=" * 60)
    print()
    
    # Step 1: Convert image to base64
    print("📸 Step 1: Converting image to base64...")
    base64_string = image_to_base64(image_path)
    print(f"   Base64 length: {len(base64_string)} characters")
    print(f"   Original size: {os.path.getsize(image_path) / 1024:.2f} KB")
    print()
    
    # Step 2: Convert to array
    print("🔢 Step 2: Converting to array...")
    data = base64_to_array(base64_string, normalize=True)
    print(f"   Array size: {len(data)}")
    print()
    
    # Step 3: Test with limited chunks
    print(f"🧪 Step 3: Testing with first {max_chunks} chunks...")
    test_data = data[:max_chunks * 20]  # Test with first N chunks
    print(f"   Testing with {len(test_data)} characters")
    print()
    
    compressor = ChunkedOneKBCompressor(chunk_size=20)
    
    # Step 4: Train (with reduced epochs)
    print(f"🎓 Step 4: Training {max_chunks} compressors (quick test, {epochs} epochs each)...")
    final_loss = compressor.train(test_data, epochs=epochs)
    print(f"   Training complete! Average loss: {final_loss:.10f}")
    print(f"   Number of models: {compressor.get_num_models()}")
    print(f"   Total model size: {compressor.get_total_model_size_kb():.2f} KB")
    print(f"   Average per model: {compressor.get_total_model_size_kb() / compressor.get_num_models():.2f} KB")
    print()
    
    # Step 5: Test compression/decompression
    print("🧪 Step 5: Testing compression...")
    compressed = compressor.compress(test_data)
    decompressed = compressor.decompress(compressed, len(test_data))
    
    # Step 6: Convert back
    print("🔄 Step 6: Converting back to base64...")
    reconstructed_base64 = array_to_base64(decompressed.flatten(), denormalize=True)
    
    # Compare
    original_test_base64 = base64_string[:len(reconstructed_base64)]
    match = (original_test_base64 == reconstructed_base64)
    similarity = sum(a == b for a, b in zip(original_test_base64, reconstructed_base64)) / len(original_test_base64) * 100
    
    print(f"   Original test length: {len(original_test_base64)}")
    print(f"   Reconstructed length: {len(reconstructed_base64)}")
    print(f"   Perfect match: {'✅ YES!' if match else '❌ NO'}")
    print(f"   Similarity: {similarity:.2f}%")
    print()
    
    # Results
    print("📊 RESULTS:")
    print("=" * 60)
    avg_size = compressor.get_total_model_size_kb() / compressor.get_num_models()
    print(f"   ✅ Each model is ~{avg_size:.2f} KB (target: 1.0 KB)")
    print(f"   ✅ Total of {compressor.get_num_models()} models for {len(test_data)} characters")
    print(f"   ✅ Reconstruction quality: {similarity:.2f}%")
    print()
    
    if avg_size <= 1.0:
        print("🎉 SUCCESS! Models are under 1KB target!")
        print()
        print("💡 To compress the full image:")
        print(f"   - Full image needs ~{len(data) // 20} models")
        print(f"   - Each model: ~{avg_size:.2f} KB")
        print(f"   - Total size: ~{len(data) // 20 * avg_size:.2f} KB")
        print(f"   - Original: {os.path.getsize(image_path) / 1024:.2f} KB")
    else:
        print(f"⚠️  Average model size: {avg_size:.2f} KB")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "assets/Morden_army_ensign.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    quick_test(image_path, max_chunks=10, epochs=20000)

