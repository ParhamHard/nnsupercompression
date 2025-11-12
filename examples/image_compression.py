#!/usr/bin/env python3
"""
Image Compression Example using Neural Network Supercompression

This script demonstrates how to compress an image to a 1KB model
by converting it to text (base64) and then compressing the text.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.compressor import OneKBCompressor
from src.utils.image_utils import (
    image_to_base64,
    base64_to_image,
    base64_to_array,
    array_to_base64
)
import numpy as np


def compress_image_to_1kb(image_path: str, output_image_path: str = None):
    """
    Compress an image to a 1KB neural network model.
    
    Args:
        image_path: Path to input image
        output_image_path: Path to save reconstructed image (optional)
    """
    print("🎯 IMAGE COMPRESSION TO 1KB MODEL")
    print("=" * 60)
    print()
    
    # Step 1: Convert image to base64 text
    print("📸 Step 1: Converting image to base64 text...")
    base64_string = image_to_base64(image_path)
    print(f"   Base64 string length: {len(base64_string)} characters")
    print(f"   Original image size: {os.path.getsize(image_path) / 1024:.2f} KB")
    print()
    
    # Step 2: Convert base64 to normalized array
    print("🔢 Step 2: Converting base64 to normalized array...")
    data = base64_to_array(base64_string, normalize=True)
    print(f"   Array size: {data.shape}")
    print(f"   Array size in KB: {len(data) * 4 / 1024:.2f} KB")
    print()
    
    # Step 3: Create 1KB compressor
    print("🤖 Step 3: Creating 1KB compressor...")
    compressor = OneKBCompressor(len(data))
    print(f"   Model size: {compressor.model_size_kb:.2f} KB")
    print(f"   Target: 1.0 KB")
    if compressor.model_size_kb <= 1.0:
        print("   ✅ Model is under 1KB target!")
    else:
        print(f"   ⚠️  Model is {compressor.model_size_kb - 1.0:.2f} KB over target")
    print()
    
    # Step 4: Train the network
    print("🎓 Step 4: Training network to memorize image data...")
    print("   This may take a while...")
    final_loss = compressor.train(data, epochs=100000)
    print(f"   Training complete! Final loss: {final_loss:.10f}")
    print()
    
    # Step 5: Test compression and decompression
    print("🧪 Step 5: Testing compression and decompression...")
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    
    # Step 6: Convert back to base64
    print("🔄 Step 6: Converting back to base64...")
    reconstructed_base64 = array_to_base64(decompressed.flatten(), denormalize=True)
    print(f"   Reconstructed base64 length: {len(reconstructed_base64)} characters")
    print()
    
    # Step 7: Save reconstructed image
    if output_image_path is None:
        output_image_path = image_path.replace('.png', '_reconstructed.png').replace('.jpg', '_reconstructed.jpg')
        if output_image_path == image_path:
            output_image_path = image_path + '_reconstructed'
    
    print("💾 Step 7: Saving reconstructed image...")
    try:
        base64_to_image(reconstructed_base64, output_image_path)
        print(f"   Saved to: {output_image_path}")
        print(f"   Reconstructed image size: {os.path.getsize(output_image_path) / 1024:.2f} KB")
    except Exception as e:
        print(f"   ⚠️  Error saving image: {e}")
        print("   This might be due to slight reconstruction differences.")
    print()
    
    # Step 8: Calculate metrics
    print("📊 COMPRESSION RESULTS:")
    print("=" * 60)
    original_size_kb = os.path.getsize(image_path) / 1024
    model_size_kb = compressor.model_size_kb
    
    print(f"   Original image size: {original_size_kb:.2f} KB")
    print(f"   Model size: {model_size_kb:.2f} KB")
    print(f"   Compression ratio: {original_size_kb / model_size_kb:.1f}x")
    print()
    
    # Compare base64 strings
    base64_match = (base64_string == reconstructed_base64)
    print(f"   Base64 strings match: {'✅ YES!' if base64_match else '❌ NO'}")
    
    if not base64_match:
        # Calculate similarity
        min_len = min(len(base64_string), len(reconstructed_base64))
        matches = sum(1 for i in range(min_len) if base64_string[i] == reconstructed_base64[i])
        similarity = matches / max(len(base64_string), len(reconstructed_base64)) * 100
        print(f"   Base64 similarity: {similarity:.2f}%")
    
    print()
    
    if compressor.model_size_kb <= 1.0:
        print("🎉 SUCCESS! Image compressed to 1KB model!")
    else:
        print(f"⚠️  Model size is {compressor.model_size_kb:.2f} KB (target: 1.0 KB)")
    
    return {
        'original_size_kb': original_size_kb,
        'model_size_kb': model_size_kb,
        'compression_ratio': original_size_kb / model_size_kb,
        'base64_match': base64_match,
        'final_loss': final_loss
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_compression.py <image_path> [output_path]")
        print()
        print("Example:")
        print("  python image_compression.py assets/Morden_army_ensign.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    try:
        results = compress_image_to_1kb(image_path, output_path)
        print("\n✅ Image compression completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

