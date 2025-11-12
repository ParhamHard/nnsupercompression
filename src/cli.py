#!/usr/bin/env python3
"""
Command-line interface for Neural Network Supercompression Library.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.compressor import TinyCompressor, PerfectTinyCompressor
from src.utils.text_utils import text_to_array, array_to_text, validate_text_reconstruction


def compress_text(text: str, method: str = "basic", epochs: int = None) -> dict:
    """
    Compress text using neural network compression.
    
    Args:
        text: Text to compress
        method: Compression method ("basic" or "perfect")
        epochs: Number of training epochs (auto-determined if None)
        
    Returns:
        Dictionary with compression results
    """
    print(f"🎯 Compressing text: '{text}'")
    print(f"📏 Length: {len(text)} characters")
    print(f"🔧 Method: {method}")
    print()
    
    # Convert text to array
    data = text_to_array(text)
    
    # Choose compressor
    if method == "basic":
        compressor = TinyCompressor(len(data))
        if epochs is None:
            epochs = 30000
    else:  # perfect
        compressor = PerfectTinyCompressor(len(data))
        if epochs is None:
            epochs = 100000
    
    print(f"🤖 Created {method} compressor")
    print(f"   Model size: {compressor.model_size_kb:.2f} KB")
    print(f"   Training for {epochs} epochs...")
    print()
    
    # Train the network
    final_loss = compressor.train(data, epochs=epochs)
    
    # Test compression
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    reconstructed = array_to_text(decompressed.flatten())
    
    # Validate results
    validation = validate_text_reconstruction(text, reconstructed)
    
    # Calculate metrics
    original_size = len(data) * 4 / 1024  # KB
    compression_ratio = original_size / compressor.model_size_kb
    
    results = {
        "original_text": text,
        "reconstructed_text": reconstructed,
        "perfect_match": validation["perfect_match"],
        "char_accuracy": validation["char_accuracy"],
        "original_size_kb": original_size,
        "model_size_kb": compressor.model_size_kb,
        "compression_ratio": compression_ratio,
        "final_loss": final_loss,
        "method": method
    }
    
    return results


def display_results(results: dict):
    """Display compression results in a formatted way."""
    print("📊 COMPRESSION RESULTS")
    print("=" * 50)
    print()
    
    print(f"📝 Original: '{results['original_text']}'")
    print(f"🔄 Reconstructed: '{results['reconstructed_text']}'")
    print()
    
    print(f"✅ Perfect match: {'YES!' if results['perfect_match'] else 'NO'}")
    print(f"📊 Character accuracy: {results['char_accuracy']:.2%}")
    print()
    
    print(f"📏 Original size: {results['original_size_kb']:.2f} KB")
    print(f"🤖 Model size: {results['model_size_kb']:.2f} KB")
    print(f"🗜️  Compression ratio: {results['compression_ratio']:.1f}x")
    print(f"🎯 Final loss: {results['final_loss']:.10f}")
    print()
    
    if results['model_size_kb'] < 10:
        print("🎉 SUCCESS! Model is under 10KB target!")
    else:
        print("⚠️  Model is over 10KB target")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Neural Network Supercompression Library CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic compression
  nnsupercompression "Hello, World!"
  
  # Perfect compression with custom epochs
  nnsupercompression "Test text" --method perfect --epochs 50000
  
  # Interactive mode
  nnsupercompression --interactive
        """
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to compress (optional if using --interactive)"
    )
    
    parser.add_argument(
        "--method",
        choices=["basic", "perfect"],
        default="basic",
        help="Compression method (default: basic)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (auto-determined if not specified)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Neural Network Supercompression Library v1.0.0"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        print("🎯 Neural Network Supercompression - Interactive Mode")
        print("=" * 55)
        print()
        
        text = input("Enter text to compress: ").strip()
        if not text:
            print("❌ No text provided. Exiting.")
            return 1
        
        print()
        method = input("Choose method (basic/perfect) [basic]: ").strip().lower()
        if method not in ["basic", "perfect"]:
            method = "basic"
        
        print()
        epochs_input = input("Training epochs (press Enter for auto): ").strip()
        epochs = int(epochs_input) if epochs_input.isdigit() else None
        
        print()
        
    elif args.text:
        text = args.text
        method = args.method
        epochs = args.epochs
    else:
        parser.print_help()
        return 1
    
    try:
        # Perform compression
        results = compress_text(text, method, epochs)
        
        # Display results
        display_results(results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Compression interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during compression: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
