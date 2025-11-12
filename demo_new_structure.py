#!/usr/bin/env python3
"""
Demo script showcasing the new project structure.

This script demonstrates how to use the restructured Neural Network
Supercompression Library with its new modular architecture.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demo_new_structure():
    """Demonstrate the new project structure and capabilities."""
    
    print("🎯 NEURAL NETWORK SUPERCOMPRESSION LIBRARY")
    print("   New Professional Structure Demo")
    print("=" * 60)
    print()
    
    try:
        # Import from the new structure
        print("📦 Importing from new structure...")
        from core.compressor import TinyCompressor, PerfectTinyCompressor
        from models.architectures import get_tiny_architecture, get_perfect_architecture
        from utils.text_utils import text_to_array, array_to_text, validate_text_reconstruction
        print("✅ All imports successful!")
        print()
        
        # Test text
        test_text = "Hello from the new structure!"
        print(f"📝 Test text: '{test_text}'")
        print(f"📏 Length: {len(test_text)} characters")
        print()
        
        # Convert to array
        data = text_to_array(test_text)
        print(f"🔢 Converted to array: {data.shape}")
        print()
        
        # Test architecture configurations
        print("🏗️  Testing architecture configurations...")
        tiny_arch = get_tiny_architecture(len(data))
        perfect_arch = get_perfect_architecture(len(data))
        
        print(f"   Tiny architecture: {tiny_arch.get_model_size_kb():.2f} KB")
        print(f"   Perfect architecture: {perfect_arch.get_model_size_kb():.2f} KB")
        print(f"   Both under 10KB: {'✅' if tiny_arch.is_efficient() and perfect_arch.is_efficient() else '❌'}")
        print()
        
        # Test basic compression
        print("🧪 Testing basic compression...")
        compressor = TinyCompressor(len(data))
        print(f"   Created compressor: {compressor.model_size_kb:.2f} KB")
        
        # Quick training (fewer epochs for demo)
        print("   Training network (quick demo)...")
        final_loss = compressor.train(data, epochs=5000)
        print(f"   Training complete! Final loss: {final_loss:.6f}")
        
        # Test compression
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)
        reconstructed = array_to_text(decompressed.flatten())
        
        # Validate results
        validation = validate_text_reconstruction(test_text, reconstructed)
        
        print(f"   Original: '{test_text}'")
        print(f"   Reconstructed: '{reconstructed}'")
        print(f"   Perfect match: {'✅ YES!' if validation['perfect_match'] else '❌ NO'}")
        print(f"   Character accuracy: {validation['char_accuracy']:.2%}")
        print()
        
        # Show results
        original_size = len(data) * 4 / 1024  # KB
        compression_ratio = original_size / compressor.model_size_kb
        
        print("📊 COMPRESSION RESULTS:")
        print(f"   Original size: {original_size:.2f} KB")
        print(f"   Model size: {compressor.model_size_kb:.2f} KB")
        print(f"   Compression ratio: {compression_ratio:.1f}x")
        print()
        
        if compressor.model_size_kb < 10:
            print("🎉 SUCCESS! Model is under 10KB target!")
        else:
            print("⚠️  Model is over 10KB target")
        
        print()
        print("🚀 NEW STRUCTURE FEATURES:")
        print("   ✅ Modular design with clear separation of concerns")
        print("   ✅ Comprehensive testing framework")
        print("   ✅ Modern Python packaging")
        print("   ✅ CLI interface")
        print("   ✅ Architecture experimentation tools")
        print("   ✅ Professional development workflow")
        print()
        print("💡 Try running the examples:")
        print("   python examples/basic_compression.py")
        print("   python examples/perfect_compression.py")
        print("   python examples/architecture_experiments.py")
        print()
        print("🔧 Development commands:")
        print("   make help          # Show all commands")
        print("   make install       # Install in development mode")
        print("   make test          # Run tests")
        print("   make examples      # Run all examples")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the project root directory")
        print("   and the src/ directory is properly set up.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = demo_new_structure()
    if success:
        print("\n🎯 Demo completed successfully!")
        print("   The new project structure is working correctly!")
    else:
        print("\n❌ Demo failed. Check the error messages above.")
        sys.exit(1)
