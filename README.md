# Neural Network Supercompression Library

A library for compressing data using tiny neural networks that learn to memorize and perfectly reconstruct input data through overfitting.

## 🎯 Overview

This project demonstrates an innovative approach to data compression using neural network overfitting. Instead of traditional compression algorithms, we train tiny neural networks to memorize specific data, making the trained model itself the compressed representation.

### Key Features

- **Tiny Models**: Compress data into models under 10KB
- **Perfect Reconstruction**: Achieve exact data reproduction
- **Multiple Architectures**: Choose between different network configurations
- **Easy to Use**: Simple API for compression and decompression
- **Well Tested**: Comprehensive test suite and examples

## 🏗️ Architecture

The library uses autoencoder neural networks with a bottleneck architecture:

```
Input → Encoder → Bottleneck → Decoder → Output
```

- **Input/Output**: Same size as your data
- **Encoder**: Compresses data to bottleneck
- **Bottleneck**: Tiny layer (2-4 neurons) for maximum compression
- **Decoder**: Reconstructs data from bottleneck

## 📦 Installation

### From Source

```bash
git clone https://github.com/yourusername/nnsupercompression.git
cd nnsupercompression
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Basic Compression

```python
from src.core.compressor import TinyCompressor
from src.utils.text_utils import text_to_array, array_to_text

# Convert text to numerical array
text = "Hello, World!"
data = text_to_array(text)

# Create and train compressor
compressor = TinyCompressor(len(data))
compressor.train(data, epochs=30000)

# Compress and decompress
compressed = compressor.compress(data)
decompressed = compressor.decompress(compressed)
reconstructed = array_to_text(decompressed.flatten())

print(f"Original: {text}")
print(f"Reconstructed: {reconstructed}")
print(f"Perfect match: {text == reconstructed}")
```

### Perfect Compression

```python
from src.core.compressor import PerfectTinyCompressor

# Use the advanced compressor for perfect reconstruction
compressor = PerfectTinyCompressor(len(data))
compressor.train(data, epochs=100000)  # More training for perfection

# This achieves exact text reconstruction!
```

## 📚 Examples

Run the provided examples to see the library in action:

```bash
# Basic compression
python examples/basic_compression.py

# Perfect compression
python examples/perfect_compression.py

# Architecture comparison
python examples/architecture_experiments.py
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run linting
make lint
```

## 🛠️ Development

### Project Structure

```
nnsupercompression/
├── src/                    # Source code
│   ├── core/              # Core compression algorithms
│   ├── models/            # Neural network architectures
│   └── utils/             # Utility functions
├── examples/               # Example scripts
├── tests/                  # Test suite
├── docs/                   # Documentation
├── setup.py                # Package setup
├── pyproject.toml          # Modern Python packaging
├── Makefile                # Development commands
└── README.md               # This file
```

### Available Commands

```bash
make help           # Show all available commands
make install        # Install in development mode
make install-dev    # Install with development dependencies
make test           # Run tests
make lint           # Run linting
make format         # Format code
make clean          # Clean build artifacts
make examples       # Run all examples
```

## 🔬 How It Works

### The Compression Concept

1. **Data Input**: Convert your data (text, numbers, etc.) to a numerical array
2. **Network Training**: Train a tiny neural network to perfectly reconstruct this data
3. **Compression**: The trained network becomes your compressed representation
4. **Decompression**: Use the network to reconstruct the original data

### Why It Works

- **Overfitting**: The network learns to memorize your specific data
- **Tiny Architecture**: Minimal parameters keep the model small
- **Perfect Recall**: With enough training, exact reconstruction is possible
- **Universal**: Works with any type of data that can be converted to numbers

### Trade-offs

- **Pros**: 
  - Extremely small compressed size
  - Perfect reconstruction possible
  - Works with any data type
  
- **Cons**:
  - One model per data piece
  - Training time required
  - Not suitable for streaming/real-time compression

## 📊 Performance

### Model Sizes

| Architecture | Hidden Size | Bottleneck | Model Size | Target |
|--------------|-------------|------------|------------|---------|
| Tiny (Sigmoid) | 16 | 4 | ~6.4 KB | ✅ <10KB |
| Perfect (ReLU) | 16 | 4 | ~6.4 KB | ✅ <10KB |
| Deep (ReLU) | 32→16→8 | 4 | ~12.8 KB | ❌ >10KB |
| Ultra-Compressed | 8 | 2 | ~3.2 KB | ✅ <10KB |

### Compression Ratios

For typical text data:
- **Original**: 100 characters = 400 bytes
- **Compressed**: 6.4 KB model
- **Ratio**: ~62.5x compression (for this specific text)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by neural network overfitting research
- Built with modern Python best practices
- Comprehensive testing and documentation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/nnsupercompression/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/nnsupercompression/discussions)
- **Documentation**: [Project Wiki](https://github.com/yourusername/nnsupercompression/wiki)

---

**Note**: This library is for educational and research purposes. While it demonstrates interesting compression concepts, traditional compression algorithms may be more practical for production use cases.
