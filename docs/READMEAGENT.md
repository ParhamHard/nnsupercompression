# Neural Network Supercompression Library - AI Agent Reference

This document provides comprehensive information for AI agents to understand and work with the Neural Network Supercompression project.

## 🎯 Project Overview

The Neural Network Supercompression Library is a Python package that demonstrates data compression using tiny neural networks. The core concept is training small neural networks to memorize specific data, making the trained model itself the compressed representation.

## 🏗️ Project Structure

```
nnsupercompression/
├── src/                           # Main source code
│   ├── __init__.py               # Package initialization
│   ├── core/                     # Core compression algorithms
│   │   ├── __init__.py          # Core module init
│   │   └── compressor.py        # Main compression classes
│   ├── models/                   # Neural network architectures
│   │   ├── __init__.py          # Models module init
│   │   └── architectures.py     # Architecture configurations
│   └── utils/                    # Utility functions
│       ├── __init__.py          # Utils module init
│       └── text_utils.py        # Text processing utilities
├── examples/                      # Example scripts
│   ├── basic_compression.py     # Basic usage example
│   ├── perfect_compression.py   # Perfect compression example
│   └── architecture_experiments.py # Architecture comparison
├── tests/                        # Test suite
│   ├── __init__.py              # Tests module init
│   └── test_compression.py      # Comprehensive tests
├── setup.py                      # Package setup script
├── pyproject.toml               # Modern Python packaging
├── Makefile                     # Development commands
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore rules
├── README.md                    # Main documentation
└── READMEAGENT.md               # This AI agent reference
```

## 🔧 Core Components

### 1. Compression Classes (`src/core/compressor.py`)

#### TinyCompressor
- **Purpose**: Basic neural network compression with sigmoid activation
- **Architecture**: Input → Hidden(16) → Bottleneck(4) → Hidden(16) → Output
- **Use Case**: General compression with moderate training time
- **Key Methods**:
  - `__init__(data_size, hidden_size=16, bottleneck_size=4)`
  - `train(data, epochs=30000, learning_rate=0.1)`
  - `compress(data)` / `decompress(compressed)`

#### PerfectTinyCompressor
- **Purpose**: Advanced compression with ReLU activation and proper backpropagation
- **Architecture**: Same as TinyCompressor but with ReLU and better training
- **Use Case**: Perfect reconstruction when exact data reproduction is needed
- **Key Methods**:
  - `__init__(data_size, hidden_size=16, bottleneck_size=4)`
  - `train(data, epochs=100000, learning_rate=0.01, patience=5000)`
  - `compress(data)` / `decompress(compressed)`

### 2. Architecture Configurations (`src/models/architectures.py`)

#### AutoencoderArchitecture
- **Purpose**: Define and validate neural network architectures
- **Features**: Configurable layer sizes, activation functions, training parameters
- **Key Methods**:
  - `get_total_parameters()`: Calculate parameter count
  - `get_model_size_kb()`: Get model size in KB
  - `get_compression_ratio()`: Calculate compression ratio
  - `is_efficient(target_size_kb=10.0)`: Check if under size target

#### Predefined Architectures
- `get_tiny_architecture(input_size)`: Basic sigmoid architecture
- `get_perfect_architecture(input_size)`: ReLU architecture
- `get_deep_architecture(input_size, hidden_layers, bottleneck_size)`: Multi-layer
- `get_ultra_compressed_architecture(input_size)`: Minimal bottleneck

### 3. Text Utilities (`src/utils/text_utils.py`)

#### Core Functions
- `text_to_array(text, normalize=True)`: Convert text to normalized array
- `array_to_text(arr, denormalize=True)`: Convert array back to text
- `validate_text_reconstruction(original, reconstructed)`: Quality metrics

## 📚 Usage Patterns

### Basic Compression Workflow

```python
from src.core.compressor import TinyCompressor
from src.utils.text_utils import text_to_array, array_to_text

# 1. Prepare data
text = "Hello, World!"
data = text_to_array(text)

# 2. Create compressor
compressor = TinyCompressor(len(data))

# 3. Train network
final_loss = compressor.train(data, epochs=30000)

# 4. Compress and decompress
compressed = compressor.compress(data)
decompressed = compressor.decompress(compressed)
reconstructed = array_to_text(decompressed.flatten())

# 5. Validate results
print(f"Perfect match: {text == reconstructed}")
print(f"Model size: {compressor.model_size_kb:.2f} KB")
```

### Architecture Experimentation

```python
from src.models.architectures import get_tiny_architecture, get_perfect_architecture

# Compare architectures
tiny_arch = get_tiny_architecture(input_size=100)
perfect_arch = get_perfect_architecture(input_size=100)

print(f"Tiny size: {tiny_arch.get_model_size_kb():.2f} KB")
print(f"Perfect size: {perfect_arch.get_model_size_kb():.2f} KB")
print(f"Tiny efficient: {tiny_arch.is_efficient()}")
```

## 🧪 Testing and Development

### Running Tests
```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
python -m pytest tests/test_compression.py
```

### Development Commands
```bash
make help           # Show all commands
make install        # Install in development mode
make install-dev    # Install with dev dependencies
make lint           # Run linting
make format         # Format code
make clean          # Clean build artifacts
```

### Test Structure
- **TestTextUtils**: Text conversion and validation tests
- **TestTinyCompressor**: Basic compressor functionality tests
- **TestPerfectTinyCompressor**: Advanced compressor tests
- **TestIntegration**: End-to-end pipeline tests

## 🔍 Key Implementation Details

### Neural Network Architecture
- **Input Layer**: Size matches input data
- **Hidden Layers**: Configurable sizes (default: 16 neurons)
- **Bottleneck Layer**: Smallest layer for maximum compression (default: 4 neurons)
- **Output Layer**: Same size as input for reconstruction

### Training Process
- **Loss Function**: Mean Squared Error (MSE)
- **Optimization**: Gradient descent with configurable learning rate
- **Early Stopping**: Available in PerfectTinyCompressor
- **Gradient Clipping**: Prevents exploding gradients

### Data Processing
- **Text Conversion**: ASCII values → normalized [0,1] range
- **Array Handling**: Automatic reshaping and flattening
- **Validation**: Character-level accuracy and perfect match detection

## 🚀 Example Scripts

### 1. Basic Compression (`examples/basic_compression.py`)
- Demonstrates TinyCompressor usage
- Shows complete compression pipeline
- Includes validation and metrics

### 2. Perfect Compression (`examples/perfect_compression.py`)
- Uses PerfectTinyCompressor for exact reconstruction
- Longer training for perfect results
- Tests with different text types

### 3. Architecture Experiments (`examples/architecture_experiments.py`)
- Compares different network configurations
- Shows trade-offs between size and quality
- Provides insights for architecture selection

## 📊 Performance Characteristics

### Model Size Targets
- **Primary Goal**: < 10 KB model size
- **Typical Results**: 6.4 KB for standard architectures
- **Ultra-Compressed**: 3.2 KB with minimal bottleneck

### Training Times
- **TinyCompressor**: ~30,000 epochs for good results
- **PerfectTinyCompressor**: ~100,000 epochs for perfect reconstruction
- **Early Stopping**: Available to reduce unnecessary training

### Compression Ratios
- **Text Data**: 50-100x compression typical
- **Efficiency**: Depends on data size and architecture
- **Trade-off**: Smaller models = harder training

## 🛠️ Development Guidelines

### Code Style
- **Formatting**: Black formatter (88 character line length)
- **Type Hints**: Full type annotation required
- **Documentation**: Comprehensive docstrings for all functions
- **Logging**: Structured logging with appropriate levels

### Testing Requirements
- **Coverage**: Aim for >90% test coverage
- **Unit Tests**: Test individual components
- **Integration Tests**: Test complete workflows
- **Edge Cases**: Handle error conditions gracefully

### Error Handling
- **Input Validation**: Validate all user inputs
- **Graceful Degradation**: Handle failures gracefully
- **Informative Messages**: Clear error messages for debugging

## 🔮 Future Development Areas

### Potential Enhancements
1. **Multi-Data Support**: Single model for multiple data pieces
2. **Adaptive Architectures**: Automatic architecture optimization
3. **Different Data Types**: Images, audio, binary data support
4. **Streaming Compression**: Real-time compression capabilities
5. **Model Persistence**: Save/load trained models

### Research Directions
1. **Information Theory**: Theoretical compression limits
2. **Architecture Search**: Optimal network configurations
3. **Training Optimization**: Faster convergence methods
4. **Quality Metrics**: Better reconstruction quality measures

## 📞 Support and Resources

### Documentation
- **Main README**: Project overview and quick start
- **Code Comments**: Inline documentation and examples
- **Type Hints**: Self-documenting code structure

### Development Tools
- **Makefile**: Common development tasks
- **pyproject.toml**: Modern Python packaging
- **setup.py**: Traditional package setup
- **Requirements**: Minimal dependencies (NumPy only)

### Testing Framework
- **pytest**: Modern Python testing
- **Coverage**: Code coverage reporting
- **Linting**: Code quality checks
- **Type Checking**: Static type analysis

---

This reference provides AI agents with comprehensive information to understand, modify, and extend the Neural Network Supercompression Library. The project follows modern Python best practices and provides a solid foundation for neural network-based compression research and development.
