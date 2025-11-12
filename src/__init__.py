"""
Neural Network Supercompression Library

A library for compressing data using tiny neural networks that learn to memorize
and perfectly reconstruct input data through overfitting.
"""

__version__ = "1.0.0"
__author__ = "Neural Compression Team"

from .core.compressor import TinyCompressor, PerfectTinyCompressor, OneKBCompressor
from .utils.text_utils import text_to_array, array_to_text
from .models.architectures import AutoencoderArchitecture, get_1kb_architecture

__all__ = [
    "TinyCompressor",
    "PerfectTinyCompressor",
    "OneKBCompressor",
    "text_to_array",
    "array_to_text",
    "AutoencoderArchitecture",
    "get_1kb_architecture"
]
