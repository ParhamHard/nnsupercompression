"""
Neural Network Supercompression Library

A library for compressing data using tiny neural networks that learn to memorize
and perfectly reconstruct input data through overfitting.
"""

__version__ = "1.0.0"
__author__ = "Neural Compression Team"

from .core.compressor import TinyCompressor, PerfectTinyCompressor
from .utils.text_utils import text_to_array, array_to_text
from .models.architectures import AutoencoderArchitecture

__all__ = [
    "TinyCompressor",
    "PerfectTinyCompressor", 
    "text_to_array",
    "array_to_text",
    "AutoencoderArchitecture"
]
