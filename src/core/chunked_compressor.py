"""
Chunked compression for large data that needs to fit in 1KB models.

This module provides functionality to split large data into chunks,
compress each chunk with a 1KB model, and reconstruct the original data.
"""

import numpy as np
from typing import List, Tuple
import logging
from .compressor import OneKBCompressor

logger = logging.getLogger(__name__)


class ChunkedOneKBCompressor:
    """
    Compresses large data by splitting it into chunks and compressing each chunk
    with a separate 1KB model.
    
    This allows compressing data of any size while maintaining the 1KB model size
    constraint per chunk.
    """
    
    def __init__(self, chunk_size: int = 20):
        """
        Initialize the chunked compressor.
        
        Args:
            chunk_size: Maximum size of each chunk (default 20 for 1KB model)
        """
        self.chunk_size = chunk_size
        self.compressors: List[OneKBCompressor] = []
        self.num_chunks = 0
    
    def _split_into_chunks(self, data: np.ndarray) -> List[np.ndarray]:
        """Split data into chunks of specified size."""
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            # Pad last chunk if needed
            if len(chunk) < self.chunk_size:
                padding = np.zeros(self.chunk_size - len(chunk))
                chunk = np.concatenate([chunk, padding])
            chunks.append(chunk)
        return chunks
    
    def train(self, data: np.ndarray, epochs: int = 50000) -> float:
        """
        Train compressors for each chunk.
        
        Args:
            data: Input data to compress
            epochs: Number of training epochs per chunk
            
        Returns:
            Average final loss across all chunks
        """
        chunks = self._split_into_chunks(data)
        self.num_chunks = len(chunks)
        self.compressors = []
        
        logger.info(f"Training {self.num_chunks} compressors for chunks of size {self.chunk_size}")
        
        total_loss = 0.0
        for i, chunk in enumerate(chunks):
            if (i + 1) % 100 == 0 or i == 0:
                logger.info(f"Training compressor {i+1}/{self.num_chunks}...")
            compressor = OneKBCompressor(len(chunk))
            loss = compressor.train(chunk, epochs=epochs)
            self.compressors.append(compressor)
            total_loss += loss
        
        avg_loss = total_loss / self.num_chunks
        logger.info(f"Average loss across all chunks: {avg_loss:.10f}")
        return avg_loss
    
    def compress(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Compress data by compressing each chunk.
        
        Args:
            data: Input data to compress
            
        Returns:
            List of compressed representations (bottlenecks) for each chunk
        """
        chunks = self._split_into_chunks(data)
        compressed_chunks = []
        
        for i, (chunk, compressor) in enumerate(zip(chunks, self.compressors)):
            compressed = compressor.compress(chunk)
            compressed_chunks.append(compressed)
        
        return compressed_chunks
    
    def decompress(self, compressed_chunks: List[np.ndarray], original_length: int) -> np.ndarray:
        """
        Decompress data by decompressing each chunk.
        
        Args:
            compressed_chunks: List of compressed representations
            original_length: Original length of the data (before padding)
            
        Returns:
            Reconstructed data array
        """
        decompressed_chunks = []
        
        for i, (compressed, compressor) in enumerate(zip(compressed_chunks, self.compressors)):
            decompressed = compressor.decompress(compressed)
            decompressed_chunks.append(decompressed.flatten())
        
        # Concatenate all chunks
        reconstructed = np.concatenate(decompressed_chunks)
        
        # Trim to original length (remove padding)
        return reconstructed[:original_length]
    
    def get_total_model_size_kb(self) -> float:
        """Get total model size in KB (sum of all chunk compressors)."""
        return sum(comp.model_size_kb for comp in self.compressors)
    
    def get_num_models(self) -> int:
        """Get number of 1KB models used."""
        return len(self.compressors)

