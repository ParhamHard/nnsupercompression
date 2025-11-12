"""
Text utility functions for converting between text and numerical arrays.

This module provides functions to:
- Convert text to normalized numerical arrays
- Convert numerical arrays back to text
- Handle ASCII encoding/decoding
"""

import numpy as np
from typing import Union, List


def text_to_array(text: str, normalize: bool = True) -> np.ndarray:
    """
    Convert text to array of ASCII values.
    
    Args:
        text: Input text string
        normalize: Whether to normalize values to [0, 1] range
        
    Returns:
        numpy array of character codes
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    char_codes = [ord(c) for c in text]
    result = np.array(char_codes, dtype=np.float32)
    
    if normalize:
        result = result / 255.0
    
    return result


def array_to_text(arr: Union[np.ndarray, List[float]], 
                  denormalize: bool = True) -> str:
    """
    Convert array back to text with proper clamping.
    
    Args:
        arr: Input array of character codes
        denormalize: Whether to denormalize from [0, 1] range
        
    Returns:
        Reconstructed text string
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array or list")
    
    # Convert to 1D array if needed
    if arr.ndim > 1:
        arr = arr.flatten()
    
    result = ""
    
    for val in arr:
        if denormalize:
            char_code = int(round(np.clip(val * 255, 0, 255)))
        else:
            char_code = int(round(np.clip(val, 0, 255)))
        
        # Ensure valid ASCII range
        char_code = max(0, min(255, char_code))
        result += chr(char_code)
    
    return result


def validate_text_reconstruction(original: str, reconstructed: str) -> dict:
    """
    Validate text reconstruction quality.
    
    Args:
        original: Original text
        reconstructed: Reconstructed text
        
    Returns:
        Dictionary with validation metrics
    """
    is_perfect = original == reconstructed
    
    # Calculate character-level accuracy
    min_len = min(len(original), len(reconstructed))
    if min_len == 0:
        char_accuracy = 0.0
    else:
        matches = sum(1 for i in range(min_len) if original[i] == reconstructed[i])
        char_accuracy = matches / min_len
    
    # Calculate length difference
    length_diff = abs(len(original) - len(reconstructed))
    
    return {
        "perfect_match": is_perfect,
        "char_accuracy": char_accuracy,
        "length_difference": length_diff,
        "original_length": len(original),
        "reconstructed_length": len(reconstructed)
    }
