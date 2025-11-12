"""
Image processing utilities for neural network compression.

This module provides functions to convert images to text representations
that can be compressed using the neural network compressor, and vice versa.
"""

import base64
import numpy as np
from typing import Tuple
from PIL import Image
import io


def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_string = base64.b64encode(image_data).decode('utf-8')
    return base64_string


def base64_to_image(base64_string: str, output_path: str) -> None:
    """
    Convert a base64 string back to an image file.
    
    Args:
        base64_string: Base64 encoded image data
        output_path: Path where the image will be saved
    """
    image_data = base64.b64decode(base64_string)
    with open(output_path, 'wb') as image_file:
        image_file.write(image_data)


def image_to_array(image_path: str, normalize: bool = True) -> np.ndarray:
    """
    Convert an image to a normalized numpy array.
    
    Args:
        image_path: Path to the image file
        normalize: If True, normalize values to [0, 1] range
        
    Returns:
        Flattened numpy array of image pixel values
    """
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)
    
    if normalize:
        # Normalize to [0, 1] range
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
    else:
        # Normalize to [0, 1] range based on data type
        if img_array.dtype == np.uint8:
            img_array = img_array / 255.0
        elif img_array.dtype == np.uint16:
            img_array = img_array / 65535.0
    
    return img_array.flatten()


def array_to_image(arr: np.ndarray, output_path: str, original_shape: Tuple[int, ...] = None) -> None:
    """
    Convert a numpy array back to an image file.
    
    Args:
        arr: Flattened numpy array of image pixel values
        output_path: Path where the image will be saved
        original_shape: Original shape of the image (height, width, channels)
                        If None, will try to infer from array size
    """
    # Denormalize if needed
    if arr.max() <= 1.0:
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    
    # Reshape if shape is provided
    if original_shape:
        arr = arr.reshape(original_shape)
    else:
        # Try to infer shape (assume square RGB image)
        size = int(np.sqrt(len(arr) / 3))
        if size * size * 3 == len(arr):
            arr = arr.reshape((size, size, 3))
        else:
            # Try grayscale
            size = int(np.sqrt(len(arr)))
            if size * size == len(arr):
                arr = arr.reshape((size, size))
            else:
                raise ValueError(f"Cannot infer image shape from array size {len(arr)}")
    
    img = Image.fromarray(arr)
    img.save(output_path)


def image_to_text_hex(image_path: str) -> str:
    """
    Convert an image file to hexadecimal string representation.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Hexadecimal string of the image bytes
    """
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        hex_string = image_data.hex()
    return hex_string


def text_hex_to_image(hex_string: str, output_path: str) -> None:
    """
    Convert a hexadecimal string back to an image file.
    
    Args:
        hex_string: Hexadecimal string of image data
        output_path: Path where the image will be saved
    """
    image_data = bytes.fromhex(hex_string)
    with open(output_path, 'wb') as image_file:
        image_file.write(image_data)


def base64_to_array(base64_string: str, normalize: bool = True) -> np.ndarray:
    """
    Convert a base64 string to a normalized numpy array.
    
    Args:
        base64_string: Base64 encoded string
        normalize: If True, normalize values to [0, 1] range
        
    Returns:
        Numpy array of normalized character values
    """
    # Convert base64 string to array of ASCII values
    arr = np.array([ord(c) for c in base64_string], dtype=np.float32)
    
    if normalize:
        # Normalize to [0, 1] range
        # Base64 characters range from '+' (43) to 'z' (122), plus '=' (61)
        # We'll normalize to [0, 1] using min=43, max=122
        min_val = 43.0
        max_val = 122.0
        arr = (arr - min_val) / (max_val - min_val)
        arr = np.clip(arr, 0.0, 1.0)
    
    return arr


def array_to_base64(arr: np.ndarray, denormalize: bool = True) -> str:
    """
    Convert a numpy array back to a base64 string.
    
    Args:
        arr: Numpy array of normalized values
        denormalize: If True, denormalize from [0, 1] range
        
    Returns:
        Base64 string
    """
    # Valid base64 characters: A-Z (65-90), a-z (97-122), 0-9 (48-57), + (43), / (47), = (61)
    valid_chars = list(range(43, 58)) + [61] + list(range(65, 91)) + list(range(97, 123))
    valid_chars = sorted(valid_chars)
    
    if denormalize:
        # Denormalize from [0, 1] back to ASCII range
        min_val = 43.0
        max_val = 122.0
        arr = arr * (max_val - min_val) + min_val
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32)
    
    # Round to nearest valid base64 character
    result = []
    for val in arr:
        # Find nearest valid base64 character
        nearest = min(valid_chars, key=lambda x: abs(x - val))
        result.append(nearest)
    
    # Convert to string
    base64_string = ''.join([chr(int(x)) for x in result])
    return base64_string

