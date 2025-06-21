"""
Helper utilities for common operations
"""

import os
import re
from pathlib import Path
from typing import Optional


def generate_filename(
    dataset_name: str,
    split: str,
    subset: Optional[str] = None,
    n_samples: Optional[int] = None,
    suffix: str = "original",
    extension: str = "jsonl"
) -> str:
    """
    Generate descriptive filename for dataset files
    
    Args:
        dataset_name: Name of the dataset (e.g., "GAIR/LIMO")
        split: Dataset split (e.g., "train", "validation", "test")
        subset: Dataset subset (if applicable)
        n_samples: Number of samples
        suffix: Additional suffix (e.g., "original", "code_switched")
        extension: File extension
        
    Returns:
        Generated filename
        
    Example:
        generate_filename("GAIR/LIMO", "train", None, 300, "original")
        -> "GAIR_LIMO_train_300_original.jsonl"
    """
    # Clean dataset name (replace / with _)
    clean_dataset = dataset_name.replace("/", "_").replace("-", "_")
    
    # Build filename parts
    parts = [clean_dataset, split]
    
    if subset:
        parts.append(subset)
    
    if n_samples:
        parts.append(str(n_samples))
    
    parts.append(suffix)
    
    # Join with underscores and add extension
    filename = "_".join(parts) + f".{extension}"
    
    return filename


def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human readable string
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"
