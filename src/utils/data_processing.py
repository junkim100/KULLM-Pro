"""
Data Processing Utilities

Provides utilities for loading, processing, and saving datasets in JSONL format.
Includes functions for dataset filtering, validation, and filename generation.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line {line_num}: {e}")
                    raise
    
    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path where to save the file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(data)} samples to {file_path}")


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a filename.
    
    Args:
        name: String to sanitize
        
    Returns:
        Sanitized string safe for use as filename
    """
    # Replace forward slashes with underscores
    sanitized = name.replace('/', '_')
    # Remove or replace other problematic characters
    sanitized = re.sub(r'[<>:"|?*]', '_', sanitized)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def filter_shortest_samples(data: List[Dict[str, Any]], n_samples: int, 
                          text_columns: List[str] = None) -> List[Dict[str, Any]]:
    """
    Filter and return the n shortest samples based on total character count.
    
    Args:
        data: List of data samples
        n_samples: Number of samples to return
        text_columns: List of column names to consider for length calculation.
                     If None, uses all string values in each sample.
        
    Returns:
        List of n shortest samples
    """
    if n_samples >= len(data):
        logger.warning(f"Requested {n_samples} samples but only {len(data)} available")
        return data
    
    def calculate_length(sample: Dict[str, Any]) -> int:
        """Calculate total character count for a sample."""
        if text_columns:
            # Use specified columns
            total_length = 0
            for col in text_columns:
                if col in sample and isinstance(sample[col], str):
                    total_length += len(sample[col])
            return total_length
        else:
            # Use all string values
            return sum(len(str(value)) for value in sample.values() 
                      if isinstance(value, str))
    
    # Sort by length and take the shortest n samples
    sorted_data = sorted(data, key=calculate_length)
    filtered_data = sorted_data[:n_samples]
    
    logger.info(f"Filtered to {len(filtered_data)} shortest samples from {len(data)} total")
    return filtered_data


def validate_jsonl_format(data: List[Dict[str, Any]], required_fields: List[str] = None) -> bool:
    """
    Validate that data is in proper format for training.
    
    Args:
        data: List of data samples to validate
        required_fields: List of required field names
        
    Returns:
        True if valid, False otherwise
    """
    if not data:
        logger.error("Data is empty")
        return False
    
    if required_fields:
        for i, sample in enumerate(data):
            for field in required_fields:
                if field not in sample:
                    logger.error(f"Sample {i} missing required field: {field}")
                    return False
    
    logger.info(f"Validated {len(data)} samples")
    return True


def create_output_filename(dataset_name: str, split: str, subset: Optional[str], 
                          n_samples: int, prefix: str = "") -> str:
    """
    Create a descriptive filename for output files.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split (train, test, etc.)
        subset: Dataset subset (can be None)
        n_samples: Number of samples
        prefix: Optional prefix for the filename
        
    Returns:
        Formatted filename
    """
    sanitized_name = sanitize_filename(dataset_name)
    
    if subset:
        base_name = f"{sanitized_name}_{split}_{subset}_{n_samples}"
    else:
        base_name = f"{sanitized_name}_{split}_{n_samples}"
    
    if prefix:
        return f"{prefix}_{base_name}.jsonl"
    else:
        return f"{base_name}.jsonl"


def load_hf_dataset(dataset_name: str, split: str = "train", 
                   subset: Optional[str] = None) -> Dataset:
    """
    Load a dataset from Hugging Face Hub.
    
    Args:
        dataset_name: Name of the dataset on HF Hub
        split: Dataset split to load
        subset: Dataset subset/configuration
        
    Returns:
        Loaded dataset
        
    Raises:
        Exception: If dataset cannot be loaded
    """
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        logger.info(f"Loaded dataset {dataset_name} ({split}) with {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise
