"""
Configuration management utilities
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")


def validate_config(config: Dict[str, Any], required_sections: Optional[list] = None) -> bool:
    """
    Validate configuration structure and values

    Args:
        config: Configuration dictionary
        required_sections: List of required top-level sections

    Returns:
        True if valid, raises ValueError if invalid
    """
    if required_sections is None:
        required_sections = ['model', 'training', 'lora']

    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate model configuration
    if 'model' in config:
        model_config = config['model']
        if 'name' not in model_config:
            raise ValueError("Model name is required in model configuration")

        max_length = model_config.get('max_length', 2048)
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")

    # Validate training configuration
    if 'training' in config:
        training_config = config['training']

        # Check numeric parameters
        numeric_params = {
            'epochs': (1, 100),
            'batch_size': (1, 128),
            'learning_rate': (1e-6, 1e-1),
            'gradient_accumulation_steps': (1, 1000)
        }

        for param, (min_val, max_val) in numeric_params.items():
            if param in training_config:
                value = training_config[param]

                # Handle string scientific notation (e.g., "2e-4")
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValueError(f"{param} must be a valid number, got: {value}")

                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    raise ValueError(f"{param} must be between {min_val} and {max_val}, got: {value}")

    # Validate LoRA configuration
    if 'lora' in config:
        lora_config = config['lora']

        # Check LoRA parameters
        if 'r' in lora_config:
            r = lora_config['r']
            if not isinstance(r, int) or r <= 0:
                raise ValueError("LoRA rank (r) must be a positive integer")

        if 'alpha' in lora_config:
            alpha = lora_config['alpha']
            if not isinstance(alpha, (int, float)) or alpha <= 0:
                raise ValueError("LoRA alpha must be a positive number")

        if 'dropout' in lora_config:
            dropout = lora_config['dropout']
            if not isinstance(dropout, (int, float)) or not (0 <= dropout <= 1):
                raise ValueError("LoRA dropout must be between 0 and 1")

    return True


def get_env_var(var_name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get environment variable with optional default and validation

    Args:
        var_name: Environment variable name
        default: Default value if not found
        required: Whether the variable is required

    Returns:
        Environment variable value or default

    Raises:
        ValueError: If required variable is not found
    """
    value = os.getenv(var_name, default)

    if required and value is None:
        raise ValueError(f"Required environment variable not found: {var_name}")

    return value
