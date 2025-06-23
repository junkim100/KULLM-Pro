"""
Basic tests for KULLM-Pro package imports and functionality.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that main modules can be imported."""
    try:
        import fine_tune
        import code_switch
        from utils import data_processing, model_utils
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_config_loading():
    """Test that configuration files can be loaded."""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'train_with_think_tokens.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections exist
        assert 'model' in config
        assert 'training' in config
        assert 'lora' in config
        assert 'wandb' in config
    else:
        pytest.skip("Config file not found")


def test_console_commands():
    """Test that console command entry points are defined."""
    try:
        from src.fine_tune import train_main
        from scripts.chat import cli_main
        assert callable(train_main)
        assert callable(cli_main)
    except ImportError as e:
        pytest.fail(f"Console command import failed: {e}")


def test_package_structure():
    """Test that required files exist."""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    required_files = [
        'setup.py',
        'README.md',
        'requirements.txt',
        'src/fine_tune.py',
        'src/code_switch.py',
        'scripts/chat.py',
        'configs/train_with_think_tokens.yaml'
    ]
    
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        assert os.path.exists(full_path), f"Required file missing: {file_path}"
