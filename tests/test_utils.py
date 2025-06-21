"""
Tests for utility functions in KULLM-Pro
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from kullm_pro.utils.config import load_config, validate_config, get_env_var
from kullm_pro.utils.helpers import generate_filename, ensure_directory, sanitize_filename
from kullm_pro.utils.logging import setup_logging, get_logger


class TestConfigUtils:
    """Test configuration utility functions"""

    def test_load_config_success(self, temp_dir):
        """Test successful config loading"""
        config_data = {
            "model": {"name": "test-model"},
            "training": {"epochs": 3},
            "lora": {"r": 16},
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        loaded_config = load_config(str(config_file))
        assert loaded_config == config_data

    def test_load_config_file_not_found(self):
        """Test config loading with non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_config.yaml")

    def test_load_config_invalid_yaml(self, temp_dir):
        """Test config loading with invalid YAML"""
        config_file = temp_dir / "invalid_config.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            load_config(str(config_file))

    def test_validate_config_success(self, sample_config):
        """Test successful config validation"""
        assert validate_config(sample_config) is True

    def test_validate_config_missing_section(self):
        """Test config validation with missing required section"""
        config = {"model": {"name": "test"}}
        
        with pytest.raises(ValueError, match="Missing required configuration section: training"):
            validate_config(config)

    def test_validate_config_invalid_learning_rate(self):
        """Test config validation with invalid learning rate"""
        config = {
            "model": {"name": "test"},
            "training": {"learning_rate": 1.0},  # Too high
            "lora": {"r": 16},
        }
        
        with pytest.raises(ValueError, match="learning_rate must be between"):
            validate_config(config)

    def test_validate_config_string_scientific_notation(self):
        """Test config validation with string scientific notation"""
        config = {
            "model": {"name": "test"},
            "training": {"learning_rate": "2e-4"},  # String scientific notation
            "lora": {"r": 16},
        }
        
        # Should not raise an error
        assert validate_config(config) is True

    def test_get_env_var_exists(self):
        """Test getting existing environment variable"""
        with patch.dict("os.environ", {"TEST_VAR": "test_value"}):
            assert get_env_var("TEST_VAR") == "test_value"

    def test_get_env_var_default(self):
        """Test getting environment variable with default"""
        assert get_env_var("NON_EXISTENT_VAR", "default") == "default"

    def test_get_env_var_required_missing(self):
        """Test getting required environment variable that's missing"""
        with pytest.raises(ValueError, match="Required environment variable not found"):
            get_env_var("NON_EXISTENT_VAR", required=True)


class TestHelperUtils:
    """Test helper utility functions"""

    def test_generate_filename_basic(self):
        """Test basic filename generation"""
        filename = generate_filename("GAIR/LIMO", "train", None, 300, "original")
        assert filename == "GAIR_LIMO_train_300_original.jsonl"

    def test_generate_filename_with_subset(self):
        """Test filename generation with subset"""
        filename = generate_filename("dataset", "train", "subset", 100, "code_switched")
        assert filename == "dataset_train_subset_100_code_switched.jsonl"

    def test_generate_filename_no_samples(self):
        """Test filename generation without sample count"""
        filename = generate_filename("dataset", "test", None, None, "original")
        assert filename == "dataset_test_original.jsonl"

    def test_ensure_directory_new(self, temp_dir):
        """Test creating new directory"""
        new_dir = temp_dir / "new_directory"
        result = ensure_directory(str(new_dir))
        
        assert result.exists()
        assert result.is_dir()

    def test_ensure_directory_existing(self, temp_dir):
        """Test with existing directory"""
        result = ensure_directory(str(temp_dir))
        assert result.exists()
        assert result.is_dir()

    def test_sanitize_filename(self):
        """Test filename sanitization"""
        dirty_filename = "file<>name|with?invalid*chars"
        clean_filename = sanitize_filename(dirty_filename)
        assert clean_filename == "file_name_with_invalid_chars"

    def test_sanitize_filename_multiple_underscores(self):
        """Test sanitization removes multiple consecutive underscores"""
        dirty_filename = "file___with___many___underscores"
        clean_filename = sanitize_filename(dirty_filename)
        assert clean_filename == "file_with_many_underscores"


class TestLoggingUtils:
    """Test logging utility functions"""

    def test_setup_logging_default(self):
        """Test default logging setup"""
        logger = setup_logging()
        assert logger.name == "kullm_pro"
        assert logger.level == 20  # INFO level

    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level"""
        logger = setup_logging(level="DEBUG")
        assert logger.level == 10  # DEBUG level

    def test_setup_logging_with_file(self, temp_dir):
        """Test logging setup with file output"""
        log_file = temp_dir / "test.log"
        logger = setup_logging(log_file=str(log_file))
        
        # Test that file handler was added
        file_handlers = [h for h in logger.handlers if hasattr(h, 'baseFilename')]
        assert len(file_handlers) > 0

    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid level"""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level="INVALID")

    def test_get_logger(self):
        """Test getting named logger"""
        logger = get_logger("test_module")
        assert logger.name == "kullm_pro.test_module"


class TestIntegration:
    """Integration tests for utility functions"""

    def test_config_and_logging_integration(self, temp_dir, sample_config):
        """Test integration between config loading and logging setup"""
        # Save config with logging settings
        config_with_logging = {
            **sample_config,
            "logging": {
                "level": "DEBUG",
                "file": str(temp_dir / "test.log"),
            }
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_with_logging, f)
        
        # Load config and setup logging
        config = load_config(str(config_file))
        validate_config(config)
        
        logging_config = config.get("logging", {})
        logger = setup_logging(
            level=logging_config.get("level", "INFO"),
            log_file=logging_config.get("file")
        )
        
        assert logger.level == 10  # DEBUG
        assert (temp_dir / "test.log").exists()

    def test_filename_generation_and_directory_creation(self, temp_dir):
        """Test integration between filename generation and directory creation"""
        # Generate filename
        filename = generate_filename("test/dataset", "train", "subset", 100, "processed")
        
        # Create directory and file path
        output_dir = ensure_directory(str(temp_dir / "outputs"))
        file_path = output_dir / filename
        
        # Verify everything works together
        assert filename == "test_dataset_train_subset_100_processed.jsonl"
        assert output_dir.exists()
        assert file_path.parent.exists()
        
        # Test file creation
        file_path.touch()
        assert file_path.exists()
