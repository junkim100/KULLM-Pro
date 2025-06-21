"""
KULLM-Pro: Korean-English Code-Switched Language Model Training Pipeline

A production-ready framework for creating code-switched datasets and fine-tuning
language models with LoRA for Korean-English mathematical reasoning.

This package provides two main components:

1. **Code Switching Module** (`code_switching`):
   - Process any Hugging Face dataset with flexible parameters
   - Generate Korean-English code-switched versions using OpenAI API
   - Support for both regular API calls and cost-efficient Batch API
   - Automatic filename generation based on dataset parameters

2. **Fine-tuning Module** (`fine_tuning`):
   - LoRA (Low-Rank Adaptation) fine-tuning for efficient training
   - Advanced training features with Accelerate/DeepSpeed support
   - Weights & Biases integration for experiment tracking
   - Checkpoint management and resumable training
   - Comprehensive evaluation and model information utilities

3. **Utilities Module** (`utils`):
   - Configuration management with YAML support
   - Centralized logging setup
   - Helper functions for file operations and validation

Example Usage:
    ```python
    from kullm_pro import CodeSwitchingPipeline, FineTuningPipeline
    from kullm_pro import load_config, setup_logging

    # Setup logging
    logger = setup_logging()

    # Load configuration
    config = load_config("config.yaml")

    # Create pipelines
    code_switch_pipeline = CodeSwitchingPipeline(...)
    fine_tune_pipeline = FineTuningPipeline(config)
    ```

Requirements:
    - Python 3.8+
    - PyTorch 2.0+
    - Transformers 4.44+
    - OpenAI API key (for code switching)
    - CUDA-compatible GPU (recommended for fine-tuning)

License:
    MIT License - see LICENSE file for details

Authors:
    KULLM-Pro Development Team

Version:
    1.0.0 - Initial release with code switching and LoRA fine-tuning
"""

__version__ = "1.0.0"
__author__ = "KULLM-Pro Development Team"
__email__ = "kullm-pro@example.com"
__license__ = "MIT"
__description__ = "Korean-English Code-Switched Language Model Training Pipeline"
__url__ = "https://github.com/junkim100/KULLM-Pro"

# Import main components with error handling
try:
    from .code_switching import CodeSwitchingPipeline
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import CodeSwitchingPipeline: {e}")
    CodeSwitchingPipeline = None

try:
    from .fine_tuning import FineTuningPipeline
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import FineTuningPipeline: {e}")
    FineTuningPipeline = None

try:
    from .utils import load_config, setup_logging
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import utilities: {e}")
    load_config = None
    setup_logging = None

# Define public API
__all__ = [
    # Main pipeline classes
    "CodeSwitchingPipeline",
    "FineTuningPipeline",

    # Utility functions
    "load_config",
    "setup_logging",

    # Package metadata
    "__version__",
    "__author__",
    "__license__",
    "__description__",
]
