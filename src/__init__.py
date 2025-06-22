"""
KULLM-Pro: Korean University Large Language Model - Professional Edition

A production-ready framework for training and deploying Korean-English bilingual
language models with think token capabilities and code-switching functionality.

Version 1.1.0 Features:
- Think token training for enhanced reasoning
- Code-switching between Korean and English
- Clean tokenizer with minimal special tokens
- Enhanced chat interface with streaming
- Production-ready training pipeline

Main Components:

1. **Code Switching Module** (`code_switch.py`):
   - Process datasets with Korean-English code-switching
   - Generate bilingual training data using OpenAI API
   - Support for LIMO and other reasoning datasets
   - Flexible batch processing and filtering

2. **Fine-tuning Module** (`fine_tune.py`):
   - LoRA fine-tuning with think token support
   - Clean tokenizer with minimal special tokens
   - Advanced training with Accelerate and Weights & Biases
   - Think token validation and statistics

3. **Chat Interface** (`chat.py`):
   - Streaming generation with think token display
   - Color-coded think tokens for better visualization
   - Korean and English language support
   - Production-ready inference interface

4. **Utilities** (`utils/`):
   - Data processing and formatting utilities
   - Model utilities and tokenizer cleaning
   - Think token validation and statistics

Example Usage:
    ```bash
    # Code switching
    python src/code_switch.py --dataset_name="GAIR/LIMO" --split="train" --n_samples=817

    # Fine-tuning with think tokens
    python src/fine_tune.py --config configs/train_with_think_tokens.yaml

    # Chat interface
    python chat.py --model_path outputs/your-model
    ```

Requirements:
    - Python 3.8+
    - PyTorch 2.0+
    - Transformers 4.44+
    - CUDA-compatible GPU (recommended)

License:
    Apache 2.0 License

Authors:
    Korea University NLP&AI Lab
"""

__version__ = "1.1.0"
__author__ = "Korea University NLP&AI Lab"
__email__ = "junkim100@gmail.com"
__license__ = "Apache-2.0"
__description__ = "Korean University Large Language Model - Professional Edition"
__url__ = "https://github.com/junkim100/KULLM-Pro"

# Define public API
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    "__url__",
]
