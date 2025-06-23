# KULLM-Pro v1.1.0

<div align="center">

**Korean University Large Language Model - Professional Edition**

*A production-ready framework for training bilingual reasoning models with think token capabilities*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.44%2B-yellow.svg)](https://huggingface.co/transformers/)

</div>

## 🚀 What's New in v1.1.0

- **🧠 Think Token Training**: Enhanced reasoning with `<think>` and `</think>` tokens
- **🧹 Clean Tokenizer**: Minimal special tokens (only 5 essential tokens)
- **🌏 Code-Switching**: Natural Korean-English bilingual reasoning
- **💬 Enhanced Chat Interface**: Streaming generation with colored think tokens
- **📊 Production Pipeline**: Complete training and deployment workflow

## ✨ Features

### 🧠 **Think Token Reasoning**
- Models learn to show step-by-step reasoning using `<think>` tags
- Transparent problem-solving process
- Enhanced mathematical and logical reasoning capabilities

### 🌏 **Code-Switching Training**
- Natural Korean-English bilingual data generation
- Preserves reasoning structure while adding linguistic diversity
- Supports LIMO and other reasoning datasets

### 🧹 **Clean Tokenizer**
- Removes unnecessary special tokens (tool_call, vision, etc.)
- Keeps only essential tokens: `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<|endoftext|>`
- Faster training and cleaner inference

### 💬 **Enhanced Chat Interface**
- Real-time streaming generation
- Color-coded think tokens for better visualization
- Full Korean and English language support
- Production-ready inference

### 📊 **Production Pipeline**
- LoRA fine-tuning with advanced features
- Weights & Biases integration
- Comprehensive validation and statistics
- Resumable training with checkpoints

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 18GB+ VRAM for Qwen2.5-7B training, 8GB+ for inference

### Quick Install
```bash
git clone https://github.com/junkim100/KULLM-Pro.git
cd KULLM-Pro
pip install -r requirements.txt
```

### Development Install (Recommended)
```bash
git clone https://github.com/junkim100/KULLM-Pro.git
cd KULLM-Pro
pip install -e .

# This enables console commands:
# kullm-chat, kullm-train, kullm-code-switch
```

## 🚀 Quick Start

### 1. **Generate Code-Switched Training Data**
```bash
kullm-code-switch process \
  --dataset_name="GAIR/LIMO" \
  --split="train" \
  --n_samples=817 \
  --output_file="data/code_switched_LIMO_train.jsonl"
```

### 2. **Train Your Model**
```bash
kullm-train \
  --config="configs/train_with_think_tokens.yaml" \
  --data_file="data/code_switched_LIMO_train.jsonl" \
  --output_dir="outputs/kullm-pro-v1.1"
```

> **Note**: Think token formatting is automatically handled during training - no separate formatting step needed!

### 3. **Chat with Your Model**
```bash
kullm-chat \
  --model_path="outputs/kullm-pro-v1.1" \
  --max_new_tokens=2048
```

## 📁 Project Structure

```
KULLM-Pro/
├── 📁 src/                          # Core source code
│   ├── 🐍 code_switch.py           # Code-switching data generation
│   ├── 🐍 fine_tune.py             # Model training pipeline (includes think token formatting)
│   ├── 📁 utils/                   # Utility modules
│   │   ├── 🐍 clean_tokenizer.py   # Tokenizer cleaning utilities
│   │   ├── 🐍 data_processing.py   # Data processing utilities
│   │   └── 🐍 model_utils.py       # Model utilities
│   └── 🐍 __init__.py              # Package initialization
├── 📁 scripts/                     # User-facing scripts
│   └── 🐍 chat.py                  # Interactive chat interface
├── 📁 tools/                       # Development and deployment tools
│   └── 🐍 merge_lora.py            # LoRA merging utility
├── 📁 configs/                     # Configuration files
│   └── 📄 train_with_think_tokens.yaml
├── 📁 data/                        # Training datasets
├── 📁 outputs/                     # Model outputs and checkpoints
├── 🐍 setup.py                     # Package setup
├── 📄 .pre-commit-config.yaml      # Code quality automation
├── 📄 requirements.txt             # Dependencies
├── 📄 README.md                    # This file
└── 📄 LICENSE                      # Apache 2.0 License
```

## 🧠 Think Token Training

KULLM-Pro v1.1.0 introduces proper think token training for enhanced reasoning capabilities.

### Example Output
```
User: Solve 2x + 5 = 13

Assistant: <think>

이 문제는 linear equation이야. Let me solve step by step.
2x + 5 = 13
2x = 13 - 5
2x = 8
x = 4

Let me verify: 2(4) + 5 = 8 + 5 = 13 ✓

</think>

The answer is x = 4.
```

### Key Features:
- **Bilingual Reasoning**: Natural code-switching between Korean and English
- **Step-by-step Thinking**: Transparent problem-solving process
- **Verification**: Models learn to check their work
- **Clean Output**: Think tokens are clearly separated from final answers

## 🌏 Code-Switching

KULLM-Pro generates natural bilingual training data using sophisticated linguistic theories:

### Linguistic Framework
- **Matrix Language Frame (MLF)**: Korean sentence structure with English technical terms
- **Equivalence Constraint**: Maintains grammatical consistency at switch points
- **Free Morpheme Constraint**: Ensures natural morpheme boundaries

### Example Code-Switched Data
```json
{
  "question": "Find the derivative of f(x) = x² + 3x + 2",
  "solution": "1. 이 문제는 polynomial differentiation이야.\n2. Power rule을 사용하면, d/dx(x^n) = nx^(n-1)이다.\n3. 따라서 f'(x) = 2x + 3이다.",
  "answer": "f'(x) = 2x + 3"
}
```

## ⚙️ Configuration

### Training Configuration (`configs/train_with_think_tokens.yaml`)
```yaml
# Model settings
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  max_length: 8192

# Dataset settings
dataset:
  think_token_start: "<think>"
  think_token_end: "</think>"

# Training hyperparameters
training:
  learning_rate: 0.0002
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  save_steps: 200  # Save checkpoint every 200 steps
  logging_steps: 10
  save_total_limit: 2

# LoRA settings
lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.1
  bias: "none"

# Wandb settings
wandb:
  enabled: true
  project: "kullm-pro"
```

## 🔧 Advanced Usage

### Console Commands (Recommended)
```bash
# Chat interface
kullm-chat --model_path outputs/your-model --max_new_tokens 2048

# Training with custom config
kullm-train \
  --config configs/train_with_think_tokens.yaml \
  --data_file data/your_training_data.jsonl \
  --output_dir outputs/your-model

# Code switching data generation
kullm-code-switch process \
  --dataset_name GAIR/LIMO \
  --n_samples 817 \
  --output_file data/code_switched_output.jsonl
```

### Direct Script Usage (Alternative)
```bash
# Chat interface
python scripts/chat.py --model_path outputs/your-model

# Training
python src/fine_tune.py train --config configs/train_with_think_tokens.yaml

# LoRA merging
python tools/merge_lora.py --model_path outputs/your-lora-model --output_path outputs/merged-model
```

## 📊 Performance

### Training Metrics (Qwen2.5-7B)
- **Training Time**: ~4-6 hours on RTX 4090 (817 samples, 3 epochs)
- **Memory Usage**: ~18GB VRAM for training, ~8GB for inference
- **Think Token Coverage**: 95%+ of training samples include think tokens
- **Code-Switch Quality**: Natural bilingual reasoning with proper linguistic constraints

### Model Capabilities
- ✅ Mathematical reasoning with step-by-step explanations
- ✅ Natural Korean-English code-switching
- ✅ Transparent problem-solving process
- ✅ Verification and error checking
- ✅ Production-ready inference

## 🛠️ Development

### Running Tests
```bash
# Validate tokenizer cleaning
python src/utils/clean_tokenizer.py --tokenizer_path Qwen/Qwen2.5-7B-Instruct

# Test chat interface
kullm-chat --model_path outputs/your_model

# Test training pipeline
kullm-train --config configs/train_with_think_tokens.yaml --data_file data/test_data.jsonl
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Korea University NLP&AI Lab** for research and development
- **OpenAI** for API services used in code-switching
- **Hugging Face** for transformers library and model hosting
- **LIMO Dataset** creators for providing high-quality reasoning data

## 📞 Contact

- **Email**: junkim100@gmail.com
- **GitHub**: [junkim100](https://github.com/junkim100)
- **Issues**: [GitHub Issues](https://github.com/junkim100/KULLM-Pro/issues)

## 📚 Citation

If you use KULLM-Pro in your research, please cite:

```bibtex
@software{kullm_pro_2024,
  title={KULLM-Pro: Korean University Large Language Model - Professional Edition},
  author={Korea University NLP\&AI Lab},
  year={2024},
  version={1.1.0},
  url={https://github.com/junkim100/KULLM-Pro}
}
```

---

<div align="center">
Made with ❤️ by Korea University NLP&AI Lab
</div>
