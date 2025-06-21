# KULLM-Pro

**Korean-English Code-Switched Language Model Training Pipeline**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![CI](https://github.com/junkim100/KULLM-Pro/workflows/CI/badge.svg)](https://github.com/junkim100/KULLM-Pro/actions)
[![codecov](https://codecov.io/gh/junkim100/KULLM-Pro/branch/main/graph/badge.svg)](https://codecov.io/gh/junkim100/KULLM-Pro)

A production-ready framework for creating code-switched datasets and fine-tuning language models with LoRA for Korean-English mathematical reasoning.

## 🚀 Features

- **Flexible Code Switching**: Process any Hugging Face dataset with configurable parameters
- **Long Content Support**: Automatic chunking and processing of extremely long system prompts and input data
- **LoRA Fine-tuning**: Advanced fine-tuning with LoRA, checkpoint management, and experiment tracking
- **OpenAI Integration**: Batch API support for cost-efficient code switching
- **Production Ready**: Proper error handling, logging, and configuration management
- **CLI Interface**: Easy-to-use command-line tools with Python Fire
- **Experiment Tracking**: Weights & Biases integration for monitoring training

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Code Switching](#code-switching)
- [Fine-tuning](#fine-tuning)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for fine-tuning)
- OpenAI API key (for code switching)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/junkim100/KULLM-Pro.git
   cd KULLM-Pro
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Configure the system:**
   ```bash
   # Edit config.yaml with your preferred settings
   vim config.yaml
   ```

## ⚡ Quick Start

### Code Switching

Generate code-switched datasets from any Hugging Face dataset:

```bash
# Basic usage
python code_switch.py run "GAIR/LIMO" --split="train" --n=300

# With custom parameters
python code_switch.py run "microsoft/orca-math-word-problems-200k" \
  --split="train" \
  --subset="default" \
  --n=1000 \
  --output_dir="./data"
```

### Fine-tuning

Train models with LoRA fine-tuning:

```bash
# Basic fine-tuning
python fine_tune.py train \
  --train_file="./data/GAIR_LIMO_train_300_code_switched.jsonl" \
  --output_dir="./outputs/qwen_limo_cs"

# With custom parameters
python fine_tune.py train \
  --train_file="./data/training_data.jsonl" \
  --val_file="./data/validation_data.jsonl" \
  --output_dir="./outputs/my_model" \
  --run_name="experiment_1" \
  --epochs=5 \
  --batch_size=4
```

## ⚙️ Configuration

KULLM-Pro uses a centralized configuration system with `config.yaml`. Key sections include:

### Model Configuration
```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  max_length: 2048
  torch_dtype: "bfloat16"
```

### Training Configuration
```yaml
training:
  epochs: 3
  batch_size: 2
  learning_rate: 2e-4
  gradient_accumulation_steps: 8
```

### LoRA Configuration
```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Environment Variables

Set up your `.env` file with required API keys:

```bash
# Required for code switching
OPENAI_API_KEY=your_openai_api_key_here

# Optional for experiment tracking
WANDB_API_KEY=your_wandb_api_key_here

# Optional for private models/datasets
HF_TOKEN=your_hugging_face_token_here
```

## 🔄 Code Switching

The code switching module can process any Hugging Face dataset and generate Korean-English code-switched versions.

### Basic Usage

```bash
# Process GAIR/LIMO dataset
python code_switch.py run "GAIR/LIMO" --split="train" --n=300

# Process with subset
python code_switch.py run "microsoft/orca-math-word-problems-200k" \
  --split="train" \
  --subset="default" \
  --n=1000
```

### Advanced Options

```bash
python code_switch.py run "GAIR/LIMO" \
  --split="train" \
  --n=500 \
  --output_dir="./custom_data" \
  --use_batch_api=true \
  --model="o4-mini-2025-04-16" \
  --text_column="solution" \
  --question_column="question" \
  --answer_column="answer"
```

### Generated Files

The code switching process generates descriptive filenames:

- **Original**: `GAIR_LIMO_train_300_original.jsonl`
- **Code-switched**: `GAIR_LIMO_train_300_code_switched.jsonl`
- **Statistics**: `processing_stats.json`
- **Failed items**: `failed_items.json` (if any)

### Batch Processing

Process multiple datasets using a configuration file:

```bash
# Create datasets_config.yaml
python code_switch.py batch_process --datasets_config="datasets_config.yaml"
```

Example `datasets_config.yaml`:
```yaml
- dataset: "GAIR/LIMO"
  split: "train"
  n: 300
- dataset: "microsoft/orca-math-word-problems-200k"
  split: "train"
  subset: "default"
  n: 1000
```

## 🎯 Fine-tuning

The fine-tuning module supports LoRA training with advanced features.

### Basic Training

```bash
# Train with default settings
python fine_tune.py train \
  --train_file="./data/GAIR_LIMO_train_300_code_switched.jsonl" \
  --output_dir="./outputs/my_model"
```

### Advanced Training

```bash
# Custom training with validation
python fine_tune.py train \
  --train_file="./data/train.jsonl" \
  --val_file="./data/val.jsonl" \
  --output_dir="./outputs/advanced_model" \
  --run_name="experiment_v2" \
  --epochs=5 \
  --batch_size=4 \
  --learning_rate=1e-4 \
  --lora_r=32 \
  --lora_alpha=64
```

### Resume Training

```bash
# Resume from checkpoint
python fine_tune.py train \
  --train_file="./data/train.jsonl" \
  --output_dir="./outputs/my_model" \
  --resume_from_checkpoint="./outputs/my_model/checkpoint-1000"
```

### Model Evaluation

```bash
# Evaluate trained model
python fine_tune.py evaluate \
  --model_dir="./outputs/my_model" \
  --eval_file="./data/test.jsonl"
```

### Model Information

```bash
# Get model information
python fine_tune.py info --model_dir="./outputs/my_model"

# List checkpoints
python fine_tune.py list_checkpoints --output_dir="./outputs/my_model"
```

## 📏 Long Content Processing

KULLM-Pro supports automatic chunking and processing of extremely long system prompts and input data.

### Configuration for Long Content

```yaml
# OpenAI Configuration for Long Content
openai:
  # Long Content Handling
  max_input_length: 150000      # 150K characters max per request
  chunk_size: 100000            # 100K character chunks
  overlap_size: 10000           # 10K character overlap
  enable_chunking: true         # Enable automatic chunking

  # Optimized settings for long content
  batch_size: 25                # Smaller batches
  timeout: 900                  # 15 minutes timeout
```

### Usage with Long Content

```bash
# Use long content optimized configuration
python code_switch.py run "GAIR/LIMO" \
  --split="train" \
  --n=100 \
  --config_file="examples/config_long_content.yaml"
```

### How Chunking Works

1. **Automatic Detection**: Content exceeding `max_input_length` is automatically chunked
2. **Smart Boundaries**: Chunks are split at sentence/paragraph boundaries when possible
3. **Context Preservation**: Overlapping content maintains context between chunks
4. **Result Combination**: Chunk results are intelligently combined into final output

### Long Content Features

- **Smart Chunking**: Preserves sentence and paragraph boundaries
- **Context Overlap**: Maintains coherence across chunks
- **Batch Processing**: Supports chunked content in batch API
- **Quality Assurance**: Validates chunk continuity and quality
- **Detailed Logging**: Comprehensive logging for debugging

For detailed information, see [Long Content Processing Guide](docs/LONG_CONTENT_GUIDE.md).

## 📊 Examples

### Example 1: LIMO Dataset Processing

```bash
# Step 1: Generate code-switched LIMO data
python code_switch.py run "GAIR/LIMO" \
  --split="train" \
  --n=500 \
  --output_dir="./data"

# Step 2: Train baseline model
python fine_tune.py train \
  --train_file="./data/GAIR_LIMO_train_500_original.jsonl" \
  --output_dir="./outputs/limo_baseline" \
  --run_name="limo_baseline_v1"

# Step 3: Train code-switched model
python fine_tune.py train \
  --train_file="./data/GAIR_LIMO_train_500_code_switched.jsonl" \
  --output_dir="./outputs/limo_code_switched" \
  --run_name="limo_code_switched_v1"

# Step 4: Compare models
python fine_tune.py evaluate \
  --model_dir="./outputs/limo_baseline" \
  --eval_file="./data/test.jsonl"

python fine_tune.py evaluate \
  --model_dir="./outputs/limo_code_switched" \
  --eval_file="./data/test.jsonl"
```

### Example 2: Custom Dataset

```bash
# Process custom math dataset
python code_switch.py run "your-org/custom-math-dataset" \
  --split="train" \
  --n=1000 \
  --text_column="explanation" \
  --question_column="problem" \
  --answer_column="solution"

# Fine-tune with custom parameters
python fine_tune.py train \
  --train_file="./data/your_org_custom_math_dataset_train_1000_code_switched.jsonl" \
  --output_dir="./outputs/custom_model" \
  --epochs=3 \
  --batch_size=2 \
  --max_length=4096
```

## 📁 Project Structure

```
KULLM-Pro/
├── code_switch.py              # Code switching CLI
├── fine_tune.py               # Fine-tuning CLI
├── config.yaml                # Configuration file
├── system_prompt.txt          # System prompt for code switching
├── requirements.txt           # Dependencies
├── .env.example              # Environment variables template
├── README.md                 # This file
├── LICENSE                   # License file
├── kullm_pro/                # Main package
│   ├── __init__.py
│   ├── code_switching/       # Code switching module
│   │   ├── __init__.py
│   │   ├── pipeline.py       # Main pipeline
│   │   ├── dataset_processor.py  # Dataset processing
│   │   └── openai_client.py  # OpenAI API client
│   ├── fine_tuning/          # Fine-tuning module
│   │   ├── __init__.py
│   │   ├── pipeline.py       # Training pipeline
│   │   ├── trainer.py        # LoRA trainer
│   │   └── data_processor.py # Data processing
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── logging.py        # Logging setup
│       └── helpers.py        # Helper functions
├── data/                     # Generated datasets
└── outputs/                  # Trained models
```

## 🔧 API Reference

### Code Switching Pipeline

```python
from kullm_pro.code_switching import CodeSwitchingPipeline, OpenAIConfig

# Initialize configuration
config = OpenAIConfig(
    model="o4-mini-2025-04-16",
    use_batch_api=True,
    batch_size=100
)

# Create pipeline
pipeline = CodeSwitchingPipeline(
    openai_config=config,
    system_prompt_path="system_prompt.txt",
    output_dir="./data"
)

# Process dataset
original_file, code_switched_file = await pipeline.process_dataset(
    dataset_name="GAIR/LIMO",
    split="train",
    n_samples=300
)
```

### Fine-tuning Pipeline

```python
from kullm_pro.fine_tuning import FineTuningPipeline
from kullm_pro.utils import load_config

# Load configuration
config = load_config("config.yaml")

# Create pipeline
pipeline = FineTuningPipeline(config)

# Train model
training_info = pipeline.train(
    train_file="./data/train.jsonl",
    output_dir="./outputs/my_model",
    run_name="experiment_1"
)

# Evaluate model
metrics = pipeline.evaluate(
    model_dir="./outputs/my_model",
    eval_file="./data/test.jsonl"
)
```

## 💰 Cost Estimates

### OpenAI API Costs (Batch API)

| Samples | Estimated Cost | Processing Time |
|---------|---------------|-----------------|
| 100     | $1.50 - $5    | 30-60 minutes   |
| 500     | $7.50 - $25   | 1-2 hours       |
| 1000    | $15 - $50     | 2-4 hours       |
| 5000    | $75 - $250    | 8-12 hours      |

*Costs based on o4-mini batch pricing with ~80% success rate*

### Training Costs (Local GPU)

| Model Size | GPU Memory | Training Time | Power Cost* |
|------------|------------|---------------|-------------|
| 7B (LoRA) | 24GB+      | 2-4 hours     | $2-8        |
| 13B (LoRA)| 40GB+      | 4-8 hours     | $4-16       |
| 70B (LoRA)| 80GB+      | 12-24 hours   | $12-48      |

*Estimated at $0.20/kWh for high-end GPU*

## 🚨 Important Notes

### Data Quality
- **Success Rate**: ~80% for mathematical content (OpenAI content filters)
- **Quality Control**: Manual review recommended for production use
- **Language Balance**: Adjust system prompt for desired Korean/English ratio

### Training Considerations
- **GPU Memory**: Monitor VRAM usage during training
- **Checkpoint Management**: Regular checkpoints prevent data loss
- **Hyperparameter Tuning**: Start with default values, then optimize
- **Validation**: Always use separate validation data

### Production Deployment
- **Model Serving**: Use appropriate serving frameworks (vLLM, TensorRT-LLM)
- **Monitoring**: Track model performance and drift
- **Updates**: Regular retraining with new data

## 🔍 Troubleshooting

### Common Issues

**OpenAI API Errors:**
```bash
# Check API key
echo $OPENAI_API_KEY

# Test API connection
python -c "import openai; print(openai.models.list())"
```

**GPU Memory Issues:**
```bash
# Reduce batch size
python fine_tune.py train --batch_size=1 --gradient_accumulation_steps=16

# Use gradient checkpointing
# (enabled by default in config.yaml)
```

**Import Errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import kullm_pro; print(kullm_pro.__file__)"
```

### Performance Optimization

**For Code Switching:**
- Use batch API for cost efficiency
- Process in smaller chunks for memory management
- Monitor API rate limits

**For Training:**
- Use mixed precision (bf16)
- Enable gradient checkpointing
- Optimize batch size for your GPU

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper tests
4. **Follow code style**: Use black, isort, and flake8
5. **Submit a pull request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black .
isort .
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the powerful language models and API
- **Hugging Face** for the transformers library and model hub
- **GAIR** for the LIMO dataset
- **Weights & Biases** for experiment tracking
- **The open-source community** for the amazing tools and libraries

---

**Happy training! 🚀**

For questions or support, please open an issue on GitHub.
