# KULLM-Pro Examples

This directory contains example configurations, sample data, and usage examples for KULLM-Pro.

## 📁 Contents

### Configuration Files
- `config_small_model.yaml` - Configuration for testing with smaller models
- `system_prompt_simple.txt` - Simple system prompt for code switching

### Sample Data
- `sample_training_data.jsonl` - Example training data in the correct format
- `datasets_config.yaml` - Example configuration for batch processing multiple datasets

### Scripts
- `quick_start.py` - Quick start script demonstrating basic usage
- `batch_processing_example.py` - Example of batch processing multiple datasets

## 🚀 Quick Start

### 1. Test Code Switching with Small Dataset

```bash
# Use the small model configuration for testing
python code_switch.py run "GAIR/LIMO" \
  --split="train" \
  --n=5 \
  --config_file="examples/config_small_model.yaml" \
  --system_prompt="examples/system_prompt_simple.txt" \
  --output_dir="examples/outputs"
```

### 2. Test Fine-tuning with Sample Data

```bash
# Train a small model with sample data
python fine_tune.py train \
  --train_file="examples/sample_training_data.jsonl" \
  --output_dir="examples/outputs/test_model" \
  --config_file="examples/config_small_model.yaml" \
  --epochs=1 \
  --batch_size=1
```

### 3. Evaluate the Trained Model

```bash
# Evaluate the trained model
python fine_tune.py evaluate \
  --model_dir="examples/outputs/test_model" \
  --eval_file="examples/sample_training_data.jsonl" \
  --config_file="examples/config_small_model.yaml"
```

## 📊 Expected Outputs

### Code Switching Output
- `examples/outputs/GAIR_LIMO_train_5_original.jsonl` - Original data
- `examples/outputs/GAIR_LIMO_train_5_code_switched.jsonl` - Code-switched data
- `examples/outputs/processing_stats.json` - Processing statistics

### Fine-tuning Output
- `examples/outputs/test_model/` - Trained model directory
  - `adapter_config.json` - LoRA adapter configuration
  - `adapter_model.safetensors` - LoRA adapter weights
  - `tokenizer.json` - Tokenizer files
  - `training_info.json` - Training information and statistics

## 🔧 Customization

### Modify System Prompt
Edit `system_prompt_simple.txt` to change how code switching is performed:

```text
You are a helpful assistant that creates Korean-English code-switched mathematical solutions.

[Your custom instructions here]
```

### Adjust Model Configuration
Edit `config_small_model.yaml` to change model settings:

```yaml
model:
  name: "your-preferred-model"  # Change model
  max_length: 1024              # Adjust context length

training:
  epochs: 3                     # More training epochs
  batch_size: 2                 # Larger batch size
```

### Create Custom Training Data
Follow the format in `sample_training_data.jsonl`:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "System message"
    },
    {
      "role": "user", 
      "content": "User question"
    },
    {
      "role": "assistant",
      "content": "Assistant response"
    }
  ]
}
```

## 🧪 Testing Different Scenarios

### Test with Different Models
```bash
# Test with different model sizes
python fine_tune.py train \
  --train_file="examples/sample_training_data.jsonl" \
  --output_dir="examples/outputs/gpt2_model" \
  --model_name="gpt2" \
  --epochs=1
```

### Test Batch Processing
```bash
# Create datasets_config.yaml and run batch processing
python code_switch.py batch_process \
  --datasets_config="examples/datasets_config.yaml"
```

### Test with Custom Parameters
```bash
# Test with custom LoRA parameters
python fine_tune.py train \
  --train_file="examples/sample_training_data.jsonl" \
  --output_dir="examples/outputs/custom_lora" \
  --lora_r=8 \
  --lora_alpha=16 \
  --learning_rate=0.001
```

## 📝 Notes

- The small model configuration is designed for testing and development
- Sample data contains only basic mathematical problems
- For production use, use the main `config.yaml` with larger models
- Adjust batch sizes and model parameters based on your hardware capabilities

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `batch_size` to 1
   - Use smaller model (e.g., "gpt2")
   - Reduce `max_length`

2. **API Rate Limits**
   - Set `use_batch_api=false` in OpenAI config
   - Reduce `max_concurrent_requests`
   - Add delays between requests

3. **Model Loading Issues**
   - Ensure you have internet connection for model download
   - Check if model name is correct
   - Try using `trust_remote_code=True` for custom models

### Getting Help

- Check the main README.md for detailed documentation
- Review the configuration files for parameter explanations
- Run commands with `--help` for usage information
- Check the logs for detailed error messages
