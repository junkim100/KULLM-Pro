# KULLM-Pro Scripts

This directory contains user-facing scripts for interacting with KULLM-Pro models.

## 📁 Scripts

### `chat.py`
Interactive chat interface for KULLM-Pro models with think token support.

**Features:**
- Real-time streaming generation
- Color-coded think token display
- Korean and English language support
- Production-ready inference interface

**Usage:**
```bash
# Basic usage
python scripts/chat.py --model_path outputs/your-model

# With custom settings
python scripts/chat.py \
  --model_path outputs/your-model \
  --max_new_tokens 2048 \
  --temperature 0.7 \
  --top_p 0.9
```

**Options:**
- `--model_path`: Path to the trained model
- `--max_new_tokens`: Maximum tokens to generate (default: 1024)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_p`: Top-p sampling parameter (default: 0.9)
- `--device`: Device to use (auto-detected)

**Example Session:**
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

User: 고마워!

Assistant: 천만에요! 도움이 되어서 기뻐요. 😊
```

**Features:**
- ✅ Think tokens are displayed in color for better visualization
- ✅ Streaming generation shows responses in real-time
- ✅ Natural Korean-English code-switching
- ✅ Mathematical reasoning with step-by-step explanations
- ✅ Clean separation between thinking and final answers

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.44+
- Trained KULLM-Pro model

**Notes:**
- The chat interface automatically detects and uses the model's chat template
- Think tokens are properly formatted and displayed
- Supports both Korean and English input/output
- Optimized for mathematical reasoning tasks