---
name: Bug Report
about: Create a report to help us improve KULLM-Pro
title: '[BUG] '
labels: ['bug']
assignees: ''

---

## 🐛 Bug Description

A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce

Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## ✅ Expected Behavior

A clear and concise description of what you expected to happen.

## ❌ Actual Behavior

A clear and concise description of what actually happened.

## 📸 Screenshots

If applicable, add screenshots to help explain your problem.

## 🖥️ Environment

**System Information:**
- OS: [e.g. Ubuntu 20.04, Windows 10, macOS 12.0]
- Python version: [e.g. 3.9.7]
- KULLM-Pro version: [e.g. 1.0.0]
- GPU: [e.g. NVIDIA RTX 4090, None]
- CUDA version: [e.g. 11.8, N/A]

**Package Versions:**
```bash
# Please run: pip list | grep -E "(torch|transformers|datasets|peft|accelerate)"
# and paste the output here
```

## 📋 Configuration

**Config file (config.yaml):**
```yaml
# Please share relevant parts of your config.yaml
# Remove any sensitive information like API keys
```

**Environment variables:**
```bash
# Please share relevant environment variables
# Remove any sensitive information like API keys
OPENAI_API_KEY=sk-your-api-key-here (set)
WANDB_API_KEY=... (set/not set)
```

## 📝 Command/Code

**Command that caused the issue:**
```bash
# The exact command you ran
python code_switch.py run "GAIR/LIMO" --split="train" --n=100
```

**Or code snippet:**
```python
# If using the Python API, provide the code snippet
from kullm_pro import CodeSwitchingPipeline
# ...
```

## 📊 Error Output

**Error message:**
```
# Please paste the full error message and stack trace here
```

**Log files:**
```
# If you have log files, please paste relevant parts here
# You can find logs in the output directory or console output
```

## 🔍 Additional Context

Add any other context about the problem here:
- Does this happen consistently or intermittently?
- Did this work in a previous version?
- Any workarounds you've found?
- Related issues or discussions?

## ✅ Checklist

- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided all the requested information above
- [ ] I have removed any sensitive information (API keys, personal data)
- [ ] I can reproduce this issue consistently
- [ ] I am using a supported Python version (3.8+)
