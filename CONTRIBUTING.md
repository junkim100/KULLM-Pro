# Contributing to KULLM-Pro

Thank you for your interest in contributing to KULLM-Pro! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

We welcome contributions in many forms:
- 🐛 Bug reports and fixes
- ✨ New features and enhancements
- 📚 Documentation improvements
- 🧪 Test coverage improvements
- 💡 Ideas and suggestions

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-compatible GPU (for testing fine-tuning features)
- OpenAI API key (for testing code switching features)

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/junkim100/KULLM-Pro.git
   cd KULLM-Pro
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies:**
   ```bash
   pip install pytest black isort flake8 pre-commit
   ```

5. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

6. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 📝 Development Guidelines

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **Type hints** for better code documentation

Run these tools before submitting:
```bash
black .
isort .
flake8 .
```

### Commit Messages

Use clear, descriptive commit messages following this format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

Examples:
```
feat(code_switching): add support for custom system prompts
fix(fine_tuning): resolve LoRA configuration validation issue
docs(readme): update installation instructions
```

### Testing

1. **Run existing tests:**
   ```bash
   pytest tests/
   ```

2. **Add tests for new features:**
   - Create test files in the `tests/` directory
   - Follow the naming convention: `test_*.py`
   - Aim for good test coverage

3. **Test your changes:**
   ```bash
   # Test code switching (requires OpenAI API key)
   python code_switch.py run "GAIR/LIMO" --split="train" --n=2

   # Test fine-tuning (requires GPU)
   python fine_tune.py train --train_file="./data/test.jsonl" --output_dir="./test_output"
   ```

## 🔄 Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes:**
   ```bash
   # Run tests
   pytest tests/

   # Check code style
   black --check .
   isort --check-only .
   flake8 .
   ```

4. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat(scope): your descriptive message"
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request:**
   - Use the PR template
   - Provide a clear description of changes
   - Link any related issues
   - Request review from maintainers

## 🐛 Reporting Issues

When reporting issues, please include:

1. **Environment information:**
   - Python version
   - Operating system
   - GPU information (if relevant)
   - Package versions

2. **Steps to reproduce:**
   - Clear, step-by-step instructions
   - Sample code or commands
   - Expected vs. actual behavior

3. **Additional context:**
   - Error messages and stack traces
   - Log files (if applicable)
   - Screenshots (if relevant)

Use our issue templates when available.

## 💡 Feature Requests

For feature requests:

1. **Check existing issues** to avoid duplicates
2. **Describe the problem** you're trying to solve
3. **Propose a solution** with implementation details
4. **Consider alternatives** and their trade-offs
5. **Discuss impact** on existing functionality

## 📚 Documentation

Help improve our documentation:

- **README.md**: Installation, usage, examples
- **Code comments**: Docstrings and inline comments
- **API documentation**: Function and class documentation
- **Tutorials**: Step-by-step guides
- **Examples**: Sample configurations and use cases

## 🏷️ Release Process

Releases follow semantic versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the README and code comments

## 🙏 Recognition

Contributors will be recognized in:
- CHANGELOG.md for their contributions
- GitHub contributors list
- Special mentions for significant contributions

## 📄 License

By contributing to KULLM-Pro, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to KULLM-Pro! 🚀
