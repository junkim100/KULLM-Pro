# Changelog

All notable changes to KULLM-Pro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Planned features and improvements will be listed here

### Changed
- Future changes will be documented here

### Fixed
- Bug fixes will be listed here

## [1.0.0] - 2025-06-21

### Added
- **Code Switching Module**: Complete pipeline for generating Korean-English code-switched datasets
  - Support for any Hugging Face dataset with flexible parameters
  - OpenAI API integration with both regular and batch API support
  - Automatic filename generation based on dataset parameters
  - Comprehensive error handling and retry logic
  - Progress tracking and usage statistics
  
- **Fine-tuning Module**: Production-ready LoRA fine-tuning pipeline
  - LoRA (Low-Rank Adaptation) fine-tuning for efficient training
  - Support for any Hugging Face language model
  - Weights & Biases integration for experiment tracking
  - Advanced training optimizations (gradient checkpointing, mixed precision)
  - Checkpoint management and resumable training
  - Model evaluation and information utilities
  
- **Utilities Module**: Shared utilities and configuration management
  - YAML-based configuration system with validation
  - Centralized logging setup with customizable levels
  - Helper functions for file operations and data processing
  - Environment variable management with python-dotenv
  
- **Command-Line Interfaces**: User-friendly CLI tools
  - `code_switch.py`: CLI for code switching operations
  - `fine_tune.py`: CLI for fine-tuning operations
  - Python Fire integration for intuitive command structure
  - Comprehensive help and documentation
  
- **Configuration and Setup**:
  - Production-ready `config.yaml` with comprehensive settings
  - `.env.example` template for environment variables
  - `requirements.txt` with proper version constraints
  - Comprehensive `.gitignore` for ML/AI projects
  
- **Documentation**:
  - Detailed README with installation and usage instructions
  - Comprehensive API documentation and examples
  - Troubleshooting guide and best practices
  - Cost estimation and performance guidelines

### Technical Features
- **Dataset Processing**: Flexible dataset loading with configurable parameters
- **API Integration**: OpenAI Batch API support for cost-efficient processing
- **Training Optimizations**: Accelerate/DeepSpeed support for advanced training
- **Error Handling**: Robust error handling and validation throughout
- **Logging**: Structured logging with configurable levels and outputs
- **Modularity**: Clean separation of concerns with well-defined interfaces

### Supported Models
- Qwen/Qwen2.5-7B-Instruct (default)
- Any Hugging Face causal language model
- Extensible architecture for future model support

### Supported Datasets
- GAIR/LIMO (mathematical reasoning)
- Microsoft Orca Math Word Problems
- Any Hugging Face dataset with text fields
- Custom JSONL datasets

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.44+
- CUDA-compatible GPU (recommended for fine-tuning)
- OpenAI API key (for code switching)

### Performance
- Batch API processing: ~80% success rate for mathematical content
- Training efficiency: LoRA reduces trainable parameters by >99%
- Memory optimization: Gradient checkpointing and mixed precision support
- Cost efficiency: Batch API reduces OpenAI costs by ~50%

### Known Issues
- OpenAI content filters may affect success rate for certain content types
- Large models require significant GPU memory (24GB+ recommended)
- Batch API processing can take 1-24 hours depending on queue

### Breaking Changes
- This is the initial release, no breaking changes from previous versions

---

## Release Notes Format

### Added
- New features and capabilities

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Features removed in this version

### Fixed
- Bug fixes

### Security
- Security improvements and fixes

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute to this project.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
