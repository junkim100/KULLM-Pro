#!/usr/bin/env python3
"""
KULLM-Pro Fine-tuning CLI

Command-line interface for LoRA (Low-Rank Adaptation) fine-tuning of language models
with advanced training features, experiment tracking, and model management.

This script provides comprehensive fine-tuning capabilities including:
- LoRA fine-tuning for efficient parameter updates
- Weights & Biases integration for experiment tracking
- Checkpoint management and resumable training
- Model evaluation and performance metrics
- Advanced training optimizations (gradient checkpointing, mixed precision)
- Support for custom datasets in JSONL format

Features:
- Configuration-driven training with YAML support
- Command-line parameter overrides
- Automatic model and tokenizer saving
- Training progress monitoring and logging
- Model information and checkpoint management
- Comprehensive error handling and validation

Example usage:
    # Basic training
    python fine_tune.py train \\
        --train_file="./data/GAIR_LIMO_train_300_code_switched.jsonl" \\
        --output_dir="./outputs/qwen_limo_cs"

    # Training with validation and custom parameters
    python fine_tune.py train \\
        --train_file="./data/train.jsonl" \\
        --val_file="./data/val.jsonl" \\
        --output_dir="./outputs/my_model" \\
        --epochs=5 --batch_size=4 --learning_rate=0.0001

    # Resume training from checkpoint
    python fine_tune.py train \\
        --train_file="./data/train.jsonl" \\
        --output_dir="./outputs/my_model" \\
        --resume_from_checkpoint="./outputs/my_model/checkpoint-1000"

    # Evaluate trained model
    python fine_tune.py evaluate \\
        --model_dir="./outputs/my_model" \\
        --eval_file="./data/test.jsonl"

    # Get model information
    python fine_tune.py info --model_dir="./outputs/my_model"

    # List available checkpoints
    python fine_tune.py list_checkpoints --output_dir="./outputs/my_model"

Requirements:
    - CUDA-compatible GPU (recommended)
    - Sufficient VRAM (24GB+ for 7B models)
    - Training data in JSONL format with 'messages' field
    - Valid config.yaml file

Output:
    - Trained LoRA adapters (adapter_model.safetensors)
    - Model configuration files
    - Tokenizer files
    - Training information and statistics
    - WandB experiment logs (if enabled)

Author: KULLM-Pro Development Team
License: MIT
"""

import sys
from typing import Optional
from pathlib import Path

import fire
from dotenv import load_dotenv

from kullm_pro.fine_tuning import FineTuningPipeline
from kullm_pro.utils import setup_logging, load_config, validate_config

# Load environment variables
load_dotenv()


class FineTuneCLI:
    """Command-line interface for fine-tuning"""

    def __init__(self):
        try:
            self.logger = setup_logging()
        except Exception as e:
            print(f"Failed to setup logging: {e}")
            import logging
            self.logger = logging.getLogger(__name__)

    def train(
        self,
        train_file: str,
        output_dir: str,
        config_file: str = "config.yaml",
        val_file: Optional[str] = None,
        run_name: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        max_length: Optional[int] = None
    ):
        """
        Train a model with LoRA fine-tuning

        Args:
            train_file: Path to training JSONL file
            output_dir: Output directory for trained model
            config_file: Path to configuration file (default: "config.yaml")
            val_file: Optional path to validation JSONL file
            run_name: WandB run name (auto-generated if not provided)
            resume_from_checkpoint: Path to checkpoint to resume training from
            model_name: Override model name from config
            epochs: Override number of epochs from config
            batch_size: Override batch size from config
            learning_rate: Override learning rate from config
            lora_r: Override LoRA rank from config
            lora_alpha: Override LoRA alpha from config
            max_length: Override max sequence length from config
        """
        self.logger.info("Starting fine-tuning")
        self.logger.info(f"Training file: {train_file}")
        self.logger.info(f"Output directory: {output_dir}")

        # Validate input files
        if not Path(train_file).exists():
            self.logger.error(f"Training file not found: {train_file}")
            sys.exit(1)

        if val_file and not Path(val_file).exists():
            self.logger.error(f"Validation file not found: {val_file}")
            sys.exit(1)

        # Load configuration
        try:
            config = load_config(config_file)
            validate_config(config)
            self.logger.info(f"Loaded configuration from {config_file}")
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_file}")
            self.logger.error("Please ensure config.yaml exists or specify a valid config file")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Failed to load or validate config: {e}")
            sys.exit(1)

        # Apply command-line overrides
        if model_name:
            config["model"]["name"] = model_name
        if epochs:
            config["training"]["epochs"] = epochs
        if batch_size:
            config["training"]["batch_size"] = batch_size
        if learning_rate:
            config["training"]["learning_rate"] = learning_rate
        if lora_r:
            config["lora"]["r"] = lora_r
        if lora_alpha:
            config["lora"]["alpha"] = lora_alpha
        if max_length:
            config["model"]["max_length"] = max_length

        # Create pipeline
        pipeline = FineTuningPipeline(config)

        # Run training
        try:
            training_info = pipeline.train(
                train_file=train_file,
                output_dir=output_dir,
                val_file=val_file,
                run_name=run_name,
                resume_from_checkpoint=resume_from_checkpoint
            )

            self.logger.info("Fine-tuning completed successfully!")
            self.logger.info(f"Model saved to: {output_dir}")

            # Print final metrics
            if "training_metrics" in training_info:
                metrics = training_info["training_metrics"]
                self.logger.info(f"Final training loss: {metrics.get('train_loss', 'N/A')}")
                self.logger.info(f"Final validation loss: {metrics.get('eval_loss', 'N/A')}")

            return training_info

        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}")

            # Provide helpful troubleshooting information
            if "does not require grad" in str(e):
                self.logger.error("This error suggests an issue with LoRA parameter setup.")
                self.logger.error("Troubleshooting suggestions:")
                self.logger.error("1. Try using a smaller model for testing (e.g., --model_name='gpt2')")
                self.logger.error("2. Reduce batch size (e.g., --batch_size=1)")
                self.logger.error("3. Check if the model supports LoRA fine-tuning")
                self.logger.error("4. Verify your training data format is correct")

            raise

    def evaluate(
        self,
        model_dir: str,
        eval_file: str,
        config_file: str = "config.yaml"
    ):
        """
        Evaluate a trained model

        Args:
            model_dir: Directory containing trained model
            eval_file: Path to evaluation JSONL file
            config_file: Path to configuration file (default: "config.yaml")
        """
        self.logger.info("Starting model evaluation")
        self.logger.info(f"Model directory: {model_dir}")
        self.logger.info(f"Evaluation file: {eval_file}")

        # Validate inputs
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        if not Path(eval_file).exists():
            raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

        # Load configuration
        try:
            config = load_config(config_file)
            self.logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise

        # Create pipeline
        pipeline = FineTuningPipeline(config)

        # Run evaluation
        try:
            metrics = pipeline.evaluate(
                model_dir=model_dir,
                eval_file=eval_file
            )

            self.logger.info("Evaluation completed successfully!")
            self.logger.info(f"Evaluation loss: {metrics.get('eval_loss', 'N/A')}")
            self.logger.info(f"Perplexity: {metrics.get('eval_perplexity', 'N/A')}")

            return metrics

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

    def info(self, model_dir: str, config_file: str = "config.yaml"):
        """
        Get information about a trained model

        Args:
            model_dir: Directory containing trained model
            config_file: Path to configuration file (default: "config.yaml")
        """
        self.logger.info(f"Getting model information for: {model_dir}")

        # Validate input
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Load configuration
        try:
            config = load_config(config_file)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
            config = {}

        # Create pipeline
        pipeline = FineTuningPipeline(config)

        # Get model info
        try:
            info = pipeline.get_model_info(model_dir)

            self.logger.info("Model Information:")
            self.logger.info(f"  Directory: {info['model_directory']}")
            self.logger.info(f"  Has LoRA adapters: {info['has_adapter_config'] and info['has_adapter_model']}")
            self.logger.info(f"  Has tokenizer: {info['has_tokenizer']}")
            self.logger.info(f"  Model files: {len(info['model_files'])}")

            if info.get("training_info"):
                training_info = info["training_info"]
                self.logger.info(f"  Training timestamp: {training_info.get('timestamp', 'N/A')}")
                self.logger.info(f"  Base model: {training_info.get('model_info', {}).get('base_model', 'N/A')}")

                if "dataset_stats" in training_info:
                    train_stats = training_info["dataset_stats"].get("train", {})
                    self.logger.info(f"  Training samples: {train_stats.get('num_samples', 'N/A')}")
                    self.logger.info(f"  Avg token length: {train_stats.get('avg_token_length', 'N/A'):.1f}")

            return info

        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            raise

    def list_checkpoints(self, output_dir: str):
        """
        List available checkpoints in an output directory

        Args:
            output_dir: Output directory to search for checkpoints
        """
        output_path = Path(output_dir)

        if not output_path.exists():
            self.logger.error(f"Output directory not found: {output_dir}")
            return []

        # Find checkpoint directories
        checkpoints = []
        for item in output_path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                checkpoints.append(item)

        # Sort by checkpoint number
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))

        self.logger.info(f"Found {len(checkpoints)} checkpoints in {output_dir}:")
        for checkpoint in checkpoints:
            self.logger.info(f"  {checkpoint.name}")

        return [str(cp) for cp in checkpoints]


def main():
    """Main entry point"""
    fire.Fire(FineTuneCLI())


if __name__ == "__main__":
    main()
