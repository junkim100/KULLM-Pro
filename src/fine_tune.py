#!/usr/bin/env python3
"""
KULLM Pro Fine-tuning Module

This module provides fine-tuning functionality for reasoning models using LoRA (Low-Rank Adaptation)
with think tokens. It supports Accelerate for distributed training, Weights & Biases for experiment
tracking, and proper checkpoint management.

Features:
- LoRA fine-tuning for efficient parameter updates
- Think token integration for reasoning models
- Accelerate library for distributed training optimization
- Weights & Biases integration for experiment tracking
- Checkpoint saving and resumable training
- Configuration-based training parameters
- Python Fire CLI interface

Example usage:
    python src/fine_tune.py --data_file="path/to/training_data.jsonl" --model_name="Qwen/Qwen2.5-7B-Instruct"
    python src/fine_tune.py --data_file="./data/train.jsonl" --output_dir="./outputs/my_model"
"""

import json
import logging
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import fire
import torch
import wandb
from dotenv import load_dotenv
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_processing import load_jsonl, validate_jsonl_format
from utils.model_utils import (
    setup_tokenizer_with_think_tokens,
    load_model_and_tokenizer,
    print_model_info,
    check_gpu_availability,
)
from utils.clean_tokenizer import clean_tokenizer_for_training

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FineTuningPipeline:
    """
    Pipeline for fine-tuning reasoning models with LoRA and think tokens.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the fine-tuning pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.accelerator = Accelerator()

        # Setup wandb if enabled
        if self.config.get("wandb", {}).get("enabled", True):
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if wandb_api_key:
                wandb.login(key=wandb_api_key)
            else:
                logger.warning("WANDB_API_KEY not found. Wandb logging may not work.")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(
                f"Config file not found: {config_path}. Using default config."
            )
            return self.get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using default config.")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "model": {
                "name": "Qwen/Qwen2.5-7B-Instruct",
                "max_length": 2048,
                "torch_dtype": "float16",
            },
            "training": {
                "num_train_epochs": 3,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "lr_scheduler_type": "cosine",
                "save_steps": 500,
                "eval_steps": 500,
                "logging_steps": 10,
                "save_total_limit": 3,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
                "fp16": True,
                "gradient_checkpointing": True,
                "dataloader_pin_memory": False,
                "remove_unused_columns": False,
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            "wandb": {
                "project": "kullm-pro-reasoning",
                "enabled": True,
                "entity": None,
                "tags": ["reasoning", "think-tokens"],
            },
            "dataset": {"think_token_start": "<think>", "think_token_end": "</think>"},
        }

    def format_training_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format training data for reasoning models with think tokens and chat template.

        For LIMO dataset: Format as chat conversation with think tokens
        For other datasets: Use existing format or apply think token wrapping

        Args:
            data: Raw training data

        Returns:
            Formatted training data
        """
        formatted_data = []
        think_start = self.config["dataset"]["think_token_start"]
        think_end = self.config["dataset"]["think_token_end"]

        for sample in data:
            # Handle LIMO dataset format specifically
            if "solution" in sample and "answer" in sample:
                question = sample.get("question", "").strip()
                solution = sample.get("solution", "").strip()
                answer = sample.get("answer", "").strip()

                if question and solution and answer:
                    # Format as <think>\n\nsolution\n\n</think>\n\nanswer
                    assistant_response = (
                        f"{think_start}\n\n{solution}\n\n{think_end}\n\n{answer}"
                    )

                    # Create chat messages
                    messages = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": assistant_response},
                    ]

                    formatted_sample = {
                        "messages": messages,
                        "text": assistant_response,  # For backward compatibility
                    }
                    formatted_data.append(formatted_sample)

            # Handle other formats
            elif "text" in sample:
                formatted_data.append(sample)
            elif "input" in sample and "output" in sample:
                formatted_data.append(sample)
            else:
                logger.warning(f"Unknown sample format: {list(sample.keys())}")

        logger.info(f"Formatted {len(formatted_data)} training samples")
        return formatted_data

    def prepare_dataset(
        self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer
    ) -> Dataset:
        """
        Prepare dataset for training with proper chat template formatting.

        Args:
            data: Formatted training data
            tokenizer: Tokenizer to use

        Returns:
            Prepared dataset
        """

        def tokenize_function(examples):
            texts = []

            # Process each example
            for i in range(len(examples.get("messages", examples.get("text", [])))):
                if "messages" in examples and i < len(examples["messages"]):
                    # Use chat template for conversation format
                    messages = examples["messages"][i]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    texts.append(text)
                elif "text" in examples and i < len(examples["text"]):
                    # Use text directly
                    texts.append(examples["text"][i])
                else:
                    # Fallback
                    texts.append("")

            # Tokenize the texts
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config["model"]["max_length"],
                return_tensors=None,
            )

            # For causal language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)

        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        logger.info(f"Prepared dataset with {len(tokenized_dataset)} samples")
        return tokenized_dataset

    def setup_lora_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Setup LoRA configuration for the model.

        Args:
            model: Base model

        Returns:
            LoRA-enabled model
        """
        # Enable gradient checkpointing if specified
        if self.config["training"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()

        # Prepare model for k-bit training if using quantization
        if hasattr(model, "config") and getattr(
            model.config, "quantization_config", None
        ):
            model = prepare_model_for_kbit_training(model)

        # Setup LoRA configuration
        lora_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["alpha"],
            lora_dropout=self.config["lora"]["dropout"],
            bias=self.config["lora"]["bias"],
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config["lora"]["target_modules"],
        )

        # Apply LoRA to model
        model = get_peft_model(model, lora_config)

        # Ensure model is in training mode
        model.train()

        # Print trainable parameters
        model.print_trainable_parameters()

        return model

    def train(
        self,
        data_file: str,
        output_dir: str,
        model_name: Optional[str] = None,
        run_name: Optional[str] = None,
        val_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model with LoRA fine-tuning.

        Args:
            data_file: Path to training data JSONL file
            output_dir: Output directory for trained model
            model_name: Model name (overrides config)
            run_name: Wandb run name
            val_file: Optional validation file

        Returns:
            Training information dictionary
        """
        logger.info("Starting fine-tuning")
        logger.info(f"Training file: {data_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(
            f"Training type: {'LoRA fine-tuning' if self.config['lora'].get('enabled', True) else 'Full fine-tuning'}"
        )

        # Check GPU availability
        gpu_info = check_gpu_availability()
        logger.info(f"GPU Info: {gpu_info}")

        # Load training data
        train_data = load_jsonl(data_file)
        if not validate_jsonl_format(train_data):
            raise ValueError("Invalid training data format")

        # Load validation data if provided
        val_data = None
        if val_file:
            val_data = load_jsonl(val_file)
            if not validate_jsonl_format(val_data):
                raise ValueError("Invalid validation data format")

        # Use model name from parameter or config
        model_name = model_name or self.config["model"]["name"]

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            think_start=self.config["dataset"]["think_token_start"],
            think_end=self.config["dataset"]["think_token_end"],
        )

        # Print model information
        print_model_info(model, tokenizer)

        # Setup LoRA if enabled, otherwise prepare for full fine-tuning
        is_lora_enabled = self.config["lora"].get("enabled", True)
        training_type = "lora" if is_lora_enabled else "full"

        if is_lora_enabled:
            model = self.setup_lora_model(model)
        else:
            # Full fine-tuning: enable gradient checkpointing if specified
            if self.config["training"]["gradient_checkpointing"]:
                model.gradient_checkpointing_enable()
            model.train()
            print(
                f"Full fine-tuning enabled - all {model.num_parameters():,} parameters will be trained"
            )

        # Format training data
        formatted_train_data = self.format_training_data(train_data)
        train_dataset = self.prepare_dataset(formatted_train_data, tokenizer)

        # Prepare validation dataset if available
        eval_dataset = None
        if val_data:
            formatted_val_data = self.format_training_data(val_data)
            eval_dataset = self.prepare_dataset(formatted_val_data, tokenizer)

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training"][
                "per_device_train_batch_size"
            ],
            per_device_eval_batch_size=self.config["training"][
                "per_device_eval_batch_size"
            ],
            gradient_accumulation_steps=self.config["training"][
                "gradient_accumulation_steps"
            ],
            learning_rate=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
            warmup_ratio=self.config["training"]["warmup_ratio"],
            lr_scheduler_type=self.config["training"]["lr_scheduler_type"],
            save_steps=self.config["training"]["save_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            logging_steps=self.config["training"]["logging_steps"],
            save_total_limit=self.config["training"]["save_total_limit"],
            load_best_model_at_end=self.config["training"]["load_best_model_at_end"],
            metric_for_best_model=self.config["training"]["metric_for_best_model"],
            greater_is_better=self.config["training"]["greater_is_better"],
            eval_strategy=self.config["training"]["eval_strategy"],
            save_strategy=self.config["training"]["save_strategy"],
            fp16=self.config["training"]["fp16"],
            gradient_checkpointing=False,  # Handled manually in LoRA setup
            dataloader_pin_memory=self.config["training"]["dataloader_pin_memory"],
            remove_unused_columns=self.config["training"]["remove_unused_columns"],
            report_to="wandb" if self.config["wandb"]["enabled"] else None,
            run_name=run_name
            or f"kullm-pro-{training_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )

        # Setup data collator with proper padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
        )

        # Initialize wandb if enabled
        if self.config["wandb"]["enabled"]:
            # Prepare tags with training type
            base_tags = self.config["wandb"].get("tags", [])
            training_tags = base_tags + [training_type, f"{training_type}-fine-tuning"]

            # Prepare config with training type info
            wandb_config = {
                "model_name": model_name,
                "training_type": training_type,
                "is_lora": is_lora_enabled,
                "training_config": self.config["training"],
                "dataset_size": len(train_dataset),
            }

            # Add LoRA config only if using LoRA
            if is_lora_enabled:
                wandb_config["lora_config"] = self.config["lora"]

            wandb.init(
                project=self.config["wandb"]["project"],
                entity=self.config["wandb"].get("entity"),
                name=training_args.run_name,
                tags=training_tags,
                config=wandb_config,
            )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Start training
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save the final model
        trainer.save_model()
        trainer.save_state()

        # Save tokenizer
        tokenizer.save_pretrained(output_dir)

        # Log final metrics
        logger.info("Training completed!")
        logger.info(f"Final training loss: {train_result.training_loss}")

        # Finish wandb run
        if self.config["wandb"]["enabled"]:
            wandb.finish()

        return {
            "model_path": output_dir,
            "training_loss": train_result.training_loss,
            "training_steps": train_result.global_step,
            "status": "completed",
        }


class FineTuneCLI:
    """
    Command-line interface for fine-tuning using Python Fire.
    """

    def __init__(self):
        """Initialize the CLI."""
        self.logger = logger

    def train(
        self,
        data_file: str,
        output_dir: str = "./outputs/kullm_pro_model",
        model_name: Optional[str] = None,
        config_file: str = "config.yaml",
        run_name: Optional[str] = None,
        val_file: Optional[str] = None,
    ):
        """
        Train a model with LoRA fine-tuning.

        Args:
            data_file: Path to training data JSONL file
            output_dir: Output directory for trained model (default: "./outputs/kullm_pro_model")
            model_name: Model name to fine-tune (overrides config)
            config_file: Path to configuration file (default: "config.yaml")
            run_name: Wandb run name (auto-generated if not provided)
            val_file: Optional path to validation JSONL file

        Example:
            python src/fine_tune.py train --data_file="./data/train.jsonl" --output_dir="./outputs/my_model"
        """
        # Create pipeline
        pipeline = FineTuningPipeline(config_path=config_file)

        # Run training
        try:
            result = pipeline.train(
                data_file=data_file,
                output_dir=output_dir,
                model_name=model_name,
                run_name=run_name,
                val_file=val_file,
            )

            self.logger.info("Fine-tuning completed successfully!")
            self.logger.info(f"Model saved to: {result['model_path']}")
            self.logger.info(f"Training loss: {result['training_loss']}")

            return result

        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}")
            raise

    def evaluate(
        self, model_dir: str, eval_file: str, config_file: str = "config.yaml"
    ):
        """
        Evaluate a trained model.

        Args:
            model_dir: Directory containing the trained model
            eval_file: Path to evaluation data JSONL file
            config_file: Path to configuration file

        Example:
            python src/fine_tune.py evaluate --model_dir="./outputs/my_model" --eval_file="./data/test.jsonl"
        """
        # TODO: Implement evaluation functionality
        self.logger.info("Evaluation functionality not yet implemented")
        return {"status": "not_implemented"}


def main():
    """Main entry point for the CLI."""
    fire.Fire(FineTuneCLI)


if __name__ == "__main__":
    main()
