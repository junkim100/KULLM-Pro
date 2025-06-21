"""
Data processing utilities for fine-tuning
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from ..utils.logging import get_logger

logger = get_logger("fine_tuning.data_processor")


class TrainingDataProcessor:
    """Processor for training data preparation"""

    def __init__(self, tokenizer_name: str, max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Initialized tokenizer: {tokenizer_name}")
        logger.info(f"Max length: {max_length}")

    def load_jsonl_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from JSONL file

        Args:
            file_path: Path to JSONL file

        Returns:
            List of loaded samples
        """
        data = []
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Training data file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue

        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data

    def format_conversation(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Format conversation for training

        Args:
            example: Training example with messages

        Returns:
            Dictionary with formatted text
        """
        messages = example["messages"]

        # Find system, user and assistant messages
        system_msg = ""
        user_msg = ""
        assistant_msg = ""

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]

        # Create input-output format with proper chat template
        if system_msg:
            text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        else:
            text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"

        return {"text": text}

    def tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """
        Tokenize examples for causal language modeling

        Args:
            examples: Batch of examples with text

        Returns:
            Tokenized examples with proper labels
        """
        # Tokenize the text
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
            add_special_tokens=True,
        )

        # For causal LM, labels are the same as input_ids
        # Copy input_ids to labels (this is the correct format for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def prepare_datasets(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        train_split_ratio: float = 0.9
    ) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets

        Args:
            train_file: Path to training JSONL file
            val_file: Optional path to validation JSONL file
            train_split_ratio: Ratio for train/val split if val_file not provided

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        logger.info("Preparing datasets...")

        # Load training data
        train_data = self.load_jsonl_data(train_file)

        # Format conversations
        formatted_data = [self.format_conversation(example) for example in train_data]

        # Create dataset
        full_dataset = Dataset.from_list(formatted_data)

        # Handle validation data
        if val_file:
            # Load separate validation file
            val_data = self.load_jsonl_data(val_file)
            val_formatted = [self.format_conversation(example) for example in val_data]
            val_dataset = Dataset.from_list(val_formatted)
            train_dataset = full_dataset
        else:
            # Split training data
            train_size = int(train_split_ratio * len(full_dataset))
            train_dataset = full_dataset.select(range(train_size))
            val_dataset = full_dataset.select(range(train_size, len(full_dataset)))

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        # Tokenize datasets
        logger.info("Tokenizing datasets...")
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training data"
        )

        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation data"
        )

        return train_dataset, val_dataset

    def get_data_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get statistics about the dataset

        Args:
            dataset: Tokenized dataset

        Returns:
            Dictionary with statistics
        """
        if len(dataset) == 0:
            return {"num_samples": 0}

        # Calculate token length statistics
        token_lengths = [len(example["input_ids"]) for example in dataset]

        stats = {
            "num_samples": len(dataset),
            "avg_token_length": sum(token_lengths) / len(token_lengths),
            "min_token_length": min(token_lengths),
            "max_token_length": max(token_lengths),
            "total_tokens": sum(token_lengths)
        }

        return stats

    def save_tokenizer(self, output_dir: str):
        """
        Save tokenizer to output directory

        Args:
            output_dir: Directory to save tokenizer
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Tokenizer saved to {output_path}")
