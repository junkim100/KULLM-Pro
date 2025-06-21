"""
Fine-tuning pipeline for LoRA training
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from datasets import Dataset

from .data_processor import TrainingDataProcessor
from .trainer import LoRATrainer
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory

logger = get_logger("fine_tuning.pipeline")


class FineTuningPipeline:
    """Main pipeline for LoRA fine-tuning"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.lora_config = config.get("lora", {})
        self.wandb_config = config.get("wandb", {})

        # Initialize data processor
        self.data_processor = TrainingDataProcessor(
            tokenizer_name=self.model_config.get("name", "Qwen/Qwen2.5-7B-Instruct"),
            max_length=self.model_config.get("max_length", 2048)
        )

        logger.info("Initialized fine-tuning pipeline")

    def train(
        self,
        train_file: str,
        output_dir: str,
        val_file: Optional[str] = None,
        run_name: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete fine-tuning pipeline

        Args:
            train_file: Path to training JSONL file
            output_dir: Output directory for model
            val_file: Optional path to validation JSONL file
            run_name: WandB run name
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results and metrics
        """
        logger.info("Starting fine-tuning pipeline")
        logger.info(f"Training file: {train_file}")
        logger.info(f"Output directory: {output_dir}")

        # Ensure output directory exists
        ensure_directory(output_dir)

        # Step 1: Prepare datasets
        logger.info("=== Step 1: Preparing datasets ===")
        train_dataset, val_dataset = self.data_processor.prepare_datasets(
            train_file=train_file,
            val_file=val_file,
            train_split_ratio=self.config.get("data", {}).get("train_split_ratio", 0.9)
        )

        # Log dataset statistics
        train_stats = self.data_processor.get_data_stats(train_dataset)
        val_stats = self.data_processor.get_data_stats(val_dataset)

        logger.info(f"Training dataset stats: {train_stats}")
        logger.info(f"Validation dataset stats: {val_stats}")

        # Step 2: Initialize trainer
        logger.info("=== Step 2: Initializing LoRA trainer ===")
        trainer = LoRATrainer(
            model_name=self.model_config.get("name", "Qwen/Qwen2.5-7B-Instruct"),
            tokenizer=self.data_processor.tokenizer,
            lora_config=self.lora_config,
            training_config=self.training_config,
            wandb_config=self.wandb_config
        )

        # Step 3: Train model
        logger.info("=== Step 3: Training model ===")

        # Generate run name if not provided
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = Path(train_file).stem
            run_name = f"{dataset_name}_{timestamp}"

        training_metrics = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=output_dir,
            run_name=run_name,
            resume_from_checkpoint=resume_from_checkpoint
        )

        # Step 4: Save training information
        logger.info("=== Step 4: Saving training information ===")
        training_info = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "train_file": train_file,
            "val_file": val_file,
            "output_dir": output_dir,
            "run_name": run_name,
            "dataset_stats": {
                "train": train_stats,
                "validation": val_stats
            },
            "training_metrics": training_metrics,
            "model_info": {
                "base_model": self.model_config.get("name"),
                "lora_config": self.lora_config,
                "training_config": self.training_config
            }
        }

        # Save training info
        info_file = Path(output_dir) / "training_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)

        logger.info(f"Training information saved to {info_file}")
        logger.info("Fine-tuning pipeline completed successfully!")

        return training_info

    def evaluate(
        self,
        model_dir: str,
        eval_file: str
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model

        Args:
            model_dir: Directory containing trained model
            eval_file: Path to evaluation JSONL file

        Returns:
            Evaluation metrics
        """
        logger.info("Starting model evaluation")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Evaluation file: {eval_file}")

        # Load evaluation data
        eval_data = self.data_processor.load_jsonl_data(eval_file)
        formatted_data = [self.data_processor.format_conversation(example) for example in eval_data]
        eval_dataset = Dataset.from_list(formatted_data)

        # Tokenize evaluation dataset
        eval_dataset = eval_dataset.map(
            self.data_processor.tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing evaluation data"
        )

        # Initialize trainer with saved model
        trainer = LoRATrainer(
            model_name=model_dir,  # Use saved model directory
            tokenizer=self.data_processor.tokenizer,
            lora_config=self.lora_config,
            training_config=self.training_config,
            wandb_config={}  # Disable wandb for evaluation
        )

        # Load model
        trainer.load_model()

        # Create dummy trainer for evaluation
        from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

        training_args = TrainingArguments(
            output_dir="./temp_eval",
            per_device_eval_batch_size=self.training_config.get("batch_size", 2),
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.data_processor.tokenizer,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        )

        eval_trainer = Trainer(
            model=trainer.model,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.data_processor.tokenizer,
        )

        # Evaluate
        metrics = eval_trainer.evaluate()

        logger.info(f"Evaluation completed: {metrics}")
        return metrics

    def get_model_info(self, model_dir: str) -> Dict[str, Any]:
        """
        Get information about a trained model

        Args:
            model_dir: Directory containing trained model

        Returns:
            Model information
        """
        model_path = Path(model_dir)

        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Load training info if available
        info_file = model_path / "training_info.json"
        if info_file.exists():
            with open(info_file, "r", encoding="utf-8") as f:
                training_info = json.load(f)
        else:
            training_info = {}

        # Get model files
        model_files = list(model_path.glob("*"))

        info = {
            "model_directory": str(model_path),
            "model_files": [str(f.name) for f in model_files],
            "training_info": training_info,
            "has_adapter_config": (model_path / "adapter_config.json").exists(),
            "has_adapter_model": (model_path / "adapter_model.safetensors").exists(),
            "has_tokenizer": (model_path / "tokenizer.json").exists(),
        }

        return info
