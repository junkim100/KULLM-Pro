"""
LoRA trainer for fine-tuning language models
"""

import os
import torch
from typing import Dict, Any, Optional
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import wandb

from ..utils.logging import get_logger

logger = get_logger("fine_tuning.trainer")

# Fix for compatibility issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Monkey patch for accelerate compatibility
try:
    import accelerate
    if hasattr(accelerate.Accelerator, "unwrap_model"):
        original_unwrap_model = accelerate.Accelerator.unwrap_model

        def patched_unwrap_model(self, model, keep_torch_compile=None):
            if keep_torch_compile is not None:
                # Ignore the keep_torch_compile parameter for older accelerate versions
                return original_unwrap_model(self, model)
            return original_unwrap_model(self, model)

        accelerate.Accelerator.unwrap_model = patched_unwrap_model
except ImportError:
    logger.warning("Accelerate not available, skipping compatibility patch")


class LoRATrainer:
    """LoRA trainer for fine-tuning language models"""

    def __init__(
        self,
        model_name: str,
        tokenizer,
        lora_config: Dict[str, Any],
        training_config: Dict[str, Any],
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.lora_config = lora_config
        self.training_config = training_config
        self.wandb_config = wandb_config or {}

        self.model = None
        self.trainer = None

        logger.info(f"Initialized LoRA trainer for model: {model_name}")

    def load_model(self) -> AutoModelForCausalLM:
        """
        Load and prepare model with LoRA

        Returns:
            Model with LoRA adapters
        """
        logger.info(f"Loading model: {self.model_name}")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",  # Avoid flash attention issues
            trust_remote_code=True,
        )

        # Ensure model is in training mode before applying LoRA
        model.train()

        # Enable gradient computation for all parameters initially
        for param in model.parameters():
            param.requires_grad = True

        # Setup LoRA
        logger.info("Setting up LoRA configuration...")

        # Get model-specific target modules
        target_modules = self._get_target_modules(model)
        logger.info(f"Using target modules: {target_modules}")

        lora_config = LoraConfig(
            r=self.lora_config.get("r", 16),
            lora_alpha=self.lora_config.get("alpha", 32),
            target_modules=target_modules,
            lora_dropout=self.lora_config.get("dropout", 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Ensure model is in training mode and gradients are enabled
        model.train()

        # Enable gradients for trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.requires_grad_(True)
                logger.info(f"Enabled gradients for: {name}")

        # Verify we have trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise ValueError("No trainable parameters found! LoRA setup may have failed.")

        logger.info(f"Total trainable parameters: {trainable_params:,}")

        self.model = model

        # Final verification
        self._verify_model_setup()

        return model

    def _verify_model_setup(self):
        """Verify that the model is properly set up for training"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Check if model is in training mode
        if not self.model.training:
            logger.warning("Model not in training mode, setting to training mode")
            self.model.train()

        # Verify trainable parameters
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)

        if not trainable_params:
            raise ValueError("No trainable parameters found!")

        logger.info(f"Found {len(trainable_params)} trainable parameter groups")
        logger.debug(f"Trainable parameters: {trainable_params[:5]}...")  # Show first 5

    def _get_target_modules(self, model):
        """
        Get appropriate target modules for LoRA based on the model architecture

        Args:
            model: The loaded model

        Returns:
            List of target module names for LoRA
        """
        # Get all module names
        module_names = set()
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                module_names.add(name.split('.')[-1])

        logger.info(f"Available linear modules: {sorted(module_names)}")

        # Model-specific target modules
        model_name_lower = self.model_name.lower()

        if "qwen" in model_name_lower:
            # Qwen models
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "llama" in model_name_lower:
            # LLaMA models
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt" in model_name_lower:
            # GPT models (GPT-2, etc.)
            target_modules = ["c_attn", "c_proj", "c_fc"]
        elif "bert" in model_name_lower:
            # BERT models
            target_modules = ["query", "key", "value", "dense"]
        else:
            # Generic approach - find common attention and MLP modules
            common_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense", "linear"]
            target_modules = [mod for mod in common_modules if mod in module_names]

            # If no common modules found, use user-specified or fall back to any linear layer
            if not target_modules:
                if "target_modules" in self.lora_config:
                    target_modules = self.lora_config["target_modules"]
                else:
                    # Use the first few linear modules found
                    target_modules = list(module_names)[:4]

        # Filter to only include modules that actually exist in the model
        available_targets = [mod for mod in target_modules if mod in module_names]

        if not available_targets:
            raise ValueError(
                f"No valid target modules found for model {self.model_name}. "
                f"Available modules: {sorted(module_names)}. "
                f"Tried: {target_modules}"
            )

        return available_targets

    def setup_training_arguments(self, output_dir: str) -> TrainingArguments:
        """
        Setup training arguments

        Args:
            output_dir: Output directory for model checkpoints

        Returns:
            TrainingArguments instance
        """
        logger.info("Setting up training arguments...")

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.training_config.get("epochs", 3),
            per_device_train_batch_size=self.training_config.get("batch_size", 2),
            per_device_eval_batch_size=self.training_config.get("batch_size", 2),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 8),
            learning_rate=self.training_config.get("learning_rate", 2e-4),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            warmup_ratio=self.training_config.get("warmup_ratio", 0.1),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "cosine"),
            logging_steps=self.training_config.get("logging_steps", 10),
            eval_steps=self.training_config.get("eval_steps", 100),
            save_steps=self.training_config.get("save_steps", 500),
            save_total_limit=self.training_config.get("save_total_limit", 3),
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=self.training_config.get("load_best_model_at_end", True),
            metric_for_best_model=self.training_config.get("metric_for_best_model", "eval_loss"),
            greater_is_better=False,
            report_to="wandb" if self.wandb_config.get("enabled", False) else "none",
            dataloader_num_workers=self.training_config.get("dataloader_num_workers", 0),
            fp16=False,
            bf16=self.training_config.get("bf16", True),
            gradient_checkpointing=self.training_config.get("gradient_checkpointing", True),
            optim=self.training_config.get("optim", "adamw_torch"),
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )

        return training_args

    def setup_trainer(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: str
    ) -> Trainer:
        """
        Setup Hugging Face Trainer

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory

        Returns:
            Configured Trainer instance
        """
        if self.model is None:
            self.load_model()

        # Setup training arguments
        training_args = self.setup_training_arguments(output_dir)

        # Custom data collator for causal language modeling
        def data_collator(features):
            """Custom data collator that handles padding properly"""
            import torch

            # Extract input_ids and labels
            input_ids = [f["input_ids"] for f in features]
            labels = [f["labels"] for f in features]

            # Pad sequences
            batch = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                max_length=2048,  # Use fixed max length
                return_tensors="pt",
                pad_to_multiple_of=8,
            )

            # Pad labels manually
            max_length = batch["input_ids"].shape[1]
            padded_labels = []

            for label_seq in labels:
                # Pad with -100 (ignore index)
                padded = label_seq + [-100] * (max_length - len(label_seq))
                padded_labels.append(padded)

            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

            return batch

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        self.trainer = trainer
        return trainer

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: str,
        run_name: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for model
            run_name: WandB run name
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training metrics
        """
        logger.info("Starting training...")

        # Initialize WandB if enabled
        if self.wandb_config.get("enabled", False):
            wandb.init(
                project=self.wandb_config.get("project", "kullm-pro-finetuning"),
                name=run_name,
                config={
                    "model_name": self.model_name,
                    "lora_config": self.lora_config,
                    "training_config": self.training_config,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                },
            )
            logger.info(f"WandB initialized: {self.wandb_config.get('project')}/{run_name}")

        # Setup trainer
        trainer = self.setup_trainer(train_dataset, val_dataset, output_dir)

        try:
            # Train
            if resume_from_checkpoint:
                logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                trainer.train()

            # Save model
            logger.info(f"Saving model to {output_dir}")
            trainer.save_model()

            # Save tokenizer
            self.tokenizer.save_pretrained(output_dir)

            # Get final metrics
            final_metrics = {}
            if hasattr(trainer.state, "log_history") and trainer.state.log_history:
                final_metrics = trainer.state.log_history[-1]
                logger.info(f"Final training metrics: {final_metrics}")

            logger.info("Training completed successfully!")
            return final_metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Clean up WandB
            if self.wandb_config.get("enabled", False):
                wandb.finish()
                logger.info("WandB session finished")

    def evaluate(self, eval_dataset: Dataset) -> Dict[str, Any]:
        """
        Evaluate the model

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")

        logger.info("Evaluating model...")
        metrics = self.trainer.evaluate(eval_dataset)
        logger.info(f"Evaluation metrics: {metrics}")

        return metrics
