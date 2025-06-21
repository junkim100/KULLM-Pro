#!/usr/bin/env python3
"""
KULLM-Pro Code Switching CLI

Command-line interface for generating Korean-English code-switched datasets from
Hugging Face datasets using OpenAI's language models.

This script provides a user-friendly interface to:
- Load any Hugging Face dataset with flexible parameters
- Generate code-switched versions using OpenAI API (regular or batch)
- Save both original and code-switched data in training-ready format
- Process multiple datasets in batch mode
- Generate descriptive filenames automatically

Features:
- Support for any Hugging Face dataset
- Configurable dataset parameters (split, subset, sample count)
- OpenAI Batch API integration for cost efficiency
- Automatic error handling and retry logic
- Progress tracking and usage statistics
- Flexible output directory management

Example usage:
    # Basic code switching
    python code_switch.py run "GAIR/LIMO" --split="train" --n=300

    # With custom parameters
    python code_switch.py run "microsoft/orca-math-word-problems-200k" \\
        --split="train" --subset="default" --n=1000 --output_dir="./data"

    # Batch processing multiple datasets
    python code_switch.py batch_process --datasets_config="datasets.yaml"

    # Get help
    python code_switch.py --help
    python code_switch.py run --help

Requirements:
    - OpenAI API key (set in OPENAI_API_KEY environment variable)
    - Internet connection for dataset download and API calls
    - Sufficient disk space for dataset storage

Output:
    - {dataset_name}_{split}_{subset}_{n_samples}_original.jsonl
    - {dataset_name}_{split}_{subset}_{n_samples}_code_switched.jsonl
    - processing_stats.json (processing statistics)
    - failed_items.json (if any items fail processing)

Author: KULLM-Pro Development Team
License: MIT
"""

import asyncio
import sys
from typing import Optional

import fire
from dotenv import load_dotenv

from kullm_pro.code_switching import CodeSwitchingPipeline, OpenAIConfig
from kullm_pro.utils import setup_logging, load_config

# Load environment variables
load_dotenv()


class CodeSwitchCLI:
    """
    Command-line interface for code switching operations.

    This class provides methods for processing datasets and generating
    code-switched versions using OpenAI's language models. It supports
    both single dataset processing and batch processing of multiple datasets.

    Attributes:
        logger: Configured logger instance for operation tracking

    Methods:
        run: Process a single dataset for code switching
        batch_process: Process multiple datasets from configuration file
    """

    def __init__(self):
        """
        Initialize the CLI with proper logging setup.

        Sets up logging configuration and handles any initialization errors
        gracefully by falling back to a basic logger.
        """
        try:
            self.logger = setup_logging()
        except Exception as e:
            print(f"Failed to setup logging: {e}")
            import logging
            self.logger = logging.getLogger(__name__)

    def run(
        self,
        dataset: str,
        split: str = "train",
        subset: Optional[str] = None,
        n: Optional[int] = None,
        output_dir: str = "./data",
        config_file: str = "config.yaml",
        system_prompt: str = "system_prompt.txt",
        use_batch_api: bool = True,
        model: str = "o4-mini-2025-04-16",
        skip_original: bool = False,
        text_column: str = "solution",
        question_column: str = "question",
        solution_column: str = "solution",
        answer_column: str = "answer"
    ):
        """
        Generate code-switched dataset from Hugging Face dataset

        Args:
            dataset: Hugging Face dataset name (e.g., "GAIR/LIMO")
            split: Dataset split to use (default: "train")
            subset: Dataset subset if applicable (default: None)
            n: Number of samples to process (shortest texts, default: None for all)
            output_dir: Output directory for generated files (default: "./data")
            config_file: Path to configuration file (default: "config.yaml")
            system_prompt: Path to system prompt file (default: "system_prompt.txt")
            use_batch_api: Use OpenAI Batch API for cost efficiency (default: True)
            model: OpenAI model to use (default: "o4-mini-2025-04-16")
            skip_original: Skip saving original dataset (default: False)
            text_column: Column name for text length sorting (default: "solution")
            question_column: Column name for questions (default: "question")
            solution_column: Column name for solutions (default: "solution")
            answer_column: Column name for answers (default: "answer")
        """
        self.logger.info("Starting code switching pipeline")
        self.logger.info(f"Dataset: {dataset}")
        self.logger.info(f"Split: {split}")
        self.logger.info(f"Subset: {subset}")
        self.logger.info(f"Samples: {n}")

        # Load configuration
        try:
            config = load_config(config_file)
            self.logger.info(f"Loaded configuration from {config_file}")
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_file}")
            self.logger.error("Please ensure config.yaml exists or specify a valid config file")
            sys.exit(1)
        except Exception as e:
            self.logger.warning(f"Could not load config file {config_file}: {e}")
            self.logger.warning("Using default configuration")
            config = {}

        # Set up OpenAI configuration
        openai_config_dict = config.get("openai", {})
        # Override with command line arguments
        openai_config_dict.update({
            "model": model,
            "use_batch_api": use_batch_api
        })
        openai_config = OpenAIConfig(**openai_config_dict)

        # Create pipeline
        pipeline = CodeSwitchingPipeline(
            openai_config=openai_config,
            system_prompt_path=system_prompt,
            output_dir=output_dir
        )

        # Run the pipeline
        try:
            original_file, code_switched_file = asyncio.run(
                pipeline.process_dataset(
                    dataset_name=dataset,
                    split=split,
                    subset=subset,
                    n_samples=n,
                    text_column=text_column,
                    question_column=question_column,
                    solution_column=solution_column,
                    answer_column=answer_column,
                    skip_original=skip_original
                )
            )

            self.logger.info("Code switching completed successfully!")
            self.logger.info(f"Original file: {original_file}")
            self.logger.info(f"Code-switched file: {code_switched_file}")

            return {
                "original_file": original_file,
                "code_switched_file": code_switched_file,
                "status": "success"
            }

        except Exception as e:
            self.logger.error(f"Code switching failed: {e}")
            raise

    def batch_process(
        self,
        datasets_config: str,
        **kwargs
    ):
        """
        Process multiple datasets from a configuration file

        Args:
            datasets_config: Path to JSON/YAML file with dataset configurations
            **kwargs: Additional arguments to override for all datasets
        """
        import json
        import yaml
        from pathlib import Path

        config_path = Path(datasets_config)

        # Load datasets configuration
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                datasets = json.load(f)
        else:
            with open(config_path, 'r') as f:
                datasets = yaml.safe_load(f)

        results = []

        for dataset_config in datasets:
            # Merge with command line overrides
            merged_config = {**dataset_config, **kwargs}

            self.logger.info(f"Processing dataset: {merged_config['dataset']}")

            try:
                result = self.run(**merged_config)
                results.append({
                    "dataset": merged_config['dataset'],
                    "result": result
                })
            except Exception as e:
                self.logger.error(f"Failed to process {merged_config['dataset']}: {e}")
                results.append({
                    "dataset": merged_config['dataset'],
                    "error": str(e)
                })

        return results


def main():
    """Main entry point"""
    fire.Fire(CodeSwitchCLI())


if __name__ == "__main__":
    main()
