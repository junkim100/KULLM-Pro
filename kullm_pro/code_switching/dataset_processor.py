"""
Dataset processing utilities for code switching
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from datasets import load_dataset
import pandas as pd

from ..utils.logging import get_logger
from ..utils.helpers import generate_filename, ensure_directory

logger = get_logger("code_switching.dataset_processor")


class DatasetProcessor:
    """Processor for Hugging Face datasets"""

    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        ensure_directory(self.output_dir)

    def load_and_filter_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
        n_samples: Optional[int] = None,
        text_column: str = "solution",
        sort_by_length: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load dataset from Hugging Face and filter for shortest text samples
        
        Args:
            dataset_name: Name of the dataset (e.g., "GAIR/LIMO")
            split: Dataset split (e.g., "train", "validation", "test")
            subset: Dataset subset (if applicable)
            n_samples: Number of samples to select (shortest if sort_by_length=True)
            text_column: Column to use for length calculation
            sort_by_length: Whether to sort by text length
            
        Returns:
            List of filtered dataset samples
        """
        logger.info(f"Loading dataset: {dataset_name}, split: {split}, subset: {subset}")

        try:
            # Load the dataset
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
                
            logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")

            # Convert to pandas for easier manipulation
            df = dataset.to_pandas()

            # Calculate text lengths if sorting is requested
            if sort_by_length and text_column in df.columns:
                df[f"{text_column}_length"] = df[text_column].str.len()
                df = df.sort_values(f"{text_column}_length")
                logger.info(f"Sorted by {text_column} length")

            # Select n_samples if specified
            if n_samples and n_samples < len(df):
                df = df.head(n_samples)
                logger.info(f"Selected {n_samples} samples")
                
                if sort_by_length and text_column in df.columns:
                    logger.info(
                        f"Text length range: {df[f'{text_column}_length'].min()} - {df[f'{text_column}_length'].max()}"
                    )

            # Convert to list of dictionaries
            selected_data = df.to_dict("records")
            
            # Clean up temporary length column
            for item in selected_data:
                if f"{text_column}_length" in item:
                    del item[f"{text_column}_length"]

            return selected_data

        except Exception as e:
            logger.error(f"Error loading/filtering dataset {dataset_name}: {e}")
            raise

    def save_original_data(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str,
        split: str,
        subset: Optional[str] = None,
        n_samples: Optional[int] = None,
        question_column: str = "question",
        solution_column: str = "solution", 
        answer_column: str = "answer"
    ) -> str:
        """
        Save original dataset as JSONL in training format
        
        Args:
            data: Dataset samples
            dataset_name: Name of the dataset
            split: Dataset split
            subset: Dataset subset
            n_samples: Number of samples
            question_column: Column name for questions
            solution_column: Column name for solutions
            answer_column: Column name for answers
            
        Returns:
            Path to saved file
        """
        filename = generate_filename(
            dataset_name, split, subset, n_samples, "original"
        )
        filepath = self.output_dir / filename
        
        logger.info(f"Saving original data to {filepath}")

        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                # Format for training
                training_item = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that solves mathematical problems step by step.",
                        },
                        {
                            "role": "user",
                            "content": f"Solve the following mathematical problem step by step. Provide a detailed solution and then give the final answer.\n\nProblem: {item[question_column]}",
                        },
                        {
                            "role": "assistant",
                            "content": f"{item[solution_column]}\n\nFinal Answer: {item[answer_column]}",
                        },
                    ]
                }
                f.write(json.dumps(training_item, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(data)} original samples to {filepath}")
        return str(filepath)

    def save_code_switched_data(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str,
        split: str,
        subset: Optional[str] = None,
        n_samples: Optional[int] = None
    ) -> str:
        """
        Save code-switched dataset as JSONL in training format
        
        Args:
            data: Code-switched dataset samples
            dataset_name: Name of the dataset
            split: Dataset split
            subset: Dataset subset
            n_samples: Number of samples
            
        Returns:
            Path to saved file
        """
        filename = generate_filename(
            dataset_name, split, subset, n_samples, "code_switched"
        )
        filepath = self.output_dir / filename
        
        logger.info(f"Saving code-switched data to {filepath}")

        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                # Format for training
                training_item = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that solves mathematical problems step by step.",
                        },
                        {
                            "role": "user",
                            "content": f"Solve the following mathematical problem step by step. Provide a detailed solution and then give the final answer.\n\nProblem: {item['question']}",
                        },
                        {
                            "role": "assistant",
                            "content": f"{item['code_switched_solution']}\n\nFinal Answer: {item['answer']}",
                        },
                    ],
                    "metadata": {
                        "original_solution": item.get("original_solution"),
                        "processing_info": item.get("processing_info"),
                    },
                }
                f.write(json.dumps(training_item, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(data)} code-switched samples to {filepath}")
        return str(filepath)

    def load_jsonl_data(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load data from JSONL file
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            List of loaded samples
        """
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data)} samples from {filepath}")
        return data
