#!/usr/bin/env python3
"""
KULLM Pro Code Switching Module - Streaming Implementation

This module provides concept-driven code switching functionality using OpenAI's streaming API
to convert English reasoning traces to Korean-English code-switched versions with sophisticated
linguistic algorithms for maximum efficiency.

Features:
- Process any Hugging Face dataset with flexible parameters
- Real-time streaming API calls with progress tracking
- Concept-driven language selection with hybrid expressions
- Sophisticated error handling and retry logic
- Generate descriptive JSONL filenames
- Output original and code-switched versions
- Python Fire CLI interface

Usage:
    python src/code_switch_new.py run --dataset_name="GAIR/LIMO" --split="train" --n_samples=10

Author: KULLM Pro Team
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import fire
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_processing import (
    load_hf_dataset,
    filter_shortest_samples,
    save_jsonl,
    create_output_filename,
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StreamingCodeSwitchingPipeline:
    """
    Pipeline for concept-driven code switching using streaming API calls.
    Transforms English reasoning traces into Korean-English code-switched versions.
    """

    def __init__(
        self,
        output_dir: str = "./data",
        api_key: Optional[str] = None,
        system_prompt_path: str = "code_switch.txt",
        model: str = "gpt-4.1-2025-04-14",
        max_completion_tokens: int = 4000,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the streaming code switching pipeline.

        Args:
            output_dir: Directory to save output files
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
            system_prompt_path: Path to the system prompt file
            model: OpenAI model to use (gpt-4o-mini supports system messages)
            max_completion_tokens: Maximum tokens for completion
            temperature: Temperature for generation
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Setup OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

        self.client = AsyncOpenAI(api_key=api_key)

        # Load system prompt from file
        self.system_prompt = self.load_system_prompt(system_prompt_path)

    def load_system_prompt(self, system_prompt_path: str) -> str:
        """
        Load system prompt from file.

        Args:
            system_prompt_path: Path to the system prompt file

        Returns:
            System prompt content
        """
        try:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            logger.info(f"Loaded system prompt from {system_prompt_path}")
            return prompt
        except FileNotFoundError:
            logger.error(f"System prompt file not found: {system_prompt_path}")
            # Fallback to default prompt
            return """You are a sophisticated linguistic agent that performs concept-based code-switching between English and Korean.
Transform the given English reasoning trace into a conceptually efficient, code-switched version that blends English and Korean.
Use Korean as the matrix language (SOV structure) and embed English concepts where they are more efficient.
Apply Korean particles to all nouns and use Korean verb conjugations.
Focus on conceptual efficiency rather than word-for-word translation.
Preserve all mathematical notation, reasoning markers, and final answer formatting exactly."""

    async def transform_solution(
        self, solution_text: str, sample_id: int
    ) -> Optional[str]:
        """
        Transform a single solution using the OpenAI API with retry logic.

        Args:
            solution_text: The English solution text to transform
            sample_id: ID of the sample for logging

        Returns:
            Transformed code-switched solution or None if failed
        """
        user_content = f"""CRITICAL: Transform each sentence individually with exact 1:1 correspondence.

MANDATORY SENTENCE-BY-SENTENCE PROTOCOL:
1. Split the input into individual sentences
2. Transform each sentence separately using concept-driven efficiency analysis
3. Output exactly the same number of sentences as input
4. Maintain identical reasoning flow structure
5. NEVER combine, split, or reorder sentences

EFFICIENCY RULES PER SENTENCE:
- Each transformed sentence must be ≤ original sentence length
- Use Korean for long terms: "therefore"→"따라서", "substitute"→"대입", "calculate"→"계산"
- Keep English for short terms: "so", "if", "but"
- Create efficient hybrids: "solve하면", "recall해보자", "check해보자"
- Preserve all mathematical notation exactly

FORBIDDEN:
- Combining sentences: "Sentence A. Sentence B." → "Combined sentence A+B"
- Splitting sentences: "Long sentence" → "Short sentence 1. Short sentence 2."
- Reordering: Changing the sequence of reasoning steps
- Summarizing: Condensing multiple ideas into fewer sentences

REQUIRED OUTPUT:
Transform each sentence individually and output in exact same order.

Input reasoning trace:
{solution_text}"""

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=self.max_completion_tokens,
                    temperature=self.temperature,
                )

                if response.choices and response.choices[0].message.content:
                    transformed_text = response.choices[0].message.content.strip()
                    if transformed_text:
                        # Length efficiency check
                        original_length = len(solution_text)
                        transformed_length = len(transformed_text)

                        if transformed_length <= original_length:
                            efficiency_ratio = (
                                (original_length - transformed_length)
                                / original_length
                                * 100
                            )
                            logger.debug(
                                f"Sample {sample_id}: {efficiency_ratio:.1f}% more efficient ({original_length}→{transformed_length} chars)"
                            )
                            return transformed_text
                        else:
                            # If transformed version is longer, try a more aggressive efficiency prompt
                            logger.warning(
                                f"Sample {sample_id}: Transformed version longer ({original_length}→{transformed_length} chars), retrying with efficiency focus"
                            )
                            if attempt == 0:  # Only retry once with efficiency focus
                                # Count sentences in original for verification
                                import re

                                original_sentences = re.split(
                                    r"[.!?]+\s+", solution_text.strip()
                                )
                                original_sentences = [
                                    s.strip() for s in original_sentences if s.strip()
                                ]

                                user_content_efficient = f"""URGENT: Transform with EXACT sentence correspondence and maximum efficiency.

CRITICAL REQUIREMENTS:
- Original has {len(original_sentences)} sentences - output EXACTLY {len(original_sentences)} sentences
- Each output sentence must be ≤ corresponding input sentence length
- Transform sentence-by-sentence, never combine or split
- Total length: {original_length} chars → must be ≤ {original_length} chars

SENTENCE-BY-SENTENCE TRANSFORMATION:
Process each sentence individually:
1. "Sentence 1" → "Korean-English efficient version 1"
2. "Sentence 2" → "Korean-English efficient version 2"
...and so on

EFFICIENCY RULES:
- Korean for long terms: "therefore"→"따라서", "substitute"→"대입", "calculate"→"계산"
- Keep English for short terms: "so", "if", "but"
- Efficient hybrids: "solve하면", "recall해보자", "check해보자"
- Preserve mathematical notation exactly

Input ({len(original_sentences)} sentences):
{solution_text}"""

                                retry_response = (
                                    await self.client.chat.completions.create(
                                        model=self.model,
                                        messages=[
                                            {
                                                "role": "system",
                                                "content": self.system_prompt,
                                            },
                                            {
                                                "role": "user",
                                                "content": user_content_efficient,
                                            },
                                        ],
                                        max_tokens=self.max_completion_tokens,
                                        temperature=self.temperature,
                                    )
                                )

                                if (
                                    retry_response.choices
                                    and retry_response.choices[0].message.content
                                ):
                                    retry_text = retry_response.choices[
                                        0
                                    ].message.content.strip()
                                    retry_length = len(retry_text)

                                    if retry_length <= original_length:
                                        efficiency_ratio = (
                                            (original_length - retry_length)
                                            / original_length
                                            * 100
                                        )
                                        logger.info(
                                            f"Sample {sample_id}: Efficiency retry successful - {efficiency_ratio:.1f}% more efficient"
                                        )
                                        return retry_text
                                    else:
                                        logger.warning(
                                            f"Sample {sample_id}: Even retry is too long, keeping original"
                                        )
                                        return solution_text
                            else:
                                logger.warning(
                                    f"Sample {sample_id}: Keeping original due to length constraint"
                                )
                                return solution_text
                    else:
                        logger.warning(f"Empty response for sample {sample_id}")
                else:
                    logger.warning(f"No content in response for sample {sample_id}")

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed for sample {sample_id}: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(
                        self.retry_delay * (2**attempt)
                    )  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for sample {sample_id}")

        return None

    async def process_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
        n_samples: Optional[int] = None,
        max_concurrent: int = 5,
    ) -> Tuple[str, str]:
        """
        Process a dataset for code switching using streaming API calls.

        Args:
            dataset_name: Name of the Hugging Face dataset
            split: Dataset split to process
            subset: Dataset subset/configuration
            n_samples: Number of samples to process (shortest by character count)
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            Tuple of (original_file_path, code_switched_file_path)
        """
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Split: {split}, Subset: {subset}, Samples: {n_samples}")
        logger.info("Code switching will transform the 'solution' column")

        # Load dataset
        dataset = load_hf_dataset(dataset_name, split, subset)
        data = list(dataset)

        # Validate that data has required columns
        if data:
            sample = data[0]
            required_columns = ["question", "solution", "answer"]
            missing_columns = [col for col in required_columns if col not in sample]
            if missing_columns:
                logger.warning(f"Missing columns in dataset: {missing_columns}")
                logger.info(f"Available columns: {list(sample.keys())}")

        # Filter to shortest samples if n_samples specified
        if n_samples and n_samples < len(data):
            data = filter_shortest_samples(data, n_samples, text_columns=["solution"])

        # Generate filenames
        base_filename = create_output_filename(dataset_name, split, subset, len(data))
        original_filename = f"original_{base_filename}"
        code_switched_filename = f"code_switched_{base_filename}"

        original_path = self.output_dir / original_filename
        code_switched_path = self.output_dir / code_switched_filename

        # Save original data
        save_jsonl(data, original_path)

        # Process solutions with streaming API calls
        logger.info(
            f"Starting code switching transformation for {len(data)} samples..."
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_sample(i: int, sample: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                solution_text = sample.get("solution", "")
                if solution_text and solution_text.strip():
                    transformed_solution = await self.transform_solution(
                        solution_text.strip(), i
                    )
                    if transformed_solution:
                        return {
                            "question": sample.get("question", ""),
                            "solution": transformed_solution,
                            "answer": sample.get("answer", ""),
                        }
                    else:
                        logger.warning(
                            f"Failed to transform sample {i}, keeping original"
                        )
                        return sample
                else:
                    logger.warning(f"Sample {i} has empty solution, keeping original")
                    return sample

        # Process all samples concurrently with progress bar
        tasks = [process_sample(i, sample) for i, sample in enumerate(data)]
        code_switched_data = []

        async for result in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Code switching"
        ):
            code_switched_data.append(await result)

        # Save code-switched data
        save_jsonl(code_switched_data, code_switched_path)

        logger.info(f"Code switching completed!")
        logger.info(f"Original file: {original_path}")
        logger.info(f"Code-switched file: {code_switched_path}")

        return str(original_path), str(code_switched_path)


class CodeSwitchCLI:
    """
    Command-line interface for streaming code switching using Python Fire.
    """

    def __init__(self):
        """Initialize the CLI."""
        self.logger = logger

    def run(
        self,
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
        n_samples: Optional[int] = None,
        output_dir: str = "./data",
        system_prompt_path: str = "code_switch.txt",
        model: str = "gpt-4.1-2025-04-14",
        max_completion_tokens: int = 12000,
        temperature: float = 0.1,
        max_concurrent: int = 5,
        max_retries: int = 3,
    ):
        """
        Run concept-driven code switching on a dataset using streaming API calls.

        Uses sophisticated linguistic theories to create Korean-English code-switched text
        that leverages the unique strengths of each language for maximum conceptual efficiency.

        Args:
            dataset_name: Name of the Hugging Face dataset (e.g., "GAIR/LIMO")
            split: Dataset split to process (default: "train")
            subset: Dataset subset/configuration (default: None)
            n_samples: Number of shortest samples to process (default: None for all)
            output_dir: Output directory for files (default: "./data")
            system_prompt_path: Path to system prompt file (default: "code_switch.txt")
            model: OpenAI model to use (default: "gpt-4o-mini")
            max_completion_tokens: Maximum tokens for completion (default: 4000)
            temperature: Temperature for generation (default: 0.1)
            max_concurrent: Maximum concurrent API calls (default: 5)
            max_retries: Maximum retries for failed requests (default: 3)

        Example:
            python src/code_switch_new.py run --dataset_name="GAIR/LIMO" --split="train" --n_samples=10
        """
        # Create pipeline with custom parameters
        pipeline = StreamingCodeSwitchingPipeline(
            output_dir=output_dir,
            system_prompt_path=system_prompt_path,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )

        # Run the processing
        try:
            original_file, code_switched_file = asyncio.run(
                pipeline.process_dataset(
                    dataset_name=dataset_name,
                    split=split,
                    subset=subset,
                    n_samples=n_samples,
                    max_concurrent=max_concurrent,
                )
            )

            self.logger.info("Code switching completed successfully!")
            self.logger.info(f"Original file: {original_file}")
            self.logger.info(f"Code-switched file: {code_switched_file}")
            self.logger.info(
                "The 'solution' column was transformed with concept-driven code switching"
            )

            return {
                "original_file": original_file,
                "code_switched_file": code_switched_file,
                "status": "success",
                "model_used": model,
                "transformed_columns": ["solution"],
            }

        except Exception as e:
            self.logger.error(f"Code switching failed: {e}")
            raise


def main():
    """Main entry point for the CLI."""
    fire.Fire(CodeSwitchCLI)


if __name__ == "__main__":
    main()
