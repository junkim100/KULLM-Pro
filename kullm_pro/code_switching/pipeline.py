"""
Main code switching pipeline
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .openai_client import OpenAIClient, OpenAIConfig
from .dataset_processor import DatasetProcessor
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory

logger = get_logger("code_switching.pipeline")


class CodeSwitchingPipeline:
    """Main pipeline for code-switching data generation"""

    def __init__(
        self,
        openai_config: OpenAIConfig,
        system_prompt_path: str = "system_prompt.txt",
        output_dir: str = "./data"
    ):
        self.openai_config = openai_config
        self.client = OpenAIClient(openai_config)
        self.dataset_processor = DatasetProcessor(output_dir)
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        self.output_dir = Path(output_dir)
        ensure_directory(self.output_dir)

    def _load_system_prompt(self, path: str) -> str:
        """Load system prompt from file"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            logger.info(f"Loaded system prompt from {path} ({len(prompt)} characters)")
            return prompt
        except Exception as e:
            logger.error(f"Failed to load system prompt from {path}: {e}")
            raise

    async def process_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
        n_samples: Optional[int] = None,
        text_column: str = "solution",
        question_column: str = "question",
        solution_column: str = "solution",
        answer_column: str = "answer",
        skip_original: bool = False
    ) -> tuple[str, str]:
        """
        Process a dataset for code-switching
        
        Args:
            dataset_name: Name of the Hugging Face dataset
            split: Dataset split to use
            subset: Dataset subset (if applicable)
            n_samples: Number of samples to process (shortest if None)
            text_column: Column to use for length sorting
            question_column: Column name for questions
            solution_column: Column name for solutions
            answer_column: Column name for answers
            skip_original: Skip saving original data
            
        Returns:
            Tuple of (original_file_path, code_switched_file_path)
        """
        logger.info(f"Starting code-switching pipeline for {dataset_name}")
        logger.info(f"Parameters: split={split}, subset={subset}, n_samples={n_samples}")

        # Step 1: Load and filter dataset
        logger.info("=== Step 1: Loading and filtering dataset ===")
        original_data = self.dataset_processor.load_and_filter_dataset(
            dataset_name=dataset_name,
            split=split,
            subset=subset,
            n_samples=n_samples,
            text_column=text_column
        )

        # Step 2: Save original data
        original_file_path = None
        if not skip_original:
            logger.info("=== Step 2: Saving original data ===")
            original_file_path = self.dataset_processor.save_original_data(
                data=original_data,
                dataset_name=dataset_name,
                split=split,
                subset=subset,
                n_samples=len(original_data),
                question_column=question_column,
                solution_column=solution_column,
                answer_column=answer_column
            )

        # Step 3: Generate code-switched versions
        logger.info("=== Step 3: Generating code-switched versions ===")
        code_switched_results, failed_items = await self._process_code_switching(original_data)

        # Step 4: Save code-switched data
        logger.info("=== Step 4: Saving code-switched data ===")
        code_switched_file_path = self.dataset_processor.save_code_switched_data(
            data=code_switched_results,
            dataset_name=dataset_name,
            split=split,
            subset=subset,
            n_samples=len(code_switched_results)
        )

        # Step 5: Save processing statistics
        await self._save_processing_stats(
            dataset_name, split, subset, n_samples,
            original_data, code_switched_results, failed_items
        )

        logger.info("=== Processing Complete ===")
        logger.info(f"Original data: {original_file_path}")
        logger.info(f"Code-switched data: {code_switched_file_path}")
        logger.info(f"Successfully processed: {len(code_switched_results)}/{len(original_data)}")

        return original_file_path, code_switched_file_path

    async def _process_code_switching(
        self, original_data: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process code-switching using OpenAI API"""
        
        if self.openai_config.use_batch_api:
            return await self._process_with_batch_api(original_data)
        else:
            return await self._process_with_regular_api(original_data)

    async def _process_with_batch_api(
        self, original_data: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process using OpenAI Batch API"""
        logger.info("Using OpenAI Batch API for processing")

        results = []
        failed_items = []

        # Extract solutions
        solutions = [item["solution"] for item in original_data]

        # Create batch file
        batch_file = self.client.create_batch_file(self.system_prompt, solutions)

        try:
            # Submit batch
            batch_id = self.client.submit_batch(batch_file)

            # Wait for completion
            batch_results = self.client.wait_for_batch(batch_id)

            # Process results
            for i, (original_item, api_result) in enumerate(
                zip(original_data, batch_results)
            ):
                if api_result.get("success", False):
                    processed_item = {
                        "question": original_item["question"],
                        "original_solution": original_item["solution"],
                        "code_switched_solution": api_result["response"],
                        "answer": original_item["answer"],
                        "processing_info": {
                            "model": api_result.get("model", self.openai_config.model),
                            "usage": api_result.get("usage"),
                            "timestamp": time.time(),
                            "batch_id": batch_id,
                        },
                    }
                    results.append(processed_item)
                else:
                    failed_item = {
                        "index": i,
                        "original_item": original_item,
                        "error": api_result.get("error", "Unknown error"),
                    }
                    failed_items.append(failed_item)
                    logger.error(f"Failed to process item {i}: {failed_item['error']}")

        finally:
            # Clean up batch file
            try:
                Path(batch_file).unlink()
            except:
                pass

        return results, failed_items

    async def _process_with_regular_api(
        self, original_data: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process using regular API calls"""
        logger.info("Using regular API calls for processing")

        results = []
        failed_items = []

        # Process in smaller batches
        batch_size = 20

        for i in range(0, len(original_data), batch_size):
            batch = original_data[i : i + batch_size]
            batch_solutions = [item["solution"] for item in batch]

            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(original_data) + batch_size - 1)//batch_size}"
            )

            # Process batch
            batch_results = await self.client.process_batch(
                self.system_prompt,
                batch_solutions,
                progress_callback=lambda current, total: logger.info(
                    f"Batch progress: {current}/{total}"
                ),
            )

            # Combine results with original data
            for j, (original_item, api_result) in enumerate(zip(batch, batch_results)):
                if api_result.get("success", False):
                    processed_item = {
                        "question": original_item["question"],
                        "original_solution": original_item["solution"],
                        "code_switched_solution": api_result["response"],
                        "answer": original_item["answer"],
                        "processing_info": {
                            "model": api_result.get("model", self.openai_config.model),
                            "usage": api_result.get("usage"),
                            "timestamp": time.time(),
                        },
                    }
                    results.append(processed_item)
                else:
                    failed_item = {
                        "index": i + j,
                        "original_item": original_item,
                        "error": api_result.get("error", "Unknown error"),
                    }
                    failed_items.append(failed_item)
                    logger.error(
                        f"Failed to process item {i + j}: {failed_item['error']}"
                    )

        return results, failed_items

    async def _save_processing_stats(
        self,
        dataset_name: str,
        split: str,
        subset: Optional[str],
        n_samples: Optional[int],
        original_data: List[Dict[str, Any]],
        code_switched_results: List[Dict[str, Any]],
        failed_items: List[Dict[str, Any]]
    ):
        """Save processing statistics"""
        stats = self.client.get_usage_stats()
        
        stats_data = {
            "timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "name": dataset_name,
                "split": split,
                "subset": subset,
                "requested_samples": n_samples,
                "actual_samples": len(original_data)
            },
            "api_stats": stats,
            "results": {
                "total_input": len(original_data),
                "successful": len(code_switched_results),
                "failed": len(failed_items),
                "success_rate": len(code_switched_results) / len(original_data) if original_data else 0,
            },
        }

        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)

        # Save failed items if any
        if failed_items:
            failed_file = self.output_dir / "failed_items.json"
            with open(failed_file, "w", encoding="utf-8") as f:
                json.dump(failed_items, f, ensure_ascii=False, indent=2)
            logger.warning(f"Saved {len(failed_items)} failed items to {failed_file}")

        logger.info(f"Processing statistics saved to {stats_file}")
        logger.info(f"API usage stats: {stats}")
