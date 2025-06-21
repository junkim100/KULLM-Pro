"""
OpenAI API client for code-switching data processing
"""

import os
import asyncio
import logging
import time
import tempfile
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

import openai
from openai import AsyncOpenAI, OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import aiohttp

from ..utils.logging import get_logger

logger = get_logger("code_switching.openai_client")


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API"""

    api_key: Optional[str] = None
    model: str = "o4-mini-2025-04-16"
    max_tokens: int = 100000
    temperature: float = 1.0
    use_batch_api: bool = True
    batch_size: int = 50
    max_concurrent_requests: int = 5
    requests_per_minute: int = 500
    max_retries: int = 5
    retry_delay: float = 1.0
    timeout: int = 600
    batch_timeout_hours: int = 48

    # Long content handling
    max_input_length: int = 200000
    chunk_size: int = 150000
    overlap_size: int = 5000
    enable_chunking: bool = True

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )


class OpenAIClient:
    """Async OpenAI client with rate limiting and error handling"""

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = AsyncOpenAI(api_key=config.api_key)
        self.sync_client = OpenAI(api_key=config.api_key)

        # Set up rate limiting
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # Track usage
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens_used = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError, aiohttp.ClientError)
        ),
    )
    async def _make_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make a single API request with retry logic"""

        async with self.semaphore:
            try:
                self.total_requests += 1

                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_completion_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    timeout=self.config.timeout,
                )

                self.successful_requests += 1

                # Track token usage
                if hasattr(response, "usage") and response.usage:
                    self.total_tokens_used += response.usage.total_tokens

                return {
                    "success": True,
                    "response": response.choices[0].message.content,
                    "usage": (response.usage.model_dump() if response.usage else None),
                    "model": response.model,
                }

            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit: {e}")
                self.failed_requests += 1
                raise

            except openai.APITimeoutError as e:
                logger.warning(f"API timeout: {e}")
                self.failed_requests += 1
                raise

            except Exception as e:
                logger.error(f"API request failed: {e}")
                self.failed_requests += 1
                return {"success": False, "error": str(e), "response": None}

    async def process_single_item(
        self, system_prompt: str, user_content: str
    ) -> Dict[str, Any]:
        """Process a single item through the API with long content support"""

        # Check content length and use chunking if needed
        total_length = len(system_prompt) + len(user_content)

        if (self.config.enable_chunking and
            total_length > self.config.max_input_length and
            len(user_content) > self.config.chunk_size):
            logger.info(f"Content is long ({total_length} chars), using chunked processing")
            return await self.process_long_content(system_prompt, user_content)

        # Standard processing for normal-length content
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input: {user_content}"},
        ]

        result = await self._make_request(messages)
        return result

    async def process_batch(
        self,
        system_prompt: str,
        items: List[str],
        progress_callback: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """Process a batch of items"""

        logger.info(f"Processing batch of {len(items)} items")

        # Create tasks for all items
        tasks = []
        for i, item in enumerate(items):
            task = self.process_single_item(system_prompt, item)
            tasks.append(task)

        # Process with progress tracking
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(items))

            # Log progress every 10 items
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(items)} items")

        return results

    def create_batch_file(self, system_prompt: str, items: List[str]) -> str:
        """Create a batch file for OpenAI Batch API with long content support"""
        batch_requests = []
        request_id = 0

        for i, item in enumerate(items):
            # Check if item is too long and needs chunking
            total_length = len(system_prompt) + len(item)

            if (self.config.enable_chunking and
                total_length > self.config.max_input_length and
                len(item) > self.config.chunk_size):

                # Chunk the item
                chunks = self._chunk_long_content(item)
                logger.info(f"Item {i} is long, creating {len(chunks)} batch requests")

                for j, chunk in enumerate(chunks):
                    chunk_system_prompt = f"{system_prompt}\n\n[CHUNK {j+1}/{len(chunks)} for item {i}] Process this part of the content:"

                    messages = [
                        {"role": "system", "content": chunk_system_prompt},
                        {"role": "user", "content": f"Input: {chunk}"},
                    ]

                    request = {
                        "custom_id": f"request-{request_id}-item-{i}-chunk-{j}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.config.model,
                            "messages": messages,
                            "max_completion_tokens": self.config.max_tokens,
                            "temperature": self.config.temperature,
                        },
                    }
                    batch_requests.append(request)
                    request_id += 1
            else:
                # Standard processing for normal-length items
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Input: {item}"},
                ]

                request = {
                    "custom_id": f"request-{request_id}-item-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.config.model,
                        "messages": messages,
                        "max_completion_tokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                    },
                }
                batch_requests.append(request)
                request_id += 1

        logger.info(f"Created {len(batch_requests)} batch requests for {len(items)} items")

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for request in batch_requests:
                f.write(json.dumps(request) + "\n")
            return f.name

    def submit_batch(self, batch_file_path: str) -> str:
        """Submit batch file to OpenAI Batch API"""
        logger.info(f"Submitting batch file: {batch_file_path}")

        with open(batch_file_path, "rb") as f:
            batch_input_file = self.sync_client.files.create(file=f, purpose="batch")

        batch = self.sync_client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        logger.info(f"Batch submitted with ID: {batch.id}")
        return batch.id

    def wait_for_batch(self, batch_id: str, check_interval: int = 60) -> List[Dict[str, Any]]:
        """Wait for batch to complete and return results"""
        logger.info(f"Waiting for batch {batch_id} to complete...")

        while True:
            batch = self.sync_client.batches.retrieve(batch_id)
            logger.info(f"Batch status: {batch.status}")

            if batch.status == "completed":
                logger.info("Batch completed successfully!")
                return self.process_batch_results(batch.output_file_id)
            elif batch.status == "failed":
                logger.error("Batch failed!")
                raise Exception(f"Batch {batch_id} failed")
            elif batch.status in ["cancelled", "expired"]:
                logger.error(f"Batch {batch.status}")
                raise Exception(f"Batch {batch_id} was {batch.status}")

            time.sleep(check_interval)

    def process_batch_results(self, output_file_id: str) -> List[Dict[str, Any]]:
        """Process batch results from output file"""
        logger.info(f"Processing batch results from file: {output_file_id}")

        # Download results
        file_response = self.sync_client.files.content(output_file_id)
        results = []

        for line in file_response.text.strip().split("\n"):
            if line:
                result = json.loads(line)
                if result.get("response"):
                    response_data = result["response"]["body"]
                    if response_data.get("choices"):
                        content = response_data["choices"][0]["message"]["content"]
                        usage = response_data.get("usage", {})

                        # Track usage
                        if usage.get("total_tokens"):
                            self.total_tokens_used += usage["total_tokens"]

                        results.append(
                            {
                                "success": True,
                                "response": content,
                                "usage": usage,
                                "model": response_data.get("model"),
                                "custom_id": result.get("custom_id"),
                            }
                        )
                    else:
                        results.append(
                            {
                                "success": False,
                                "error": "No choices in response",
                                "custom_id": result.get("custom_id"),
                            }
                        )
                else:
                    error_msg = result.get("error", {}).get("message", "Unknown error")
                    results.append(
                        {
                            "success": False,
                            "error": error_msg,
                            "custom_id": result.get("custom_id"),
                        }
                    )

        # Group and combine chunked results
        combined_results = self._combine_chunked_batch_results(results)

        self.successful_requests += len([r for r in combined_results if r["success"]])
        self.failed_requests += len([r for r in combined_results if not r["success"]])
        self.total_requests += len(combined_results)

        return combined_results

    def _combine_chunked_batch_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine chunked batch results back into single items"""
        # Group results by item
        item_groups = {}

        for result in results:
            custom_id = result.get("custom_id", "")

            # Parse custom_id to extract item number
            if "-item-" in custom_id:
                parts = custom_id.split("-item-")
                if len(parts) > 1:
                    item_part = parts[1]
                    if "-chunk-" in item_part:
                        item_num = int(item_part.split("-chunk-")[0])
                        chunk_num = int(item_part.split("-chunk-")[1])
                    else:
                        item_num = int(item_part)
                        chunk_num = 0
                else:
                    # Fallback for old format
                    item_num = int(custom_id.split("-")[1])
                    chunk_num = 0
            else:
                # Fallback for old format
                item_num = int(custom_id.split("-")[1])
                chunk_num = 0

            if item_num not in item_groups:
                item_groups[item_num] = []

            item_groups[item_num].append((chunk_num, result))

        # Combine chunks for each item
        combined_results = []
        for item_num in sorted(item_groups.keys()):
            chunks = sorted(item_groups[item_num], key=lambda x: x[0])

            if len(chunks) == 1:
                # Single chunk (normal item)
                combined_results.append(chunks[0][1])
            else:
                # Multiple chunks - combine them
                successful_chunks = [chunk[1] for chunk in chunks if chunk[1].get("success", False)]

                if len(successful_chunks) == len(chunks):
                    # All chunks successful - combine responses
                    combined_response = "\n\n".join([chunk["response"] for chunk in successful_chunks])
                    combined_usage = {
                        "total_tokens": sum(chunk.get("usage", {}).get("total_tokens", 0) for chunk in successful_chunks),
                        "prompt_tokens": sum(chunk.get("usage", {}).get("prompt_tokens", 0) for chunk in successful_chunks),
                        "completion_tokens": sum(chunk.get("usage", {}).get("completion_tokens", 0) for chunk in successful_chunks),
                    }

                    combined_results.append({
                        "success": True,
                        "response": combined_response,
                        "usage": combined_usage,
                        "model": successful_chunks[0].get("model"),
                        "custom_id": f"combined-item-{item_num}",
                        "chunks_processed": len(chunks),
                    })
                else:
                    # Some chunks failed
                    failed_chunks = [i for i, (_, chunk) in enumerate(chunks) if not chunk.get("success", False)]
                    combined_results.append({
                        "success": False,
                        "error": f"Failed chunks for item {item_num}: {failed_chunks}",
                        "custom_id": f"combined-item-{item_num}",
                        "chunk_results": [chunk[1] for chunk in chunks],
                    })

        return combined_results

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "total_tokens_used": self.total_tokens_used,
            "estimated_cost": self.estimate_cost(),
        }

    def estimate_cost(self) -> float:
        """Estimate cost based on token usage"""
        # o4-mini batch pricing: $0.55 per 1M input tokens, $2.20 per 1M output tokens
        # For simplicity, we'll estimate average cost (assuming ~20% input, 80% output)
        estimated_cost_per_1k_tokens = (
            0.00287  # Rough average: (0.55*0.2 + 2.20*0.8)/1000
        )
        return (self.total_tokens_used / 1000) * estimated_cost_per_1k_tokens

    def _chunk_long_content(self, content: str) -> List[str]:
        """
        Split extremely long content into manageable chunks with overlap

        Args:
            content: The long content to chunk

        Returns:
            List of content chunks with overlap for context preservation
        """
        if len(content) <= self.config.max_input_length:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            # Calculate end position
            end = start + self.config.chunk_size

            # If this is not the last chunk, try to find a good break point
            if end < len(content):
                # Look for sentence boundaries within the last 1000 characters
                search_start = max(end - 1000, start)
                sentence_breaks = [
                    content.rfind('.', search_start, end),
                    content.rfind('!', search_start, end),
                    content.rfind('?', search_start, end),
                    content.rfind('\n\n', search_start, end),
                ]

                # Use the latest sentence break if found
                best_break = max([b for b in sentence_breaks if b > search_start], default=end)
                if best_break > search_start:
                    end = best_break + 1

            # Extract chunk
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            if end >= len(content):
                break
            start = end - self.config.overlap_size

        logger.info(f"Split content into {len(chunks)} chunks (original: {len(content)} chars)")
        return chunks

    async def process_long_content(
        self, system_prompt: str, content: str
    ) -> Dict[str, Any]:
        """
        Process extremely long content by chunking if necessary

        Args:
            system_prompt: The system prompt
            content: The content to process (potentially very long)

        Returns:
            Combined result from all chunks
        """
        # Check if chunking is needed
        total_length = len(system_prompt) + len(content)

        if not self.config.enable_chunking or total_length <= self.config.max_input_length:
            # Process normally without chunking
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Input: {content}"},
            ]
            return await self._make_request(messages)

        # Chunk the content
        chunks = self._chunk_long_content(content)
        logger.info(f"Processing {len(chunks)} chunks for long content")

        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            # Add chunk context to system prompt
            chunk_system_prompt = f"{system_prompt}\n\n[CHUNK {i+1}/{len(chunks)}] Process this part of the content:"

            # Process chunk directly to avoid recursion
            messages = [
                {"role": "system", "content": chunk_system_prompt},
                {"role": "user", "content": f"Input: {chunk}"},
            ]
            result = await self._make_request(messages)
            chunk_results.append(result)

        # Combine results
        if all(r.get("success", False) for r in chunk_results):
            combined_response = "\n\n".join([r["response"] for r in chunk_results])
            combined_usage = {
                "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in chunk_results),
                "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in chunk_results),
                "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in chunk_results),
            }

            return {
                "success": True,
                "response": combined_response,
                "usage": combined_usage,
                "model": chunk_results[0].get("model"),
                "chunks_processed": len(chunks),
            }
        else:
            # Return error if any chunk failed
            failed_chunks = [i for i, r in enumerate(chunk_results) if not r.get("success", False)]
            return {
                "success": False,
                "error": f"Failed to process chunks: {failed_chunks}",
                "chunk_results": chunk_results,
            }
