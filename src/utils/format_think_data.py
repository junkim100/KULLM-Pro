#!/usr/bin/env python3
"""
Format training data to include proper think tokens for KULLM-Pro training.
Converts LIMO dataset format to think token format for reasoning model training.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_limo_with_think_tokens(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a LIMO dataset sample to include think tokens.
    
    Args:
        sample: Original LIMO sample with question, solution, answer
        
    Returns:
        Formatted sample with think tokens
    """
    
    question = sample.get("question", "").strip()
    solution = sample.get("solution", "").strip()
    answer = sample.get("answer", "").strip()
    
    if not all([question, solution, answer]):
        logger.warning(f"Incomplete sample: missing question, solution, or answer")
        return sample
    
    # Format as: <think>\n\nsolution\n\n</think>\n\nanswer
    formatted_response = f"<think>\n\n{solution}\n\n</think>\n\n{answer}"
    
    # Create chat format
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": formatted_response}
    ]
    
    return {
        "messages": messages,
        "question": question,
        "formatted_response": formatted_response,
        "original_solution": solution,
        "original_answer": answer,
        "think_token_format": True
    }


def validate_think_token_format(text: str) -> bool:
    """
    Validate that think tokens are properly formatted.
    
    Args:
        text: Text to validate
        
    Returns:
        True if properly formatted
    """
    
    think_opens = text.count("<think>")
    think_closes = text.count("</think>")
    
    if think_opens != think_closes:
        return False
    
    if think_opens == 0:
        return False  # Should have at least one think token
    
    # Check basic structure: should have <think>content</think>answer
    if not text.strip().startswith("<think>"):
        return False
        
    if "</think>" not in text:
        return False
    
    return True


def get_dataset_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about the formatted dataset.
    
    Args:
        data: List of formatted samples
        
    Returns:
        Statistics dictionary
    """
    
    stats = {
        "total_samples": len(data),
        "valid_think_format": 0,
        "invalid_think_format": 0,
        "average_solution_length": 0,
        "average_answer_length": 0,
        "average_total_length": 0
    }
    
    total_solution_length = 0
    total_answer_length = 0
    total_response_length = 0
    
    for sample in data:
        formatted_response = sample.get("formatted_response", "")
        
        if validate_think_token_format(formatted_response):
            stats["valid_think_format"] += 1
        else:
            stats["invalid_think_format"] += 1
        
        # Calculate lengths
        solution_length = len(sample.get("original_solution", ""))
        answer_length = len(sample.get("original_answer", ""))
        response_length = len(formatted_response)
        
        total_solution_length += solution_length
        total_answer_length += answer_length
        total_response_length += response_length
    
    if len(data) > 0:
        stats["average_solution_length"] = total_solution_length // len(data)
        stats["average_answer_length"] = total_answer_length // len(data)
        stats["average_total_length"] = total_response_length // len(data)
    
    return stats


def format_dataset(
    input_file: str,
    output_file: str,
    format_type: str = "think_tokens"
) -> None:
    """
    Format a dataset file to include think tokens.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        format_type: Type of formatting to apply
    """
    
    logger.info(f"Formatting dataset: {input_file} -> {output_file}")
    logger.info(f"Format type: {format_type}")
    
    # Load input data
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                data.append(sample)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} samples from {input_file}")
    
    # Format data
    formatted_data = []
    
    for sample in tqdm(data, desc="Formatting samples"):
        if format_type == "think_tokens":
            formatted_sample = format_limo_with_think_tokens(sample)
            formatted_data.append(formatted_sample)
        else:
            logger.warning(f"Unknown format type: {format_type}")
            formatted_data.append(sample)
    
    # Get statistics
    stats = get_dataset_statistics(formatted_data)
    
    logger.info("Dataset Statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Valid think format: {stats['valid_think_format']}")
    logger.info(f"  Invalid think format: {stats['invalid_think_format']}")
    logger.info(f"  Average solution length: {stats['average_solution_length']} chars")
    logger.info(f"  Average answer length: {stats['average_answer_length']} chars")
    logger.info(f"  Average total length: {stats['average_total_length']} chars")
    
    # Save formatted data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in formatted_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(formatted_data)} formatted samples to {output_file}")
    
    # Save statistics
    stats_file = output_path.with_suffix('.stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved statistics to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Format training data with think tokens")
    parser.add_argument("--input_file", required=True, help="Input JSONL file")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    parser.add_argument(
        "--format", 
        default="think_tokens", 
        choices=["think_tokens"],
        help="Format type to apply"
    )
    
    args = parser.parse_args()
    
    try:
        format_dataset(args.input_file, args.output_file, args.format)
        logger.info("✅ Dataset formatting completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error formatting dataset: {e}")
        raise


if __name__ == "__main__":
    main()
