#!/usr/bin/env python3
"""
Clean tokenizer by removing unnecessary special tokens and keeping only essential ones.
This script removes tool_call, box, vision, and other unnecessary tokens while preserving
im_start, im_end, and think tokens.
"""

import json
import os
import logging
from pathlib import Path
from transformers import AutoTokenizer
from typing import Dict, Set, Optional
import argparse

logger = logging.getLogger(__name__)


def clean_added_tokens(tokenizer_path: str, output_path: str = None):
    """
    Clean the added_tokens.json file to keep only essential tokens.

    Args:
        tokenizer_path: Path to the tokenizer directory
        output_path: Optional output path (defaults to same directory)
    """

    # Essential tokens we want to keep
    essential_tokens = {
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<|endoftext|>",  # Keep this as it's often needed for EOS
    }

    added_tokens_file = Path(tokenizer_path) / "added_tokens.json"

    if not added_tokens_file.exists():
        print(f"❌ added_tokens.json not found in {tokenizer_path}")
        return

    # Load current added tokens
    with open(added_tokens_file, "r") as f:
        current_tokens = json.load(f)

    print(f"🔍 Current added tokens: {len(current_tokens)}")
    for token in current_tokens.keys():
        print(f"  - {token}")

    # Filter to keep only essential tokens
    cleaned_tokens = {
        token: token_id
        for token, token_id in current_tokens.items()
        if token in essential_tokens
    }

    print(f"\n✅ Keeping {len(cleaned_tokens)} essential tokens:")
    for token in cleaned_tokens.keys():
        print(f"  - {token}")

    print(
        f"\n🗑️  Removing {len(current_tokens) - len(cleaned_tokens)} unnecessary tokens:"
    )
    removed_tokens = set(current_tokens.keys()) - set(cleaned_tokens.keys())
    for token in removed_tokens:
        print(f"  - {token}")

    # Determine output path
    if output_path is None:
        output_path = tokenizer_path

    output_file = Path(output_path) / "added_tokens.json"

    # Save cleaned tokens
    with open(output_file, "w") as f:
        json.dump(cleaned_tokens, f, indent=2)

    print(f"\n💾 Saved cleaned added_tokens.json to {output_file}")

    return cleaned_tokens


def clean_tokenizer_config(tokenizer_path: str, output_path: str = None):
    """
    Clean tokenizer_config.json to remove references to unnecessary tokens.
    """

    config_file = Path(tokenizer_path) / "tokenizer_config.json"

    if not config_file.exists():
        print(f"⚠️  tokenizer_config.json not found in {tokenizer_path}")
        return

    with open(config_file, "r") as f:
        config = json.load(f)

    # Remove unnecessary special tokens from config
    if "added_tokens_decoder" in config:
        essential_tokens = {
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<|endoftext|>",
        }

        original_count = len(config["added_tokens_decoder"])
        config["added_tokens_decoder"] = {
            token_id: token_info
            for token_id, token_info in config["added_tokens_decoder"].items()
            if token_info.get("content", "") in essential_tokens
        }

        print(
            f"🧹 Cleaned added_tokens_decoder: {original_count} → {len(config['added_tokens_decoder'])}"
        )

    # Determine output path
    if output_path is None:
        output_path = tokenizer_path

    output_file = Path(output_path) / "tokenizer_config.json"

    # Save cleaned config
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"💾 Saved cleaned tokenizer_config.json to {output_file}")


def create_clean_tokenizer(base_model_path: str, output_path: str):
    """
    Create a clean tokenizer with only essential special tokens.

    Args:
        base_model_path: Path to the base model/tokenizer
        output_path: Path where to save the cleaned tokenizer
    """

    print(f"🚀 Creating clean tokenizer from {base_model_path}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Get current special tokens
    print(f"📊 Current special tokens: {len(tokenizer.added_tokens_decoder)}")

    # Essential tokens to keep
    essential_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]

    # Create a new tokenizer with only essential tokens
    # Note: We'll manually clean the files after saving
    tokenizer.save_pretrained(output_path)

    # Clean the saved files
    clean_added_tokens(output_path)
    clean_tokenizer_config(output_path)

    print(f"✅ Clean tokenizer saved to {output_path}")


def clean_tokenizer_for_training(
    tokenizer: AutoTokenizer, essential_tokens: Optional[Set[str]] = None
) -> AutoTokenizer:
    """
    Clean tokenizer by removing unnecessary special tokens for training.

    Args:
        tokenizer: The tokenizer to clean
        essential_tokens: Set of tokens to keep (defaults to think tokens + chat tokens)

    Returns:
        Cleaned tokenizer
    """

    if essential_tokens is None:
        essential_tokens = {
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<|endoftext|>",
        }

    # Get current added tokens
    current_tokens = (
        set(tokenizer.added_tokens_decoder.values())
        if hasattr(tokenizer, "added_tokens_decoder")
        else set()
    )

    # Find tokens to remove
    tokens_to_remove = []
    for token_id, token_content in tokenizer.added_tokens_decoder.items():
        if hasattr(token_content, "content"):
            token_str = token_content.content
        else:
            token_str = str(token_content)

        if token_str not in essential_tokens:
            tokens_to_remove.append(token_str)

    if tokens_to_remove:
        logger.info(f"Removing {len(tokens_to_remove)} unnecessary special tokens:")
        for token in tokens_to_remove:
            logger.info(f"  - {token}")

        # Note: We can't actually remove tokens from the tokenizer easily,
        # but we can log what would be removed and ensure training data doesn't use them
        logger.warning(
            "Note: Tokens are logged for removal but tokenizer modification requires manual intervention"
        )

    logger.info(f"Keeping {len(essential_tokens)} essential tokens:")
    for token in essential_tokens:
        logger.info(f"  + {token}")

    return tokenizer


def validate_think_token_format(text: str) -> bool:
    """
    Validate that think tokens are properly formatted in text.

    Args:
        text: Text to validate

    Returns:
        True if think tokens are properly formatted
    """
    import re

    # Count opening and closing think tokens
    think_opens = len(re.findall(r"<think>", text))
    think_closes = len(re.findall(r"</think>", text))

    if think_opens != think_closes:
        logger.warning(
            f"Unbalanced think tokens: {think_opens} opens, {think_closes} closes"
        )
        return False

    return True


def get_think_token_stats(dataset: list) -> Dict[str, int]:
    """
    Get statistics about think token usage in dataset.

    Args:
        dataset: List of data samples

    Returns:
        Statistics dictionary
    """
    stats = {
        "total_samples": len(dataset),
        "samples_with_think_tokens": 0,
        "samples_with_balanced_tokens": 0,
        "total_think_opens": 0,
        "total_think_closes": 0,
        "average_think_length": 0,
    }

    total_think_length = 0
    think_count = 0

    for sample in dataset:
        # Check different possible text fields
        text_fields = ["formatted_response", "text", "solution", "content"]
        text = ""

        for field in text_fields:
            if field in sample:
                text = str(sample[field])
                break

        if not text:
            continue

        # Count think tokens
        think_opens = text.count("<think>")
        think_closes = text.count("</think>")

        stats["total_think_opens"] += think_opens
        stats["total_think_closes"] += think_closes

        if think_opens > 0 or think_closes > 0:
            stats["samples_with_think_tokens"] += 1

        if think_opens == think_closes and think_opens > 0:
            stats["samples_with_balanced_tokens"] += 1

        # Calculate think content length
        import re

        think_matches = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
        for match in think_matches:
            total_think_length += len(match.strip())
            think_count += 1

    if think_count > 0:
        stats["average_think_length"] = total_think_length // think_count

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean tokenizer special tokens")
    parser.add_argument(
        "--tokenizer_path", required=True, help="Path to tokenizer directory"
    )
    parser.add_argument("--output_path", help="Output path (optional)")
    parser.add_argument(
        "--create_new", action="store_true", help="Create new clean tokenizer"
    )

    args = parser.parse_args()

    if args.create_new:
        if not args.output_path:
            args.output_path = args.tokenizer_path + "_clean"
        create_clean_tokenizer(args.tokenizer_path, args.output_path)
    else:
        clean_added_tokens(args.tokenizer_path, args.output_path)
        clean_tokenizer_config(args.tokenizer_path, args.output_path)


if __name__ == "__main__":
    main()
