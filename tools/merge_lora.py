#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for evaluation.

This script creates a merged model that can be used with lm_eval
without vocabulary size mismatches.
"""

import sys
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def merge_lora_adapter(base_model_path, lora_path, output_path):
    """
    Merge LoRA adapter with base model and save as a complete model.

    Args:
        base_model_path: Path to base model
        lora_path: Path to LoRA adapter
        output_path: Path to save merged model
    """
    print(f"🔄 Merging LoRA adapter for evaluation")
    print(f"📋 Base model: {base_model_path}")
    print(f"📋 LoRA adapter: {lora_path}")
    print(f"📋 Output path: {output_path}")
    print()

    # Load tokenizer from LoRA adapter (has the correct vocabulary)
    print("📝 Loading tokenizer from LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")

    # Load base model
    print("🔧 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Resize base model embeddings to match tokenizer
    original_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    target_vocab_size = len(tokenizer)

    if original_vocab_size != target_vocab_size:
        print(
            f"⚠️ Resizing model embeddings: {original_vocab_size} → {target_vocab_size}"
        )
        base_model.resize_token_embeddings(target_vocab_size)

    # Load LoRA adapter
    print("🔗 Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    # Merge LoRA weights into base model
    print("🔀 Merging LoRA weights...")
    merged_model = model.merge_and_unload()

    # Save merged model
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("✅ Merged model saved successfully!")
    print(f"📁 Model ready for evaluation at: {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter for evaluation")
    parser.add_argument("lora_path", help="Path to LoRA adapter")
    parser.add_argument(
        "--base_model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model path"
    )
    parser.add_argument(
        "--output_path",
        default="./eval_models/merged_model",
        help="Output path for merged model",
    )

    args = parser.parse_args()

    try:
        merged_path = merge_lora_adapter(
            args.base_model, args.lora_path, args.output_path
        )

        print(f"✅ Merged model saved to: {merged_path}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
