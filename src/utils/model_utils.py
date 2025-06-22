"""
Model Utilities

Provides utilities for model and tokenizer operations, including think token setup,
model loading, and model information retrieval.
"""

import logging
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)


def create_think_aware_chat_template():
    """
    Create a clean Jinja2 chat template for KULLM-Pro with think token support.

    Returns:
        str: Jinja2 template string with think token support (no tool calling)
    """
    template = """
{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
{%- else %}
    {{- '<|im_start|>system\\nYou are KULLM-Pro, a hybrid thinking model developed by Korea University NLP&AI Lab. You are a helpful assistant that can show your reasoning process when needed.<|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant") %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if think_mode is defined %}
        {%- if think_mode %}
            {{- '<think>\\n\\n' }}
        {%- else %}
            {{- '<think></think>\\n\\n' }}
        {%- endif %}
    {%- endif %}
{%- endif %}
""".strip()
    return template


def setup_tokenizer_with_think_tokens(
    model_name: str, think_start: str = "<think>", think_end: str = "</think>"
) -> PreTrainedTokenizer:
    """
    Setup tokenizer with think tokens and custom chat template.

    Args:
        model_name: Name or path of the model
        think_start: Start token for thinking
        think_end: End token for thinking

    Returns:
        Tokenizer with think tokens and custom template
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add think tokens as special tokens
    special_tokens = {"additional_special_tokens": [think_start, think_end]}

    num_added = tokenizer.add_special_tokens(special_tokens)
    logger.info(
        f"Added {num_added} special tokens to tokenizer: {think_start}, {think_end}"
    )

    # Set custom chat template with think token support
    tokenizer.chat_template = create_think_aware_chat_template()
    logger.info("Set custom chat template with think token support")

    return tokenizer


def save_tokenizer_with_think_template(tokenizer: PreTrainedTokenizer, output_dir: str):
    """
    Save tokenizer with think-aware chat template to directory.

    Args:
        tokenizer: Tokenizer with custom template
        output_dir: Directory to save tokenizer
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved tokenizer with think-aware template to {output_dir}")


def load_model_and_tokenizer(
    model_name: str,
    think_start: str = "<think>",
    think_end: str = "</think>",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer with think tokens.

    Args:
        model_name: Name or path of the model
        think_start: Start token for thinking
        think_end: End token for thinking
        device_map: Device mapping strategy
        torch_dtype: Torch data type for model
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer: {model_name}")

    # Load tokenizer with think tokens
    tokenizer = setup_tokenizer_with_think_tokens(model_name, think_start, think_end)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens")

    return model, tokenizer


def get_model_info(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Get information about a model.

    Args:
        model: The model to analyze

    Returns:
        Dictionary containing model information
    """
    info = {
        "model_type": model.config.model_type,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "vocab_size": model.config.vocab_size,
        "hidden_size": getattr(model.config, "hidden_size", None),
        "num_layers": getattr(model.config, "num_hidden_layers", None),
        "num_attention_heads": getattr(model.config, "num_attention_heads", None),
    }

    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    info["model_size_mb"] = (param_size + buffer_size) / (1024 * 1024)

    return info


def calculate_model_size(model: PreTrainedModel) -> Dict[str, float]:
    """
    Calculate detailed model size information.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with size information in different units
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate memory usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "size_bytes": total_size,
        "size_mb": total_size / (1024 * 1024),
        "size_gb": total_size / (1024 * 1024 * 1024),
    }


def print_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
    """
    Print comprehensive model information.

    Args:
        model: The model to analyze
        tokenizer: The tokenizer to analyze
    """
    info = get_model_info(model)
    size_info = calculate_model_size(model)

    print("\n" + "=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Model Type: {info['model_type']}")
    print(f"Vocabulary Size: {info['vocab_size']:,}")
    print(f"Tokenizer Length: {len(tokenizer):,}")

    if info["hidden_size"]:
        print(f"Hidden Size: {info['hidden_size']:,}")
    if info["num_layers"]:
        print(f"Number of Layers: {info['num_layers']:,}")
    if info["num_attention_heads"]:
        print(f"Attention Heads: {info['num_attention_heads']:,}")

    print(f"\nParameters:")
    print(f"  Total: {size_info['total_parameters']:,}")
    print(f"  Trainable: {size_info['trainable_parameters']:,}")
    print(f"  Non-trainable: {size_info['non_trainable_parameters']:,}")

    print(f"\nModel Size:")
    print(f"  {size_info['size_mb']:.2f} MB")
    print(f"  {size_info['size_gb']:.2f} GB")
    print("=" * 50 + "\n")


def save_model_info(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_path: str
) -> None:
    """
    Save model information to a JSON file.

    Args:
        model: The model to analyze
        tokenizer: The tokenizer to analyze
        output_path: Path to save the information
    """
    import json

    info = get_model_info(model)
    size_info = calculate_model_size(model)

    combined_info = {
        "model_info": info,
        "size_info": size_info,
        "tokenizer_length": len(tokenizer),
        "special_tokens": tokenizer.special_tokens_map,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_info, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved model information to {output_path}")


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return information.

    Returns:
        Dictionary with GPU information
    """
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        ),
        "device_names": [],
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info["device_names"].append(torch.cuda.get_device_name(i))

    return gpu_info
