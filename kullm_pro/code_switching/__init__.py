"""
Code Switching Module

Handles dataset processing and code-switching generation using OpenAI API.
Supports flexible dataset loading from Hugging Face with configurable parameters.
"""

from .pipeline import CodeSwitchingPipeline
from .dataset_processor import DatasetProcessor
from .openai_client import OpenAIClient, OpenAIConfig

__all__ = [
    "CodeSwitchingPipeline",
    "DatasetProcessor", 
    "OpenAIClient",
    "OpenAIConfig",
]
