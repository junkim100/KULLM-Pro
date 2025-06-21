"""
Pytest configuration and shared fixtures for KULLM-Pro tests
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_DATA_DIR.mkdir(exist_ok=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "model": {
            "name": "microsoft/DialoGPT-small",  # Small model for testing
            "max_length": 512,
            "torch_dtype": "float32",
        },
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.0002,
            "gradient_accumulation_steps": 2,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "eval_steps": 10,
            "save_steps": 20,
            "logging_steps": 5,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "bf16": False,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 0,
            "optim": "adamw_torch",
        },
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
        },
        "wandb": {
            "project": "kullm-pro-test",
            "enabled": False,
        },
        "data": {
            "train_split_ratio": 0.8,
        },
        "openai": {
            "model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 1.0,
            "use_batch_api": False,
            "batch_size": 10,
            "max_concurrent_requests": 2,
            "max_retries": 1,
            "timeout": 30,
        },
    }


@pytest.fixture
def sample_training_data():
    """Sample training data in the expected format"""
    return [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that solves mathematical problems.",
                },
                {
                    "role": "user", 
                    "content": "Solve: 2 + 2 = ?",
                },
                {
                    "role": "assistant",
                    "content": "2 + 2 = 4",
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that solves mathematical problems.",
                },
                {
                    "role": "user",
                    "content": "What is 5 * 3?",
                },
                {
                    "role": "assistant", 
                    "content": "5 * 3 = 15",
                },
            ]
        },
    ]


@pytest.fixture
def sample_dataset_data():
    """Sample dataset data for code switching tests"""
    return [
        {
            "question": "What is 2 + 2?",
            "solution": "To solve this, we add 2 + 2 = 4",
            "answer": "4",
        },
        {
            "question": "Calculate 5 * 3",
            "solution": "We multiply 5 by 3 to get 5 * 3 = 15",
            "answer": "15",
        },
    ]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    with patch("kullm_pro.code_switching.openai_client.OpenAI") as mock_client:
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Korean-English code-switched response"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.return_value.chat.completions.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_wandb():
    """Mock Weights & Biases for testing"""
    with patch("kullm_pro.fine_tuning.trainer.wandb") as mock_wandb:
        mock_wandb.init.return_value = Mock()
        mock_wandb.finish.return_value = None
        yield mock_wandb


@pytest.fixture
def skip_gpu_tests():
    """Skip tests that require GPU if not available"""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture
def skip_api_tests():
    """Skip tests that require API access if credentials not available"""
    if not os.getenv("KULLM_TEST_OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not available for testing")


@pytest.fixture
def mock_huggingface_dataset():
    """Mock Hugging Face dataset for testing"""
    with patch("kullm_pro.code_switching.dataset_processor.load_dataset") as mock_load:
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.to_pandas.return_value.to_dict.return_value = [
            {
                "question": "What is 2 + 2?",
                "solution": "2 + 2 = 4",
                "answer": "4",
            }
        ]
        mock_load.return_value = mock_dataset
        yield mock_load


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "api: mark test as requiring API access")


# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "slow" in item.nodeid or "train" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        
        # Mark API tests
        if "api" in item.nodeid or "openai" in item.nodeid:
            item.add_marker(pytest.mark.api)
