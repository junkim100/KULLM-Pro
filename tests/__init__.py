"""
Test suite for KULLM-Pro

This package contains unit tests, integration tests, and test utilities
for the KULLM-Pro Korean-English code-switched language model training pipeline.

Test Structure:
- test_code_switching/: Tests for code switching functionality
- test_fine_tuning/: Tests for fine-tuning functionality  
- test_utils/: Tests for utility functions
- test_cli/: Tests for command-line interfaces
- conftest.py: Shared test fixtures and configuration
- utils.py: Test utility functions

Running Tests:
    # Run all tests
    pytest tests/
    
    # Run specific test categories
    pytest tests/test_code_switching/
    pytest tests/test_fine_tuning/
    
    # Run with coverage
    pytest tests/ --cov=kullm_pro --cov-report=html
    
    # Run only fast tests (exclude slow/integration tests)
    pytest tests/ -m "not slow"
    
    # Run only tests that don't require GPU
    pytest tests/ -m "not gpu"
    
    # Run only tests that don't require API access
    pytest tests/ -m "not api"

Test Markers:
- slow: Tests that take a long time to run
- integration: Integration tests that test multiple components
- gpu: Tests that require GPU access
- api: Tests that require external API access (OpenAI, etc.)

Environment Variables for Testing:
- KULLM_TEST_OPENAI_API_KEY: OpenAI API key for API tests
- KULLM_TEST_WANDB_API_KEY: WandB API key for integration tests
- KULLM_TEST_DATA_DIR: Directory for test data files
- KULLM_TEST_OUTPUT_DIR: Directory for test outputs
"""

__version__ = "1.0.0"
