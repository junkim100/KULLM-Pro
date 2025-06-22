#!/usr/bin/env python3
"""
Setup script for KULLM-Pro: Korean University Large Language Model - Professional Edition
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="kullm-pro",
    version="1.1.0",
    author="Korea University NLP&AI Lab",
    author_email="junkim100@gmail.com",
    description="Korean University Large Language Model - Professional Edition with Think Token Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/junkim100/KULLM-Pro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "pre-commit>=2.15",
        ],
        "training": [
            "wandb>=0.15.0",
            "tensorboard>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kullm-chat=scripts.chat:main",
            "kullm-train=src.fine_tune:main",
            "kullm-code-switch=src.code_switch:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["*.yaml", "*.txt"],
        "configs": ["*.yaml"],
    },
    keywords="llm, korean, nlp, ai, machine-learning, transformers, think-tokens, reasoning",
    project_urls={
        "Bug Reports": "https://github.com/junkim100/KULLM-Pro/issues",
        "Source": "https://github.com/junkim100/KULLM-Pro",
        "Documentation": "https://github.com/junkim100/KULLM-Pro/blob/main/README.md",
    },
)
