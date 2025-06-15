#!/usr/bin/env python3
"""
Setup script for AL-FEP package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="al-fep",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Active Learning and Reinforcement Learning for Molecular Virtual Screening",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AL_FEP",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "pre-commit",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
        ],
        "gpu": [
            "cupy",
            "torch-geometric",
        ],
    },
    entry_points={
        "console_scripts": [
            "al-fep=al_fep.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
