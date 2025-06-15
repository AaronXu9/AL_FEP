#!/usr/bin/env python3
"""
Setup script for AL-FEP package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core requirements (minimal for basic functionality)
core_requirements = [
    "numpy>=1.20.0",
    "pandas>=1.3.0", 
    "scikit-learn>=1.0.0",
    "torch>=1.10.0",
    "rdkit-pypi>=2022.3.0",
    "pyyaml>=6.0",
    "tqdm>=4.60.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "joblib>=1.1.0",
    "scipy>=1.7.0",
]

setup(
    name="al-fep",
    version="0.1.0",
    author="AL-FEP Team",
    author_email="contact@alfep.org",
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
    install_requires=core_requirements,
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
        "full": [
            # Optional heavy dependencies
            "openmm>=7.7.0",
            "mdtraj>=1.9.0", 
            "openff-toolkit>=0.11.0",
            "wandb>=0.17.0",
            "gym>=0.21.0",
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
