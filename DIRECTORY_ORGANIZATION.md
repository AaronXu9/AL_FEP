# AL_FEP Directory Organization

This document describes the organized structure of the AL_FEP project.

## Directory Structure

```
AL_FEP/
├── src/                    # Main package source code
│   └── al_fep/
│       ├── active_learning/   # Active learning strategies
│       ├── molecular/         # Molecular processing and featurization
│       ├── oracles/          # Scoring oracles (FEP, docking, ML)
│       ├── reinforcement/    # Reinforcement learning components
│       └── utils/           # Utility functions
├── examples/               # Example scripts and tutorials
│   ├── docking/           # Docking oracle examples
│   ├── fep/              # FEP oracle examples and tests
│   └── tutorials/        # Comprehensive tutorials
├── tests/                 # Unit and integration tests
│   └── integration/      # Integration tests
├── scripts/               # Utility scripts for setup and preparation
├── notebooks/             # Jupyter notebooks
├── config/                # Configuration files
│   └── targets/          # Target-specific configurations
├── data/                  # Data files
│   ├── external/         # External data sources
│   ├── processed/        # Processed data
│   ├── raw/             # Raw data
│   ├── results/         # Computation results
│   └── targets/         # Target-specific data (e.g., protein structures)
└── temp/                  # Temporary files and debug scripts
```

## Usage

### Running Examples

All example scripts use relative imports and should be run from their respective directories:

```bash
# FEP examples
cd examples/fep
python test_fep_basic.py

# Docking examples  
cd examples/docking
python test_vina_vs_gnina.py

# Tutorials
cd examples/tutorials
python test_7jvr_comprehensive.py
```

### Running Tests

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/
```

### Installing the Package

```bash
# Development install
pip install -e .

# Or using the requirements
pip install -r requirements.txt
```

## Import Structure

The package follows a clean import structure:

```python
# Main package imports
from al_fep import FEPOracle, DockingOracle, MLFEPOracle
from al_fep import ActiveLearningPipeline
from al_fep import PPOAgent, MolecularEnvironment
from al_fep import MolecularDataset, MolecularFeaturizer
from al_fep import setup_logging, load_config

# Direct module imports
from al_fep.oracles.fep_oracle import FEPOracle
from al_fep.molecular.featurizer import MolecularFeaturizer
from al_fep.utils.config import load_config
```

## Development

- **temp/**: Use this directory for temporary development scripts and debug files
- **examples/**: Add new example scripts to the appropriate subdirectory
- **tests/**: Add unit tests to match the src/ structure
- **scripts/**: Add utility scripts for data preparation and setup

## Notes

- All example scripts have been updated with correct relative import paths
- The package can be imported from any location when properly installed
- Configuration files are centralized in `config/`
- Data files are organized by type and processing stage
