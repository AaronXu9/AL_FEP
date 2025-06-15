# AL-FEP: Active Learning for Free Energy Perturbation in Molecular Virtual Screening

A comprehensive framework for applying active learning and reinforcement learning to molecular virtual screening, with a focus on FEP (Free Energy Perturbation) and docking oracles for target 7JVR.

## Project Overview

This project implements:
- **Active Learning**: Iterative molecular selection and evaluation
- **Reinforcement Learning**: Agent-based molecular discovery
- **Multi-Oracle System**: FEP, Docking, and ML-FEP evaluations
- **Target-Specific**: Optimized for 7JVR protein target

## Quick Start

### 1. Environment Setup

Create and activate the conda environment:

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate al_fep

# Verify installation
python -c "import rdkit; print('RDKit version:', rdkit.__version__)"
```

### 2. Project Structure

```
AL_FEP/
├── environment.yml          # Conda environment specification
├── requirements.txt         # Additional pip requirements
├── setup.py                # Package installation
├── config/                 # Configuration files
│   ├── targets/            # Target-specific configs
│   └── experiments/        # Experiment configurations
├── src/                    # Source code
│   ├── al_fep/            # Main package
│   │   ├── oracles/       # FEP, Docking, ML-FEP oracles
│   │   ├── active_learning/ # AL algorithms
│   │   ├── reinforcement/  # RL algorithms
│   │   ├── molecular/     # Molecular utilities
│   │   └── utils/         # Common utilities
├── data/                  # Data directory
│   ├── targets/          # Target protein structures
│   ├── molecules/        # Molecular datasets
│   └── results/          # Experiment results
├── notebooks/            # Jupyter notebooks
├── scripts/              # Standalone scripts
└── tests/               # Unit tests
```

### 3. Target 7JVR Setup

The project is pre-configured for the 7JVR target. Key files:
- `config/targets/7jvr.yaml`: Target-specific parameters
- `data/targets/7jvr/`: Protein structures and binding site info
- `notebooks/01_7jvr_analysis.ipynb`: Target analysis notebook

## Oracle Systems

### 1. FEP Oracle
- High-accuracy free energy calculations
- GPU-accelerated simulations
- AMBER/GROMACS integration

### 2. Docking Oracle
- AutoDock Vina integration
- Multiple conformer generation
- Binding pose analysis

### 3. ML-FEP Oracle
- Fast ML-based FEP predictions
- Pre-trained on experimental data
- Cost-effective screening

## Active Learning Workflows

1. **Uncertainty Sampling**: Select molecules with highest prediction uncertainty
2. **Query by Committee**: Ensemble-based selection
3. **Expected Improvement**: Optimize acquisition functions
4. **Diversity-Based**: Ensure chemical space coverage

## Reinforcement Learning Agents

1. **Molecular REINFORCE**: Policy gradient for molecular generation
2. **Actor-Critic**: Value-based molecular optimization
3. **PPO**: Proximal policy optimization for stable training
4. **Multi-Objective**: Balance multiple molecular properties

## Usage Examples

### Basic Active Learning Run
```python
from al_fep import ActiveLearningPipeline
from al_fep.oracles import FEPOracle, DockingOracle

# Initialize oracles
fep_oracle = FEPOracle(target="7jvr")
docking_oracle = DockingOracle(target="7jvr")

# Setup active learning
al_pipeline = ActiveLearningPipeline(
    oracles=[fep_oracle, docking_oracle],
    strategy="uncertainty_sampling",
    budget=100
)

# Run active learning loop
results = al_pipeline.run()
```

### Reinforcement Learning Training
```python
from al_fep import RLAgent
from al_fep.environments import MolecularEnv

# Setup environment
env = MolecularEnv(target="7jvr", oracle="ml_fep")

# Initialize agent
agent = RLAgent(algorithm="ppo", env=env)

# Train agent
agent.train(total_timesteps=100000)
```

## Configuration

All experiments are configured via YAML files in `config/`:
- Global settings in `config/default.yaml`
- Target-specific in `config/targets/7jvr.yaml`
- Experiment-specific in `config/experiments/`

## Development

### Running Tests
```bash
pytest tests/ -v --cov=src/al_fep
```

### Code Formatting
```bash
black src/ tests/
flake8 src/ tests/
```

### Type Checking
```bash
mypy src/al_fep
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```bibtex
@article{al_fep_2025,
  title={Active Learning and Reinforcement Learning for Molecular Virtual Screening},
  author={Your Name},
  journal={Journal of Chemical Information and Modeling},
  year={2025}
}
```
