# AL-FEP: Active Learning for Free Energy Perturbation in Molecular Virtual Screening

[![CI/CD Pipeline](https://github.com/yourusername/AL_FEP/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/AL_FEP/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive framework for applying active learning and reinforcement learning to molecular virtual screening, with a focus on FEP (Free Energy Perturbation) and docking oracles for target 7JVR (SARS-CoV-2 Main Protease).

## 🚀 Quick Start

### Clone Repository
```bash
git clone https://github.com/yourusername/AL_FEP.git
cd AL_FEP
```

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

## 🌐 Remote Deployment

### GitHub Repository Setup

This project is ready for GitHub deployment with:
- ✅ Git repository initialized
- ✅ Comprehensive `.gitignore` for Python/scientific computing
- ✅ GitHub Actions CI/CD pipeline
- ✅ Pre-commit hooks for code quality
- ✅ Issue and PR templates

### Deploy to Remote Server

1. **Clone on remote server:**
   ```bash
   git clone https://github.com/yourusername/AL_FEP.git
   cd AL_FEP
   ```

2. **Setup environment:**
   ```bash
   conda env create -f environment.yml
   conda activate al_fep
   pip install -e .
   ```

3. **Run tests to verify:**
   ```bash
   python -m pytest tests/ -v
   ```

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## 🛠 Development

### Code Quality Tools
```bash
# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run all quality checks
black src/ tests/           # Code formatting
isort src/ tests/           # Import sorting  
flake8 src/ tests/          # Linting
mypy src/                   # Type checking
```

### Running Tests
```bash
pytest tests/ -v --cov=src/al_fep
```

### Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## 📈 CI/CD Pipeline

The project includes a comprehensive GitHub Actions pipeline that:
- Tests across Python 3.9, 3.10, 3.11 on Ubuntu and macOS
- Runs linting, formatting, and type checking
- Performs security vulnerability scanning
- Builds and validates the package

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📞 Support

- 📖 Documentation: See notebooks and docstrings
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/AL_FEP/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/AL_FEP/discussions)
- 📧 Contact: your.email@example.com

## 🏆 Acknowledgments

- RDKit for molecular handling
- OpenMM for molecular dynamics
- AutoDock Vina for docking
- PyTorch for machine learning

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
