"""
AL-FEP: Active Learning for Free Energy Perturbation in Molecular Virtual Screening
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import with error handling to avoid dependency issues during development
__all__ = []

try:
    from .oracles import FEPOracle, DockingOracle, MLFEPOracle
    __all__.extend(["FEPOracle", "DockingOracle", "MLFEPOracle"])
except ImportError:
    pass

try:
    from .active_learning import ActiveLearningPipeline
    __all__.append("ActiveLearningPipeline")
except ImportError:
    pass

try:
    from .reinforcement import PPOAgent, MolecularEnvironment
    __all__.extend(["PPOAgent", "MolecularEnvironment"])
except ImportError:
    pass

try:
    from .molecular import MolecularDataset, MolecularFeaturizer
    __all__.extend(["MolecularDataset", "MolecularFeaturizer"])
except ImportError:
    pass

try:
    from .utils import setup_logging, load_config
    __all__.extend(["setup_logging", "load_config"])
except ImportError:
    pass
