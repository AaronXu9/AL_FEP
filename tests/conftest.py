"""
Test configuration and utilities.
"""

import pytest
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import tempfile
import os
from typing import List, Dict, Any

# Test molecules (valid SMILES)
TEST_SMILES = [
    "CCO",  # Ethanol
    "CC(=O)O",  # Acetic acid
    "c1ccccc1",  # Benzene
    "CC(C)O",  # Isopropanol
    "CCN(CC)CC",  # Triethylamine
    "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
    "CC(C)(C)c1ccc(O)cc1",  # 4-tert-butylphenol
    "COc1ccc(CC(=O)O)cc1",  # 4-methoxyphenylacetic acid
    "Nc1ccc(C(=O)O)cc1",  # 4-aminobenzoic acid
    "CC1=CC(=O)C=CC1=O"  # 2-methyl-1,4-benzoquinone
]

# Invalid SMILES for testing error handling
INVALID_SMILES = [
    "INVALID",
    "C(C)(C",
    "CC1CC",
    "",
    "XYZ"
]

# Test molecular data
def generate_test_data(n_molecules: int = 100, include_targets: bool = True) -> pd.DataFrame:
    """Generate test molecular data."""
    np.random.seed(42)
    
    # Use test SMILES and add random variations
    base_smiles = TEST_SMILES * (n_molecules // len(TEST_SMILES) + 1)
    smiles = base_smiles[:n_molecules]
    
    data = {"SMILES": smiles}
    
    if include_targets:
        # Generate synthetic targets (binding affinity)
        targets = np.random.normal(5.0, 2.0, n_molecules)  # pIC50 values
        data["target"] = targets
        
        # Add some molecular properties
        properties = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                props = {
                    "MW": Descriptors.MolWt(mol),
                    "LogP": Descriptors.MolLogP(mol),
                    "HBD": Descriptors.NumHDonors(mol),
                    "HBA": Descriptors.NumHAcceptors(mol)
                }
            else:
                props = {"MW": np.nan, "LogP": np.nan, "HBD": np.nan, "HBA": np.nan}
            properties.append(props)
        
        prop_df = pd.DataFrame(properties)
        data.update(prop_df.to_dict('list'))
    
    return pd.DataFrame(data)


def create_temp_config(config_dict: Dict[str, Any]) -> str:
    """Create a temporary configuration file."""
    import yaml
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config_dict, temp_file)
    temp_file.close()
    return temp_file.name


@pytest.fixture
def sample_smiles():
    """Fixture providing sample SMILES strings."""
    return TEST_SMILES.copy()


@pytest.fixture
def invalid_smiles():
    """Fixture providing invalid SMILES strings."""
    return INVALID_SMILES.copy()


@pytest.fixture
def sample_molecular_data():
    """Fixture providing sample molecular dataset."""
    return generate_test_data(n_molecules=50)


@pytest.fixture
def temp_dir():
    """Fixture providing temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_features():
    """Fixture providing sample molecular features."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    return np.random.randn(n_samples, n_features)


@pytest.fixture
def sample_targets():
    """Fixture providing sample target values."""
    np.random.seed(42)
    n_samples = 100
    return np.random.normal(5.0, 2.0, n_samples)


class MockOracle:
    """Mock oracle for testing."""
    
    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
        self.call_count = 0
    
    def evaluate(self, smiles: str) -> float:
        """Mock evaluation based on molecular weight."""
        self.call_count += 1
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        
        # Simple scoring based on MW (with noise)
        mw = Descriptors.MolWt(mol)
        score = 10.0 - abs(mw - 300) / 100.0  # Prefer MW around 300
        noise = np.random.normal(0, self.noise_level)
        return max(0.0, score + noise)
    
    def batch_evaluate(self, smiles_list: List[str]) -> List[float]:
        """Batch evaluation."""
        return [self.evaluate(smi) for smi in smiles_list]


class MockModel:
    """Mock ML model for testing."""
    
    def __init__(self):
        self.is_fitted = False
        self.feature_dim = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Mock fitting."""
        self.is_fitted = True
        self.feature_dim = X.shape[1]
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Simple prediction based on feature mean
        return np.mean(X, axis=1) + np.random.normal(0, 0.1, len(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Mock probability prediction."""
        predictions = self.predict(X)
        # Create fake probabilities
        prob_pos = 1 / (1 + np.exp(-predictions))  # Sigmoid
        return np.column_stack([1 - prob_pos, prob_pos])


# Test constants
TEST_CONFIG = {
    'target': '7jvr',
    'data_dir': './data',
    'oracle': {
        'type': 'docking',
        'software': 'vina',
        'receptor_file': 'receptor.pdbqt',
        'binding_site': {
            'center': [0, 0, 0],
            'size': [20, 20, 20]
        }
    },
    'active_learning': {
        'strategy': 'uncertainty',
        'batch_size': 10,
        'max_iterations': 5
    },
    'molecular': {
        'featurizer': 'morgan',
        'fingerprint_bits': 1024
    }
}

TOLERANCE = 1e-6  # Numerical tolerance for tests
