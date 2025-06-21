"""
Tests for oracle implementations
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Test imports
from al_fep.oracles.base_oracle import BaseOracle
from al_fep.oracles.ml_fep_oracle import MLFEPOracle
from al_fep.oracles.docking_oracle import DockingOracle
from al_fep.oracles.fep_oracle import FEPOracle


# Mock oracle for testing base functionality
class MockTestOracle(BaseOracle):
    """Test implementation of BaseOracle for testing."""
    
    def _evaluate_single(self, smiles: str) -> Dict[str, Any]:
        """Mock evaluation that returns a simple score."""
        if smiles == "invalid":
            return {"score": None, "error": "Invalid SMILES"}
        return {"score": len(smiles) * 0.5, "method": "test"}


class TestBaseOracle:
    """Test BaseOracle functionality."""
    
    def test_init(self):
        """Test oracle initialization."""
        oracle = MockTestOracle(name="TestOracle", target="test_target")
        
        assert oracle.name == "TestOracle"
        assert oracle.target == "test_target"
        assert oracle.call_count == 0
        assert oracle.total_time == 0.0
        assert oracle.cache == True
    
    def test_init_with_config(self):
        """Test oracle initialization with configuration."""
        config = {"test_param": "test_value"}
        oracle = MockTestOracle(name="TestOracle", target="test_target", config=config)
        
        assert oracle.config == config
    
    def test_evaluate_single_molecule(self):
        """Test single molecule evaluation."""
        oracle = MockTestOracle(name="TestOracle", target="test_target")
        
        result = oracle.evaluate("CCO")
        
        assert isinstance(result, dict)
        assert "score" in result
        assert result["score"] == 1.5  # len("CCO") * 0.5
        assert oracle.call_count == 1
    
    def test_evaluate_multiple_molecules(self):
        """Test multiple molecule evaluation."""
        oracle = MockTestOracle(name="TestOracle", target="test_target")
        
        molecules = ["CCO", "CCCC", "c1ccccc1"]
        results = oracle.evaluate(molecules)
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
        assert oracle.call_count == 3
    
    def test_caching(self):
        """Test result caching."""
        oracle = MockTestOracle(name="TestOracle", target="test_target", cache=True)
        
        # First evaluation
        result1 = oracle.evaluate("CCO")
        assert oracle.call_count == 1
        
        # Second evaluation should use cache
        result2 = oracle.evaluate("CCO")
        assert oracle.call_count == 1  # Should not increment
        assert result1 == result2
    
    def test_no_caching(self):
        """Test behavior without caching."""
        oracle = MockTestOracle(name="TestOracle", target="test_target", cache=False)
        
        # First evaluation
        oracle.evaluate("CCO")
        assert oracle.call_count == 1
        
        # Second evaluation should not use cache
        oracle.evaluate("CCO")
        assert oracle.call_count == 2
    
    def test_error_handling(self):
        """Test error handling for invalid molecules."""
        oracle = MockTestOracle(name="TestOracle", target="test_target")
        
        result = oracle.evaluate("invalid")
        
        assert result["score"] is None
        assert "error" in result
    
    def test_get_statistics(self):
        """Test statistics collection."""
        oracle = MockTestOracle(name="TestOracle", target="test_target")
        
        # Make some evaluations
        oracle.evaluate(["CCO", "CCCC", "c1ccccc1"])
        
        stats = oracle.get_statistics()
        
        assert isinstance(stats, dict)
        assert stats["call_count"] == 3
        assert stats["total_time"] >= 0.0
        assert stats["avg_time"] >= 0.0


class TestMLFEPOracle:
    """Test ML-FEP Oracle functionality."""
    
    def test_init(self):
        """Test ML-FEP oracle initialization."""
        oracle = MLFEPOracle(target="test_target")
        
        assert oracle.name == "ML-FEP"
        assert oracle.target == "test_target"
        assert oracle.is_trained == True  # Should initialize with mock data
        assert oracle.models is not None
        assert oracle.scaler is not None
    
    def test_init_with_config(self):
        """Test ML-FEP oracle initialization with configuration."""
        config = {
            "ml_fep": {
                "model_type": "ensemble",
                "uncertainty_threshold": 0.3,
                "retrain_frequency": 50
            }
        }
        oracle = MLFEPOracle(target="test_target", config=config)
        
        assert oracle.model_type == "ensemble"
        assert oracle.uncertainty_threshold == 0.3
        assert oracle.retrain_frequency == 50
    
    def test_evaluate_valid_molecule(self):
        """Test evaluation of valid molecules."""
        oracle = MLFEPOracle(target="test_target")
        
        result = oracle.evaluate("c1ccccc1")  # benzene
        
        assert isinstance(result, dict)
        assert "score" in result
        assert "ml_fep_score" in result
        assert "uncertainty" in result
        assert "confidence" in result
        assert result["score"] is not None
        assert result["error"] is None
    
    def test_evaluate_invalid_molecule(self):
        """Test evaluation of invalid molecules."""
        oracle = MLFEPOracle(target="test_target")
        
        result = oracle.evaluate("INVALID_SMILES")
        
        assert result["score"] is None
        assert "error" in result
        assert "Failed to calculate molecular descriptors" in result["error"]
    
    def test_descriptor_calculation(self):
        """Test molecular descriptor calculation."""
        oracle = MLFEPOracle(target="test_target")
        
        features = oracle._calculate_molecular_descriptors("CCO")
        
        assert features is not None
        assert features.shape == (1, 13)  # 13 descriptors
    
    def test_descriptor_calculation_invalid(self):
        """Test descriptor calculation for invalid SMILES."""
        oracle = MLFEPOracle(target="test_target")
        
        features = oracle._calculate_molecular_descriptors("INVALID")
        
        assert features is None
    
    def test_prediction_with_uncertainty(self):
        """Test prediction with uncertainty estimation."""
        oracle = MLFEPOracle(target="test_target")
        
        # Create dummy features
        features = np.random.randn(1, 13)
        
        prediction, uncertainty = oracle._predict_with_uncertainty(features)
        
        assert isinstance(prediction, float)
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0.0
    
    def test_retrain_model(self):
        """Test model retraining."""
        oracle = MLFEPOracle(target="test_target")
        
        # Create training data
        smiles_list = ["CCO", "CCC", "c1ccccc1"]
        fep_scores = [-5.2, -4.1, -6.3]
        
        oracle.retrain_model(smiles_list, fep_scores)
        
        # Check that model is still trained
        assert oracle.is_trained == True
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        oracle = MLFEPOracle(target="test_target")
        
        importance = oracle.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(oracle.feature_names)
        assert all(isinstance(v, float) for v in importance.values())
    
    @patch('pickle.dump')
    @patch('os.makedirs')
    def test_save_model(self, mock_makedirs, mock_pickle_dump):
        """Test model saving."""
        oracle = MLFEPOracle(target="test_target")
        
        with patch('builtins.open', create=True) as mock_open:
            oracle.save_model()
            mock_open.assert_called_once()
            mock_pickle_dump.assert_called_once()


class TestDockingOracle:
    """Test Docking Oracle functionality."""
    
    def test_init(self):
        """Test docking oracle initialization."""
        oracle = DockingOracle(target="test_target")
        
        assert oracle.name == "Docking"
        assert oracle.target == "test_target"
        assert oracle.center_x == 0.0
        assert oracle.center_y == 0.0
        assert oracle.center_z == 0.0
    
    def test_init_with_config(self):
        """Test docking oracle initialization with configuration."""
        config = {
            "docking": {
                "center_x": 10.5,
                "center_y": -7.2,
                "center_z": 15.8,
                "size_x": 20.0,
                "size_y": 20.0,
                "size_z": 20.0,
                "exhaustiveness": 12
            }
        }
        oracle = DockingOracle(target="test_target", config=config)
        
        assert oracle.center_x == 10.5
        assert oracle.center_y == -7.2
        assert oracle.center_z == 15.8
        assert oracle.exhaustiveness == 12
    
    def test_smiles_to_3d(self):
        """Test SMILES to 3D conversion."""
        oracle = DockingOracle(target="test_target")
        
        mol_3d = oracle._smiles_to_3d("CCO")
        
        assert mol_3d is not None
        assert mol_3d.GetNumConformers() > 0
    
    def test_smiles_to_3d_invalid(self):
        """Test SMILES to 3D conversion for invalid SMILES."""
        oracle = DockingOracle(target="test_target")
        
        mol_3d = oracle._smiles_to_3d("INVALID")
        
        assert mol_3d is None
    
    @patch('subprocess.run')
    def test_run_vina_docking_mock(self, mock_subprocess):
        """Test Vina docking with mocked subprocess."""
        # Mock successful vina output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "   1         -7.5      0.000      0.000\n"
        
        oracle = DockingOracle(target="test_target")
        
        with tempfile.NamedTemporaryFile(suffix=".pdbqt") as temp_file:
            result = oracle._run_vina_docking(temp_file.name)
            
            assert result is not None
            assert "affinity" in result
    
    @patch('subprocess.run')
    def test_run_vina_docking_failure(self, mock_subprocess):
        """Test Vina docking failure."""
        # Mock failed vina execution
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Error message"
        
        oracle = DockingOracle(target="test_target")
        
        with tempfile.NamedTemporaryFile(suffix=".pdbqt") as temp_file:
            result = oracle._run_vina_docking(temp_file.name)
            
            assert result is None
    
    def test_evaluate_single_mock_mode(self):
        """Test single molecule evaluation in mock mode."""
        oracle = DockingOracle(target="test_target")
        oracle.mock_mode = True  # Enable mock mode
        
        result = oracle._evaluate_single("CCO")
        
        assert isinstance(result, dict)
        assert "score" in result
        assert result["score"] is not None
        assert result["method"] == "Docking (Mock)"


class TestFEPOracle:
    """Test FEP Oracle functionality."""
    
    def test_init(self):
        """Test FEP oracle initialization."""
        oracle = FEPOracle(target="test_target")
        
        assert oracle.name == "FEP"
        assert oracle.target == "test_target"
        assert oracle.force_field == "amber14"
        assert oracle.water_model == "tip3p"
    
    def test_init_with_config(self):
        """Test FEP oracle initialization with configuration."""
        config = {
            "fep": {
                "force_field": "charmm36",
                "water_model": "tip4p",
                "num_lambda_windows": 20,
                "simulation_time": 10.0,
                "temperature": 310.0
            }
        }
        oracle = FEPOracle(target="test_target", config=config)
        
        assert oracle.force_field == "charmm36"
        assert oracle.water_model == "tip4p"
        assert oracle.num_lambda_windows == 20
        assert oracle.simulation_time == 10.0
        assert oracle.temperature == 310.0
    
    def test_prepare_ligand(self):
        """Test ligand preparation."""
        oracle = FEPOracle(target="test_target")
        
        mol = oracle._prepare_ligand("CCO")
        
        assert mol is not None
        assert mol.GetNumConformers() > 0
    
    def test_prepare_ligand_invalid(self):
        """Test ligand preparation for invalid SMILES."""
        oracle = FEPOracle(target="test_target")
        
        mol = oracle._prepare_ligand("INVALID")
        
        assert mol is None
    
    def test_calculate_mock_fep(self):
        """Test mock FEP calculation."""
        oracle = FEPOracle(target="test_target")
        
        delta_g = oracle._calculate_mock_fep("CCO")
        
        assert isinstance(delta_g, float)
        assert -15.0 <= delta_g <= 5.0  # Reasonable FEP range
    
    def test_evaluate_single_mock_mode(self):
        """Test single molecule evaluation in mock mode."""
        config = {"fep": {"mock_mode": True}}
        oracle = FEPOracle(target="test_target", config=config)
        
        result = oracle._evaluate_single("CCO")
        
        assert isinstance(result, dict)
        assert "score" in result
        assert "fep_score" in result
        assert result["score"] is not None
        assert result["method"] == "FEP (Mock)"
    
    def test_evaluate_single_invalid(self):
        """Test evaluation of invalid molecule."""
        config = {"fep": {"mock_mode": True}}
        oracle = FEPOracle(target="test_target", config=config)
        
        result = oracle._evaluate_single("INVALID")
        
        assert result["score"] is None
        assert "error" in result


class TestOracleIntegration:
    """Integration tests for multiple oracles."""
    
    def test_oracle_comparison(self):
        """Test using multiple oracles on the same molecules."""
        ml_oracle = MLFEPOracle(target="test")
        docking_oracle = DockingOracle(target="test", config={"docking": {"mock_mode": True}})
        fep_oracle = FEPOracle(target="test", config={"fep": {"mock_mode": True}})
        
        test_molecules = ["CCO", "c1ccccc1", "CCC"]
        
        ml_results = ml_oracle.evaluate(test_molecules)
        docking_results = docking_oracle.evaluate(test_molecules)
        fep_results = fep_oracle.evaluate(test_molecules)
        
        assert len(ml_results) == len(test_molecules)
        assert len(docking_results) == len(test_molecules)
        assert len(fep_results) == len(test_molecules)
        
        # Check that all results have scores
        for results in [ml_results, docking_results, fep_results]:
            for result in results:
                assert "score" in result
    
    def test_oracle_statistics(self):
        """Test statistics collection across oracles."""
        oracles = [
            MLFEPOracle(target="test"),
            DockingOracle(target="test", config={"docking": {"mock_mode": True}}),
            FEPOracle(target="test", config={"fep": {"mock_mode": True}})
        ]
        
        test_molecules = ["CCO", "CCC"]
        
        # Evaluate with all oracles
        for oracle in oracles:
            oracle.evaluate(test_molecules)
        
        # Check statistics
        for oracle in oracles:
            stats = oracle.get_statistics()
            assert stats["call_count"] == 2
            assert stats["total_time"] >= 0.0


# Fixtures for oracle testing
@pytest.fixture
def ml_fep_oracle():
    """Create ML-FEP oracle for testing."""
    return MLFEPOracle(target="test_target")


@pytest.fixture
def docking_oracle():
    """Create docking oracle for testing."""
    config = {"docking": {"mock_mode": True}}
    oracle = DockingOracle(target="test_target", config=config)
    return oracle


@pytest.fixture
def fep_oracle():
    """Create FEP oracle for testing."""
    config = {"fep": {"mock_mode": True}}
    oracle = FEPOracle(target="test_target", config=config)
    return oracle


@pytest.fixture
def sample_molecules():
    """Sample molecules for testing."""
    return ["CCO", "c1ccccc1", "CCC", "CC(=O)O"]


# Parametrized tests for all oracle types
@pytest.mark.parametrize("oracle_name", ["ml_fep_oracle", "docking_oracle", "fep_oracle"])
def test_oracle_basic_functionality(oracle_name, request, sample_molecules):
    """Test basic functionality for all oracle types."""
    oracle = request.getfixturevalue(oracle_name)
    
    # Test single molecule evaluation
    result = oracle.evaluate(sample_molecules[0])
    assert isinstance(result, dict)
    assert "score" in result
    
    # Test batch evaluation
    results = oracle.evaluate(sample_molecules)
    assert isinstance(results, list)
    assert len(results) == len(sample_molecules)


if __name__ == "__main__":
    pytest.main([__file__])
