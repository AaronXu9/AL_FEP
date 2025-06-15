"""
Tests for molecular featurization utilities.
"""

import pytest
import numpy as np
from rdkit import Chem

from al_fep.molecular.featurizer import (
    MolecularFeaturizer, 
    DescriptorCalculator, 
    batch_featurize
)


class TestMolecularFeaturizer:
    """Test MolecularFeaturizer class."""
    
    def test_init_default(self):
        """Test default initialization."""
        featurizer = MolecularFeaturizer()
        assert featurizer.fingerprint_type == "morgan"
        assert featurizer.fingerprint_radius == 2
        assert featurizer.fingerprint_bits == 2048
        assert featurizer.include_descriptors is True
        assert featurizer.include_fragments is False
    
    def test_init_custom(self):
        """Test custom initialization."""
        featurizer = MolecularFeaturizer(
            fingerprint_type="rdkit",
            fingerprint_radius=3,
            fingerprint_bits=1024,
            include_descriptors=False,
            include_fragments=True
        )
        assert featurizer.fingerprint_type == "rdkit"
        assert featurizer.fingerprint_radius == 3
        assert featurizer.fingerprint_bits == 1024
        assert featurizer.include_descriptors is False
        assert featurizer.include_fragments is True
    
    def test_featurize_valid_molecule(self, sample_smiles):
        """Test featurization of valid molecules."""
        featurizer = MolecularFeaturizer()
        
        for smiles in sample_smiles[:3]:  # Test first 3
            features = featurizer.featurize_molecule(smiles)
            assert features is not None
            assert isinstance(features, np.ndarray)
            assert len(features) > 0
            assert features.dtype == np.float32
    
    def test_featurize_invalid_molecule(self, invalid_smiles):
        """Test featurization of invalid molecules."""
        featurizer = MolecularFeaturizer()
        
        for smiles in invalid_smiles:
            features = featurizer.featurize_molecule(smiles)
            assert features is None
    
    def test_featurize_molecules_batch(self, sample_smiles):
        """Test batch featurization."""
        featurizer = MolecularFeaturizer()
        
        features, valid_smiles = featurizer.featurize_molecules(sample_smiles)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == len(valid_smiles)
        assert len(valid_smiles) <= len(sample_smiles)  # Some might be invalid
        assert all(smi in sample_smiles for smi in valid_smiles)
    
    def test_different_fingerprint_types(self, sample_smiles):
        """Test different fingerprint types."""
        fingerprint_types = ["morgan", "rdkit", "maccs", "topological"]
        test_smiles = sample_smiles[0]  # Use one molecule
        
        for fp_type in fingerprint_types:
            featurizer = MolecularFeaturizer(fingerprint_type=fp_type)
            features = featurizer.featurize_molecule(test_smiles)
            
            assert features is not None
            assert len(features) > 0
            
            # Check expected dimensions
            if fp_type == "maccs":
                expected_fp_size = 167
            else:
                expected_fp_size = featurizer.fingerprint_bits
            
            # Should include fingerprint + descriptors (default)
            expected_total = expected_fp_size + 21  # 21 descriptors
            assert len(features) == expected_total
    
    def test_feature_dimensions(self):
        """Test feature dimension calculation."""
        # Morgan fingerprint with descriptors
        featurizer = MolecularFeaturizer(
            fingerprint_type="morgan",
            fingerprint_bits=1024,
            include_descriptors=True,
            include_fragments=False
        )
        expected_dim = 1024 + 21  # fingerprint + descriptors
        assert featurizer.get_feature_dim() == expected_dim
        
        # MACCS with descriptors and fragments
        featurizer = MolecularFeaturizer(
            fingerprint_type="maccs",
            include_descriptors=True,
            include_fragments=True
        )
        expected_dim = 167 + 21 + 45  # MACCS + descriptors + fragments
        assert featurizer.get_feature_dim() == expected_dim
    
    def test_feature_names(self, sample_smiles):
        """Test feature name generation."""
        featurizer = MolecularFeaturizer(
            fingerprint_type="morgan",
            fingerprint_bits=512,
            include_descriptors=True,
            include_fragments=False
        )
        
        # Need to featurize at least one molecule to initialize names
        featurizer.featurize_molecule(sample_smiles[0])
        
        names = featurizer.get_feature_names()
        assert len(names) == featurizer.get_feature_dim()
        assert any("MORGAN_" in name for name in names)
        assert any(name in ["MolWt", "LogP", "NumHDonors"] for name in names)
    
    def test_descriptors_only(self, sample_smiles):
        """Test featurization with descriptors only."""
        featurizer = MolecularFeaturizer(
            fingerprint_bits=0,  # This won't actually work, but we can test the logic
            include_descriptors=True,
            include_fragments=False
        )
        
        # For this test, we'll manually check descriptor calculation
        mol = Chem.MolFromSmiles(sample_smiles[0])
        descriptors = featurizer._get_descriptors(mol)
        
        assert isinstance(descriptors, dict)
        assert "MolWt" in descriptors
        assert "LogP" in descriptors
        assert "NumHDonors" in descriptors
        assert "NumHAcceptors" in descriptors
        assert len(descriptors) >= 20  # Should have many descriptors
    
    def test_fragments(self, sample_smiles):
        """Test fragment calculation."""
        featurizer = MolecularFeaturizer(include_fragments=True)
        mol = Chem.MolFromSmiles(sample_smiles[0])
        fragments = featurizer._get_fragments(mol)
        
        assert isinstance(fragments, dict)
        assert len(fragments) > 0
        assert all(isinstance(v, int) for v in fragments.values())


class TestDescriptorCalculator:
    """Test DescriptorCalculator class."""
    
    def test_lipinski_descriptors(self, sample_smiles):
        """Test Lipinski descriptor calculation."""
        mol = Chem.MolFromSmiles(sample_smiles[0])
        descriptors = DescriptorCalculator.lipinski_descriptors(mol)
        
        expected_keys = ["MW", "LogP", "HBD", "HBA", "TPSA"]
        assert all(key in descriptors for key in expected_keys)
        assert all(isinstance(v, (int, float)) for v in descriptors.values())
    
    def test_drug_like_descriptors(self, sample_smiles):
        """Test drug-like descriptor calculation."""
        mol = Chem.MolFromSmiles(sample_smiles[0])
        descriptors = DescriptorCalculator.drug_like_descriptors(mol)
        
        expected_keys = ["MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds", "AromaticRings"]
        assert all(key in descriptors for key in expected_keys)
    
    def test_lipinski_compliance(self, sample_smiles):
        """Test Lipinski rule compliance checking."""
        mol = Chem.MolFromSmiles(sample_smiles[0])  # Ethanol
        compliance = DescriptorCalculator.check_lipinski(mol)
        
        expected_keys = ["MW_ok", "LogP_ok", "HBD_ok", "HBA_ok", "lipinski_compliant"]
        assert all(key in compliance for key in expected_keys)
        assert all(isinstance(v, bool) for v in compliance.values())
        
        # Ethanol should be Lipinski compliant
        assert compliance["lipinski_compliant"] is True


class TestBatchFeaturize:
    """Test batch featurization function."""
    
    def test_batch_featurize_default(self, sample_smiles):
        """Test batch featurization with default featurizer."""
        features, valid_smiles = batch_featurize(sample_smiles)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == len(valid_smiles)
        assert len(valid_smiles) <= len(sample_smiles)
    
    def test_batch_featurize_custom(self, sample_smiles):
        """Test batch featurization with custom featurizer."""
        featurizer = MolecularFeaturizer(
            fingerprint_type="rdkit",
            fingerprint_bits=512,
            include_descriptors=False
        )
        
        features, valid_smiles = batch_featurize(sample_smiles, featurizer=featurizer)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[1] == 512  # Only fingerprint, no descriptors
    
    def test_batch_featurize_small_batch(self, sample_smiles):
        """Test batch featurization with small batch size."""
        features, valid_smiles = batch_featurize(
            sample_smiles, 
            batch_size=3
        )
        
        assert isinstance(features, np.ndarray)
        assert len(features) == len(valid_smiles)
    
    def test_batch_featurize_empty_input(self):
        """Test batch featurization with empty input."""
        features, valid_smiles = batch_featurize([])
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 0
        assert len(valid_smiles) == 0
    
    def test_batch_featurize_mixed_validity(self, sample_smiles, invalid_smiles):
        """Test batch featurization with mixed valid/invalid SMILES."""
        mixed_smiles = sample_smiles + invalid_smiles
        features, valid_smiles = batch_featurize(mixed_smiles)
        
        assert len(valid_smiles) == len(sample_smiles)  # Only valid ones should remain
        assert len(features) == len(valid_smiles)
        assert all(smi in sample_smiles for smi in valid_smiles)


@pytest.mark.parametrize("fingerprint_type", ["morgan", "rdkit", "maccs"])
def test_fingerprint_consistency(fingerprint_type, sample_smiles):
    """Test that fingerprints are consistent across multiple calls."""
    featurizer = MolecularFeaturizer(fingerprint_type=fingerprint_type)
    test_smiles = sample_smiles[0]
    
    # Featurize the same molecule multiple times
    features1 = featurizer.featurize_molecule(test_smiles)
    features2 = featurizer.featurize_molecule(test_smiles)
    
    assert features1 is not None
    assert features2 is not None
    np.testing.assert_array_equal(features1, features2)


def test_featurizer_memory_efficiency():
    """Test that featurizer doesn't consume excessive memory."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Generate many features
    featurizer = MolecularFeaturizer()
    large_smiles_list = ["CCO"] * 1000  # 1000 copies of ethanol
    
    features, valid_smiles = featurizer.featurize_molecules(large_smiles_list)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (less than 100MB for this test)
    assert memory_increase < 100 * 1024 * 1024  # 100MB
    assert len(features) == 1000
    assert len(valid_smiles) == 1000
