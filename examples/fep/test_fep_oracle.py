#!/usr/bin/env python3
"""
Comprehensive test for the FEP Oracle
"""

import os
import sys
import time
import tempfile
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from al_fep.utils.config import load_config
from al_fep.oracles.fep_oracle import FEPOracle


def test_fep_oracle_mock_mode():
    """Test the FEP oracle in mock mode."""
    print("🧪 Testing FEP Oracle (Mock Mode)")
    print("=" * 50)
    
    # Create config with mock mode enabled
    config = {
        "fep": {
            "mock_mode": True,
            "force_field": "amber14",
            "water_model": "tip3p", 
            "num_lambda_windows": 12,
            "simulation_time": 5.0,
            "temperature": 298.15,
            "pressure": 1.0
        }
    }
    
    # Initialize oracle
    oracle = FEPOracle(target="test", config=config)
    
    print(f"✓ Oracle initialized: {oracle}")
    print(f"✓ Mock mode: {oracle.mock_mode}")
    print(f"✓ Force field: {oracle.force_field}")
    print(f"✓ Simulation time: {oracle.simulation_time} ns")
    
    # Test molecules with varying properties
    test_molecules = [
        ("Ethanol", "CCO"),
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Propranolol", "CC(C)NCC(COC1=CC=CC2=C1C=CN2)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Invalid SMILES", "INVALID_SMILES"),
    ]
    
    results = []
    for name, smiles in test_molecules:
        print(f"\n🔬 Testing {name}: {smiles}")
        start_time = time.time()
        
        try:
            result = oracle.evaluate(smiles)
            end_time = time.time()
            
            if result.get('error'):
                print(f"   ❌ Error: {result['error']}")
            else:
                score = result.get('score', 'N/A')
                fep_score = result.get('fep_score', 'N/A')
                binding_energy = result.get('binding_free_energy', 'N/A')
                method = result.get('method', 'N/A')
                
                print(f"   ✅ Success!")
                print(f"   - Score: {score:.3f}" if isinstance(score, (int, float)) else f"   - Score: {score}")
                print(f"   - FEP Score: {fep_score:.3f} kcal/mol" if isinstance(fep_score, (int, float)) else f"   - FEP Score: {fep_score}")
                print(f"   - Binding Free Energy: {binding_energy:.3f} kcal/mol" if isinstance(binding_energy, (int, float)) else f"   - Binding Free Energy: {binding_energy}")
                print(f"   - Method: {method}")
                print(f"   - Time: {end_time - start_time:.3f}s")
                
            results.append((name, result, end_time - start_time))
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append((name, {"error": str(e)}, 0))
    
    return results


def test_fep_oracle_ligand_preparation():
    """Test ligand preparation functionality."""
    print("\n🧪 Testing FEP Oracle Ligand Preparation")
    print("=" * 50)
    
    # Create oracle in mock mode
    config = {"fep": {"mock_mode": True}}
    oracle = FEPOracle(target="test", config=config)
    
    # Test molecules
    test_molecules = [
        ("Ethanol", "CCO"),
        ("Benzene", "C1=CC=CC=C1"),
        ("Water", "O"),
        ("Methane", "C"),
        ("Invalid", "INVALID"),
    ]
    
    print("Testing ligand preparation:")
    for name, smiles in test_molecules:
        print(f"\n🔬 Preparing {name}: {smiles}")
        
        try:
            mol = oracle._prepare_ligand(smiles)
            
            if mol is None:
                print(f"   ❌ Failed to prepare ligand")
            else:
                num_atoms = mol.GetNumAtoms()
                num_conformers = mol.GetNumConformers()
                print(f"   ✅ Success!")
                print(f"   - Atoms: {num_atoms}")
                print(f"   - Conformers: {num_conformers}")
                
                # Check if 3D coordinates exist
                if num_conformers > 0:
                    conformer = mol.GetConformer(0)
                    pos = conformer.GetAtomPosition(0)
                    print(f"   - First atom position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")


def test_fep_oracle_mock_calculations():
    """Test mock FEP calculations and scoring consistency."""
    print("\n🧪 Testing FEP Mock Calculations")
    print("=" * 50)
    
    config = {"fep": {"mock_mode": True}}
    oracle = FEPOracle(target="test", config=config)
    
    # Test scoring consistency
    test_smiles = "CCO"  # Ethanol
    scores = []
    
    print(f"Testing scoring consistency for ethanol ({test_smiles}):")
    print("Running 5 evaluations to check consistency...")
    
    for i in range(5):
        result = oracle.evaluate(test_smiles)
        if not result.get('error'):
            score = result.get('fep_score')
            scores.append(score)
            print(f"   Run {i+1}: {score:.3f} kcal/mol")
        else:
            print(f"   Run {i+1}: Error - {result['error']}")
    
    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"\n📊 Statistics:")
        print(f"   - Mean: {mean_score:.3f} kcal/mol")
        print(f"   - Std Dev: {std_score:.3f} kcal/mol")
        print(f"   - Range: [{min(scores):.3f}, {max(scores):.3f}] kcal/mol")
    
    # Test different molecular properties
    print(f"\n🔬 Testing different molecular types:")
    diverse_molecules = [
        ("Small polar", "O"),  # Water
        ("Small nonpolar", "C"),  # Methane
        ("Medium drug-like", "CC(=O)OC1=CC=CC=C1C(=O)O"),  # Aspirin
        ("Large drug-like", "CC(C)NCC(COC1=CC=CC2=C1C=CN2)O"),  # Propranolol
        ("Very hydrophobic", "CCCCCCCCCCCCCCCC"),  # Long alkyl chain
    ]
    
    for name, smiles in diverse_molecules:
        result = oracle.evaluate(smiles)
        if not result.get('error'):
            fep_score = result.get('fep_score')
            print(f"   {name:<20}: {fep_score:.3f} kcal/mol")
        else:
            print(f"   {name:<20}: Error")


def test_fep_oracle_configuration():
    """Test FEP oracle configuration options."""
    print("\n🧪 Testing FEP Oracle Configuration")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        {
            "name": "Default Config",
            "config": {"fep": {"mock_mode": True}}
        },
        {
            "name": "Custom Force Field",
            "config": {
                "fep": {
                    "mock_mode": True,
                    "force_field": "charmm36",
                    "water_model": "tip4p"
                }
            }
        },
        {
            "name": "Long Simulation",
            "config": {
                "fep": {
                    "mock_mode": True,
                    "simulation_time": 20.0,
                    "num_lambda_windows": 20
                }
            }
        },
        {
            "name": "High Temperature", 
            "config": {
                "fep": {
                    "mock_mode": True,
                    "temperature": 350.0,
                    "pressure": 2.0
                }
            }
        }
    ]
    
    test_smiles = "CCO"
    
    for config_test in configs:
        print(f"\n🔧 Testing {config_test['name']}:")
        
        try:
            oracle = FEPOracle(target="test", config=config_test['config'])
            
            print(f"   ✓ Force field: {oracle.force_field}")
            print(f"   ✓ Water model: {oracle.water_model}")
            print(f"   ✓ Simulation time: {oracle.simulation_time} ns")
            print(f"   ✓ Temperature: {oracle.temperature} K")
            print(f"   ✓ Lambda windows: {oracle.num_lambda_windows}")
            
            # Test evaluation
            result = oracle.evaluate(test_smiles)
            if not result.get('error'):
                print(f"   ✓ Evaluation successful: {result['fep_score']:.3f} kcal/mol")
            else:
                print(f"   ❌ Evaluation failed: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")


def test_fep_oracle_caching():
    """Test caching functionality from BaseOracle."""
    print("\n🧪 Testing FEP Oracle Caching")
    print("=" * 50)
    
    config = {"fep": {"mock_mode": True}}
    oracle = FEPOracle(target="test", config=config)
    
    test_smiles = "CCO"
    
    print(f"Testing caching with {test_smiles}:")
    
    # First evaluation
    print("   First evaluation...")
    start_time = time.time()
    result1 = oracle.evaluate(test_smiles)
    time1 = time.time() - start_time
    
    # Second evaluation (should be cached)
    print("   Second evaluation (should be cached)...")
    start_time = time.time()
    result2 = oracle.evaluate(test_smiles)
    time2 = time.time() - start_time
    
    print(f"   First call time: {time1:.4f}s")
    print(f"   Second call time: {time2:.4f}s")
    
    if time2 < time1 * 0.1:  # Should be much faster
        print("   ✅ Caching working correctly!")
    else:
        print("   ⚠️  Caching may not be working as expected")
    
    # Check if results are identical
    if result1 == result2:
        print("   ✅ Cached results are identical")
    else:
        print("   ❌ Cached results differ")
    
    # Check cache stats
    stats = oracle.get_statistics()
    print(f"   Cache stats: {stats}")


def main():
    """Main function to run all FEP oracle tests."""
    print("🎯 Testing FEP Oracle")
    print("=" * 60)
    
    try:
        # Test mock mode
        mock_results = test_fep_oracle_mock_mode()
        successful_mock = sum(1 for _, result, _ in mock_results if not result.get('error'))
        print(f"\n✅ Mock Mode: {successful_mock}/{len(mock_results)} successful")
        
        # Test ligand preparation
        test_fep_oracle_ligand_preparation()
        
        # Test mock calculations
        test_fep_oracle_mock_calculations()
        
        # Test configuration
        test_fep_oracle_configuration()
        
        # Test caching
        test_fep_oracle_caching()
        
        print(f"\n🎉 FEP Oracle testing completed!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
