#!/usr/bin/env python3
"""
Test script for SDF file docking functionality in DockingOracle.
Demonstrates how to dock SDF files directly instead of SMILES strings.
"""

import os
import sys
import tempfile
import logging
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from al_fep.oracles.docking_oracle import DockingOracle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_sdf_file(smiles_list: List[str], output_path: str) -> bool:
    """Create a test SDF file from SMILES strings."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        writer = Chem.SDWriter(output_path)
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Could not create molecule from SMILES: {smiles}")
                continue
            
            # Add hydrogens and generate 3D coordinates
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol) == 0:
                AllChem.MMFFOptimizeMolecule(mol)
            
            # Set molecule name
            mol.SetProp("_Name", f"molecule_{i+1}")
            mol.SetProp("SMILES", smiles)
            
            writer.write(mol)
        
        writer.close()
        return True
        
    except ImportError:
        logger.error("RDKit not available - cannot create SDF file")
        return False
    except Exception as e:
        logger.error(f"Error creating SDF file: {e}")
        return False

def test_sdf_vs_smiles_docking():
    """Test docking with SDF files vs SMILES strings."""
    print("=== Testing SDF File vs SMILES Docking ===")
    
    # Initialize oracle in mock mode for testing
    oracle = DockingOracle(
        target="test",
        config={
            "docking": {
                "engine": "vina",
                "mock_mode": True,
                "receptor_file": "data/targets/test/test_prepared.pdbqt"
            }
        }
    )
    
    # Test molecules
    test_smiles = [
        "CCO",  # Ethanol
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    ]
    
    # Create temporary SDF file
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp_file:
        sdf_path = tmp_file.name
    
    try:
        # Create SDF file
        if not create_test_sdf_file(test_smiles, sdf_path):
            print("❌ Could not create test SDF file")
            return
        
        print(f"✅ Created test SDF file: {sdf_path}")
        
        # Test 1: Individual SMILES evaluation
        print("\n--- Individual SMILES Evaluation ---")
        smiles_results = []
        for smiles in test_smiles:
            result = oracle.evaluate(smiles)
            smiles_results.append(result)
            print(f"SMILES: {smiles[:30]:<30} Score: {result['score']}")
        
        # Test 2: SDF file evaluation
        print("\n--- SDF File Evaluation ---")
        sdf_result = oracle.dock_sdf_file(sdf_path)
        print(f"SDF File: {os.path.basename(sdf_path):<30} Score: {sdf_result['score']}")
        print(f"Extracted SMILES: {sdf_result.get('smiles', 'N/A')}")
        
        # Test 3: Batch SMILES evaluation
        print("\n--- Batch SMILES Evaluation ---")
        batch_smiles_results = oracle.evaluate(test_smiles)
        for i, result in enumerate(batch_smiles_results):
            print(f"Batch[{i}]: {test_smiles[i][:25]:<25} Score: {result['score']}")
        
        # Test 4: Multiple SDF files (create individual files)
        print("\n--- Multiple SDF Files Evaluation ---")
        sdf_files = []
        for i, smiles in enumerate(test_smiles[:2]):  # Just first 2 for demo
            individual_sdf = tempfile.NamedTemporaryFile(suffix=f"_mol{i}.sdf", delete=False)
            create_test_sdf_file([smiles], individual_sdf.name)
            sdf_files.append(individual_sdf.name)
        
        try:
            batch_sdf_results = oracle.dock_sdf_files(sdf_files)
            for i, result in enumerate(batch_sdf_results):
                print(f"SDF[{i}]: {os.path.basename(sdf_files[i]):<25} Score: {result['score']}")
        finally:
            # Clean up individual SDF files
            for sdf_file in sdf_files:
                try:
                    os.unlink(sdf_file)
                except:
                    pass
        
        print("\n--- Oracle Statistics ---")
        stats = oracle.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    finally:
        # Clean up main SDF file
        try:
            os.unlink(sdf_path)
        except:
            pass

def test_sdf_error_handling():
    """Test error handling with invalid SDF files."""
    print("\n=== Testing SDF Error Handling ===")
    
    oracle = DockingOracle(
        target="test",
        config={
            "docking": {
                "engine": "vina",
                "mock_mode": True
            }
        }
    )
    
    # Test 1: Non-existent SDF file
    print("\n--- Non-existent SDF File ---")
    result = oracle.evaluate("non_existent_file.sdf")
    print(f"Result: Score={result['score']}, Error={result['error']}")
    
    # Test 2: Invalid SDF file (create empty file)
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp_file:
        tmp_file.write(b"invalid sdf content")
        invalid_sdf = tmp_file.name
    
    try:
        print("\n--- Invalid SDF File ---")
        result = oracle.evaluate(invalid_sdf)
        print(f"Result: Score={result['score']}, Error={result['error']}")
    finally:
        try:
            os.unlink(invalid_sdf)
        except:
            pass
    
    # Test 3: Mix of valid SMILES and SDF files
    print("\n--- Mixed Input Types ---")
    mixed_inputs = [
        "CCO",  # Valid SMILES
        "non_existent.sdf",  # Invalid SDF path
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Valid SMILES
    ]
    
    results = oracle.evaluate(mixed_inputs)
    for i, result in enumerate(results):
        input_type = "SDF" if mixed_inputs[i].endswith('.sdf') else "SMILES"
        print(f"{input_type}: {mixed_inputs[i][:30]:<30} Score: {result['score']} Error: {result.get('error', 'None')}")

def main():
    """Run all SDF docking tests."""
    print("🧪 Testing SDF File Docking Functionality")
    print("=" * 50)
    
    try:
        test_sdf_vs_smiles_docking()
        test_sdf_error_handling()
        print("\n🎉 All SDF docking tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
