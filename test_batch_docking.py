#!/usr/bin/env python3
"""
Test script for batch docking functionality in DockingOracle.
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


from al_fep.oracles.docking_oracle import DockingOracle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_batch_vs_single_docking():
    """
    Test batch docking vs single docking to verify functionality and performance.
    """
    print("=== Testing Batch vs Single Docking ===")
    
    # Test molecules (drug-like SMILES)
    test_smiles = [
        "CCO",  # Ethanol (simple)
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
    ]
    
    # Configuration for mock testing (so we don't need actual docking software)
    config = {
        "docking": {
            "engine": "gnina",  
            "mock_mode": False,  # Enable mock mode for testing
            "center_x": 0.0,
            "center_y": 0.0, 
            "center_z": 0.0,
            "size_x": 20.0,
            "size_y": 20.0,
            "size_z": 20.0,
            "exhaustiveness": 8,
            "num_poses": 3,
            "receptor_file": "./data/test_oracle.pdb",
            "gnina_path": "/home/aoxu/projects/PoseBench/forks/GNINA/gnina",  # Path to GNINA executable
        }
    }
    
    # Initialize oracle
    oracle = DockingOracle(target="test", config=config)
    
    print(f"Oracle supports batch processing: {oracle.supports_batch_processing()}")
    
    # Test single evaluation
    print("\n--- Single Evaluation Test ---")
    start_time = time.time()
    single_results = []
    for smiles in test_smiles:
        result = oracle.evaluate(smiles)  # This returns a single dict
        single_results.append(result)
        print(f"SMILES: {smiles[:20]:<20} CNNscore: {result.get('CNNscore', 'N/A')}")
    single_time = time.time() - start_time
    print(f"Single evaluation time: {single_time:.2f}s")
    
    # Test batch evaluation
    print("\n--- Batch Evaluation Test ---")
    start_time = time.time()
    batch_results = oracle.evaluate(test_smiles)  # This returns a list of dicts
    batch_time = time.time() - start_time
    print(f"Batch evaluation time: {batch_time:.2f}s")
    
    # Compare results
    print("\n--- Results Comparison ---")
    for i, (single, batch, smiles) in enumerate(zip(single_results, batch_results, test_smiles)):
        single_score = single.get('CNNscore')
        batch_score = batch.get('CNNscore')
        match = "✓" if single_score == batch_score else "✗"
        print(f"{i+1}. {match} Single: {single_score}, Batch: {batch_score} ({smiles[:30]})")
    
    # Performance comparison
    if batch_time > 0:
        speedup = single_time / batch_time
        print(f"\nSpeedup: {speedup:.2f}x")
    
    # Test oracle statistics
    print("\n--- Oracle Statistics ---")
    stats = oracle.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

def test_error_handling():
    """
    Test error handling in batch processing.
    """
    print("\n\n=== Testing Error Handling ===")
    
    # Test with invalid SMILES
    invalid_smiles = [
        "CCO",  # Valid
        "INVALID_SMILES",  # Invalid
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Valid
        "",  # Empty
        "XYZ123",  # Invalid
    ]
    
    config = {
        "docking": {
            "engine": "gnina",
            "mock_mode": False,
            "center_x": 0.0,
            "center_y": 0.0, 
            "center_z": 0.0,
            "size_x": 20.0,
            "size_y": 20.0,
            "size_z": 20.0,
            "receptor_file": "./data/test_oracle.pdb",
            "gnina_path": "/home/aoxu/projects/PoseBench/forks/GNINA/gnina",
        }
    }

    oracle = DockingOracle(target="test", receptor_file="./data/test_oracle.pdb", config=config)

    print("Testing batch evaluation with mixed valid/invalid SMILES:")
    results = oracle.evaluate(invalid_smiles)  # This returns a list of dicts
    
    for i, (smiles, result) in enumerate(zip(invalid_smiles, results)):
        error = result.get('error')
        score = result.get('score')
        status = "ERROR" if error else "OK"
        print(f"{i+1}. [{status:5}] {smiles[:20]:<20} Score: {score} Error: {error}")

def test_different_engines():
    """
    Test both Vina and GNINA engines in mock mode.
    """
    print("\n\n=== Testing Different Engines ===")
    
    test_smiles = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O"]  # Ethanol and Aspirin
    
    engines = ["vina", "gnina"]
    
    for engine in engines:
        print(f"\n--- Testing {engine.upper()} Engine ---")
        
        config = {
            "docking": {
                "engine": engine,
                "mock_mode": True,
                "center_x": 0.0,
                "center_y": 0.0, 
                "center_z": 0.0,
                "size_x": 20.0,
                "size_y": 20.0,
                "size_z": 20.0,
                "receptor_file": "./data/test_oracle.pdb",
                "gnina_path": "/home/aoxu/projects/PoseBench/forks/GNINA/gnina" if engine == "gnina" else None,
            }
        }
        
        oracle = DockingOracle(target="test", config=config)
        results = oracle.evaluate(test_smiles)  # This returns a list of dicts
        
        for smiles, result in zip(test_smiles, results):
            score = result.get('score')
            method = result.get('method')
            error = result.get('error')
            print(f"SMILES: {smiles:<30} Score: {score} Method: {method} Error: {error}")

if __name__ == "__main__":
    try:
        test_batch_vs_single_docking()
        test_error_handling()
        test_different_engines()
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
