#!/usr/bin/env python3
"""
Simple test for the FEP Oracle
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from al_fep.oracles.fep_oracle import FEPOracle

def simple_test():
    """Simple test of FEP oracle."""
    print("🧪 Simple FEP Oracle Test")
    print("=" * 30)
    
    # Create config with mock mode enabled
    config = {
        "fep": {
            "mock_mode": True,
            "force_field": "amber14",
        }
    }
    
    # Initialize oracle
    print("Creating FEP oracle...")
    oracle = FEPOracle(target="test", config=config)
    print(f"✓ Oracle created: {oracle}")
    print(f"✓ Mock mode: {oracle.mock_mode}")
    
    # Test simple molecule
    test_smiles = "CCO"  # Ethanol
    print(f"\nTesting molecule: {test_smiles}")
    
    try:
        result = oracle.evaluate(test_smiles)
        print(f"✓ Evaluation completed")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, dict):
            print("Result keys:", list(result.keys()))
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Success!")
                print(f"  Score: {result.get('score')}")
                print(f"  FEP Score: {result.get('fep_score')}")
                print(f"  Method: {result.get('method')}")
        else:
            print(f"Unexpected result type: {type(result)}")
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
