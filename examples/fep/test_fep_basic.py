#!/usr/bin/env python3
"""
Basic FEP Oracle test
"""

import sys
import os

# Add src to path (go up two levels to reach project root, then to src)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_fep_oracle_basic():
    """Test basic FEP oracle functionality."""
    print("🧪 Basic FEP Oracle Test")
    print("=" * 30)
    
    try:
        # Import
        from al_fep.oracles.fep_oracle import FEPOracle
        print("✓ Import successful")
        
        # Create config
        config = {
            "fep": {
                "mock_mode": True,
                "force_field": "amber14",
                "simulation_time": 5.0
            }
        }
        print("✓ Config created")
        
        # Create oracle
        oracle = FEPOracle(target="test", config=config)
        print(f"✓ Oracle created: {oracle}")
        print(f"  Mock mode: {oracle.mock_mode}")
        print(f"  Force field: {oracle.force_field}")
        
        # Test evaluation
        test_smiles = "CCO"  # Ethanol
        print(f"\nTesting evaluation with {test_smiles}...")
        
        result = oracle.evaluate(test_smiles)
        print(f"✓ Evaluation completed")
        print(f"  Result type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"  Keys: {list(result.keys())}")
            if 'error' in result and result['error']:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  ✅ Success!")
                print(f"    Score: {result.get('score')}")
                print(f"    FEP Score: {result.get('fep_score')}")
                print(f"    Method: {result.get('method')}")
        else:
            print(f"  Unexpected result: {result}")
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fep_oracle_basic()
