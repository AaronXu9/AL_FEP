#!/usr/bin/env python3
"""
Test real FEP calculations
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_real_fep_oracle():
    """Test the real FEP oracle implementation."""
    print("🧪 Testing Real FEP Oracle")
    print("=" * 40)
    
    from al_fep.oracles.fep_oracle import FEPOracle
    
    # Test configuration for real FEP (not mock mode)
    config = {
        "fep": {
            "mock_mode": False,  # Enable real FEP calculations
            "force_field": "amber14",
            "water_model": "tip3p",
            "num_lambda_windows": 5,  # Reduced for testing
            "simulation_time": 0.1,   # Very short for testing (100 ps)
            "temperature": 298.15,
            "pressure": 1.0
        }
    }
    
    try:
        # Test OpenMM availability first
        print("Checking OpenMM installation...")
        import openmm
        print(f"✓ OpenMM version {openmm.version.version} found")
        
        # Create oracle
        print("Creating FEP oracle...")
        oracle = FEPOracle(target="test", config=config)
        print(f"✓ Oracle created: {oracle}")
        print(f"✓ Real FEP mode: {not oracle.mock_mode}")
        print(f"✓ Lambda windows: {oracle.num_lambda_windows}")
        print(f"✓ Simulation time: {oracle.simulation_time} ns")
        
        # Test with a simple molecule
        test_smiles = "CCO"  # Ethanol
        print(f"\n🔬 Testing real FEP calculation with {test_smiles}...")
        print("⚠️  Note: This will attempt a real MD simulation (may take time)")
        
        # This will likely fail due to missing receptor file, but tests the pipeline
        result = oracle.evaluate(test_smiles)
        
        print(f"✓ Evaluation completed")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, dict):
            if result.get('error'):
                print(f"❌ Expected error (missing receptor): {result['error']}")
                if "Receptor file not found" in result['error']:
                    print("✅ Error handling working correctly - missing receptor file")
                else:
                    print("❌ Unexpected error type")
            else:
                print(f"✅ Real FEP calculation succeeded!")
                print(f"  Score: {result.get('score')}")
                print(f"  FEP Score: {result.get('fep_score')} kcal/mol")
                print(f"  Method: {result.get('method')}")
                print(f"  Force field: {result.get('force_field')}")
                print(f"  Lambda windows: {result.get('num_lambda_windows')}")
        
        print("\n🎉 Real FEP Oracle test completed!")
        
    except ImportError as e:
        print(f"❌ OpenMM not available: {e}")
        print("Install with: conda install -c conda-forge openmm")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_mock_vs_real_comparison():
    """Compare mock vs real FEP oracle setup."""
    print("\n🔄 Mock vs Real FEP Comparison")
    print("=" * 40)
    
    from al_fep.oracles.fep_oracle import FEPOracle
    
    # Mock oracle
    mock_config = {"fep": {"mock_mode": True}}
    mock_oracle = FEPOracle(target="test", config=mock_config)
    
    # Real oracle
    real_config = {"fep": {"mock_mode": False, "simulation_time": 0.1}}
    
    try:
        real_oracle = FEPOracle(target="test", config=real_config)
        
        print("Oracle Comparison:")
        print(f"  Mock Oracle - Mode: {mock_oracle.mock_mode}")
        print(f"  Real Oracle - Mode: {real_oracle.mock_mode}")
        print(f"  Force field: {real_oracle.force_field}")
        print(f"  Simulation time: {real_oracle.simulation_time} ns")
        print(f"  Lambda windows: {real_oracle.num_lambda_windows}")
        
        # Test both with same molecule
        test_smiles = "CCO"
        
        print(f"\nTesting both oracles with {test_smiles}:")
        
        # Mock result
        mock_result = mock_oracle.evaluate(test_smiles)
        print(f"  Mock result: {mock_result.get('fep_score', 'Error')}")
        
        # Real result (will likely error due to missing files)
        real_result = real_oracle.evaluate(test_smiles)
        if real_result.get('error'):
            print(f"  Real result: Error (expected)")
        else:
            print(f"  Real result: {real_result.get('fep_score', 'Error')}")
            
        print("✅ Comparison completed!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    test_real_fep_oracle()
    test_mock_vs_real_comparison()
