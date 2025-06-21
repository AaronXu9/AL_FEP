#!/usr/bin/env python3
"""
Test real FEP calculations with properly prepared protein
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_real_fep_with_prepared_protein():
    """Test FEP oracle with the prepared protein structure."""
    print("🧪 Real FEP Test with Prepared Protein")
    print("=" * 45)
    
    from al_fep.oracles.fep_oracle import FEPOracle
    
    # Configuration for real FEP
    config = {
        "fep": {
            "mock_mode": False,
            "force_field": "amber14",
            "water_model": "tip3p",
            "num_lambda_windows": 3,        # Very few for testing
            "simulation_time": 0.001,       # Very short (1 ps)
            "temperature": 298.15,
            "pressure": 1.0,
            "equilibration_steps": 10,      # Minimal equilibration
            "production_steps": 50,         # Minimal production
        }
    }
    
    # Use the prepared protein file
    prepared_protein = "data/targets/7jvr/7jvr_system_prepared.pdb"
    
    if not os.path.exists(prepared_protein):
        print(f"❌ Prepared protein not found: {prepared_protein}")
        print("Run prepare_protein_advanced.py first!")
        return
    
    print(f"✅ Using prepared protein: {prepared_protein}")
    
    try:
        # Create oracle with prepared protein
        oracle = FEPOracle(
            target="7jvr", 
            receptor_file=prepared_protein,
            config=config
        )
        
        print(f"✅ FEP Oracle created:")
        print(f"  Target: {oracle.target}")
        print(f"  Receptor: {oracle.receptor_file}")
        print(f"  Mock mode: {oracle.mock_mode}")
        print(f"  Lambda windows: {oracle.num_lambda_windows}")
        print(f"  Simulation time: {oracle.simulation_time} ns")
        
        # Test with ethanol (simple molecule)
        test_smiles = "CCO"
        print(f"\n🔬 Testing Real FEP with {test_smiles}")
        print("⚠️  This performs actual MD simulations...")
        
        start_time = time.time()
        result = oracle.evaluate(test_smiles)
        end_time = time.time()
        
        print(f"⏱️  Calculation completed in {end_time - start_time:.1f} seconds")
        
        # Check results
        if isinstance(result, dict):
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
                return False
            else:
                print(f"🎉 Real FEP calculation succeeded!")
                print(f"  Score: {result.get('score'):.3f}")
                print(f"  FEP Score: {result.get('fep_score'):.3f} kcal/mol")
                print(f"  Binding Free Energy: {result.get('binding_free_energy'):.3f} kcal/mol")
                print(f"  Method: {result.get('method')}")
                print(f"  Force field: {result.get('force_field')}")
                print(f"  Simulation time: {result.get('simulation_time')} ns")
                return True
        else:
            print(f"❌ Unexpected result type: {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_molecules():
    """Test FEP with multiple molecules."""
    print("\n🧪 Testing Multiple Molecules")
    print("=" * 35)
    
    from al_fep.oracles.fep_oracle import FEPOracle
    
    # Very minimal config for speed
    config = {
        "fep": {
            "mock_mode": False,
            "num_lambda_windows": 2,  # Just 2 windows for speed
            "simulation_time": 0.0005,  # 0.5 ps
            "equilibration_steps": 5,
            "production_steps": 10,
        }
    }
    
    prepared_protein = "data/targets/7jvr/7jvr_system_prepared.pdb"
    oracle = FEPOracle(target="7jvr", receptor_file=prepared_protein, config=config)
    
    # Test molecules of increasing complexity
    test_molecules = [
        ("Water", "O"),
        ("Methanol", "CO"),
        ("Ethanol", "CCO"),
    ]
    
    results = []
    for name, smiles in test_molecules:
        print(f"\n🔬 Testing {name}: {smiles}")
        
        start_time = time.time()
        result = oracle.evaluate(smiles)
        end_time = time.time()
        
        if result.get('error'):
            print(f"   ❌ Error: {result['error']}")
            results.append((name, "Error", end_time - start_time))
        else:
            fep_score = result.get('fep_score', 0.0)
            print(f"   ✅ Success: {fep_score:.3f} kcal/mol ({end_time - start_time:.1f}s)")
            results.append((name, "Success", end_time - start_time))
    
    # Summary
    print(f"\n📊 Summary:")
    successful = sum(1 for _, status, _ in results if status == "Success")
    print(f"Successful calculations: {successful}/{len(test_molecules)}")
    
    for name, status, time_taken in results:
        print(f"  {name:<10}: {status:<7} ({time_taken:.1f}s)")

def main():
    """Main testing function."""
    print("🎯 Real FEP Testing with Prepared Protein")
    print("=" * 50)
    
    # Test single molecule first
    success = test_real_fep_with_prepared_protein()
    
    if success:
        print("\n" + "="*50)
        # If first test passes, try multiple molecules
        test_multiple_molecules()
    
    print(f"\n🏁 Testing complete!")

if __name__ == "__main__":
    main()
