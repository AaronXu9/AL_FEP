#!/usr/bin/env python3
"""
Real FEP Test with Prepared Data
"""

import os
import sys
import json
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def load_test_config():
    """Load the prepared test configuration."""
    config_file = "fep_test_config.json"
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        print("Run prepare_fep_test.py first")
        return None
    
    with open(config_file, 'r') as f:
        return json.load(f)

def test_real_fep_with_7jvr():
    """Test real FEP calculations with 7JVR protein."""
    print("🧪 Real FEP Test with 7JVR")
    print("=" * 40)
    
    # Load configuration
    config = load_test_config()
    if config is None:
        return False
    
    print(f"Configuration loaded:")
    print(f"  Lambda windows: {config['fep']['num_lambda_windows']}")
    print(f"  Simulation time: {config['fep']['simulation_time']} ns")
    print(f"  Production steps: {config['fep']['production_steps']}")
    
    # Check if protein file exists
    protein_file = "data/targets/7jvr/7jvr_system.pdb"
    if not os.path.exists(protein_file):
        print(f"❌ Protein file not found: {protein_file}")
        print("Run prepare_fep_test.py first")
        return False
    
    print(f"✅ Protein file found: {protein_file}")
    
    try:
        # Import FEP oracle
        from al_fep.oracles.fep_oracle import FEPOracle
        
        # Create oracle with 7JVR target and test configuration
        print("\nCreating FEP Oracle...")
        oracle = FEPOracle(
            target="7jvr", 
            receptor_file=protein_file,
            config=config
        )
        
        print(f"✅ FEP Oracle created")
        print(f"  Target: {oracle.target}")
        print(f"  Receptor: {oracle.receptor_file}")
        print(f"  Mock mode: {oracle.mock_mode}")
        print(f"  Force field: {oracle.force_field}")
        
        # Test molecules (start simple)
        test_molecules = [
            ("Ethanol", "CCO"),
            # ("Methanol", "CO"),  # Add more if first works
            # ("Propanol", "CCCO"),
        ]
        
        results = []
        
        for name, smiles in test_molecules:
            print(f"\n🔬 Testing {name}: {smiles}")
            print("⚠️  This will run actual MD simulations (may take several minutes)")
            
            start_time = time.time()
            
            try:
                # Run real FEP calculation
                result = oracle.evaluate(smiles)
                end_time = time.time()
                
                elapsed = end_time - start_time
                print(f"⏱️  Calculation completed in {elapsed:.1f} seconds")
                
                if result.get('error'):
                    print(f"❌ Error: {result['error']}")
                    results.append((name, "Error", result['error'], elapsed))
                else:
                    fep_score = result.get('fep_score')
                    binding_energy = result.get('binding_free_energy')
                    score = result.get('score')
                    
                    print(f"✅ Success!")
                    print(f"  FEP Score: {fep_score:.3f} kcal/mol")
                    print(f"  Binding Free Energy: {binding_energy:.3f} kcal/mol")
                    print(f"  Normalized Score: {score:.3f}")
                    print(f"  Method: {result.get('method')}")
                    
                    results.append((name, "Success", fep_score, elapsed))
                
            except Exception as e:
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"❌ Exception after {elapsed:.1f}s: {e}")
                results.append((name, "Exception", str(e), elapsed))
                
                # Print detailed error for debugging
                import traceback
                print("Detailed error:")
                traceback.print_exc()
        
        # Summary
        print(f"\n📊 Real FEP Test Results")
        print("=" * 40)
        
        successful = sum(1 for _, status, _, _ in results if status == "Success")
        print(f"Successful calculations: {successful}/{len(results)}")
        
        for name, status, value, time_taken in results:
            if status == "Success":
                print(f"✅ {name:<10}: {value:.3f} kcal/mol ({time_taken:.1f}s)")
            else:
                print(f"❌ {name:<10}: {status} ({time_taken:.1f}s)")
        
        if successful > 0:
            print(f"\n🎉 Real FEP calculations successful!")
            print(f"   This proves the FEP oracle is working with actual MD simulations!")
        else:
            print(f"\n⚠️  No successful calculations - may need debugging")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_fep():
    """Test with absolute minimal settings for debugging."""
    print("\n🔬 Minimal FEP Test (Debug Mode)")
    print("=" * 40)
    
    # Ultra-minimal config for debugging
    minimal_config = {
        "fep": {
            "mock_mode": False,
            "force_field": "amber14",
            "water_model": "tip3p",
            "num_lambda_windows": 2,  # Just 2 points: 0.0, 1.0
            "simulation_time": 0.001,  # 1 ps
            "temperature": 298.15,
            "equilibration_steps": 10,  # Almost no equilibration
            "production_steps": 50,     # Very short production
        }
    }
    
    print("Minimal configuration:")
    for key, value in minimal_config["fep"].items():
        print(f"  {key}: {value}")
    
    try:
        from al_fep.oracles.fep_oracle import FEPOracle
        
        oracle = FEPOracle(
            target="7jvr",
            receptor_file="data/targets/7jvr/7jvr_system.pdb",
            config=minimal_config
        )
        
        print("Testing with water molecule (smallest possible)...")
        result = oracle.evaluate("O")  # Water
        
        if result.get('error'):
            print(f"Even minimal test failed: {result['error']}")
            return False
        else:
            print(f"✅ Minimal test passed: {result.get('fep_score')} kcal/mol")
            return True
            
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🎯 Real FEP Testing")
    print("=" * 50)
    
    # Check if data is prepared
    if not os.path.exists("fep_test_config.json"):
        print("❌ Test not prepared. Run prepare_fep_test.py first")
        return
    
    if not os.path.exists("data/targets/7jvr/7jvr_system.pdb"):
        print("❌ Protein file missing. Run prepare_fep_test.py first")
        return
    
    # Run real FEP test
    print("Starting real FEP test...")
    print("This will perform actual molecular dynamics simulations!")
    
    success = test_real_fep_with_7jvr()
    
    if not success:
        print("\nTrying minimal debug test...")
        minimal_success = test_minimal_fep()
        
        if minimal_success:
            print("✅ Minimal test passed - FEP framework is working")
            print("💡 The full test may need longer simulation times or different parameters")
        else:
            print("❌ Even minimal test failed - check OpenMM installation and protein file")
    
    print(f"\n🏁 Testing complete!")

if __name__ == "__main__":
    main()
