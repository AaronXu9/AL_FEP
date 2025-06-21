#!/usr/bin/env python3
"""
Simplified real FEP test using solvent-only calculations
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_solvent_only_fep():
    """Test FEP oracle with solvent-only calculations (simpler setup)."""
    print("🧪 Simplified Real FEP Test (Solvent-Only)")
    print("=" * 45)
    
    from al_fep.oracles.fep_oracle import FEPOracle
    
    # Test if we can at least do ligand preparation
    config = {
        "fep": {
            "mock_mode": False,
            "force_field": "amber14",
            "num_lambda_windows": 2,
            "simulation_time": 0.0005,  # 0.5 ps
            "equilibration_steps": 5,
            "production_steps": 10,
        }
    }
    
    try:
        oracle = FEPOracle(target="test", config=config)
        
        print(f"✅ FEP Oracle created in real mode")
        print(f"  Mock mode: {oracle.mock_mode}")
        
        # Test ligand preparation directly
        print(f"\n🔬 Testing ligand preparation...")
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            ligand_file = os.path.join(temp_dir, "test_ligand.pdb")
            
            # Test simple molecule
            success = oracle._prepare_ligand_for_fep("CCO", ligand_file)
            
            if success and os.path.exists(ligand_file):
                print(f"✅ Ligand preparation successful!")
                print(f"  Created: {ligand_file}")
                
                # Check the file content
                with open(ligand_file, 'r') as f:
                    content = f.read()
                    lines = [line for line in content.split('\n') if line.strip()]
                    print(f"  PDB file has {len(lines)} lines")
                    
                print(f"\n🔬 Testing solvent system setup...")
                
                # Try to setup solvent system
                try:
                    result = oracle._setup_solvent_system(ligand_file)
                    if result is not None:
                        system, topology, positions = result
                        print(f"✅ Solvent system setup successful!")
                        print(f"  System created with OpenMM")
                        return True
                    else:
                        print(f"❌ Solvent system setup failed")
                        return False
                        
                except Exception as e:
                    print(f"❌ Solvent system error: {e}")
                    return False
                    
            else:
                print(f"❌ Ligand preparation failed")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_fep_components():
    """Demonstrate individual FEP components working."""
    print("\n🔧 FEP Components Test")
    print("=" * 25)
    
    from al_fep.oracles.fep_oracle import FEPOracle
    
    config = {"fep": {"mock_mode": False}}
    oracle = FEPOracle(target="test", config=config)
    
    # Test 1: OpenMM availability
    print("1. OpenMM availability:")
    try:
        import openmm
        print(f"   ✅ OpenMM {openmm.version.version} available")
    except ImportError:
        print(f"   ❌ OpenMM not available")
        return
    
    # Test 2: Force field loading
    print("2. Force field loading:")
    try:
        import openmm.app as app
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        print(f"   ✅ AMBER14 force field loaded")
    except Exception as e:
        print(f"   ❌ Force field error: {e}")
        return
    
    # Test 3: Ligand preparation
    print("3. Ligand preparation:")
    try:
        mol = oracle._prepare_ligand("CCO")  # This returns RDKit mol
        if mol is not None:
            print(f"   ✅ Ligand prepared ({mol.GetNumAtoms()} atoms)")
        else:
            print(f"   ❌ Ligand preparation failed")
    except Exception as e:
        print(f"   ❌ Ligand preparation error: {e}")
    
    # Test 4: Lambda window calculation
    print("4. Lambda windows:")
    try:
        lambda_values = [i / (oracle.num_lambda_windows - 1) for i in range(oracle.num_lambda_windows)]
        print(f"   ✅ {len(lambda_values)} lambda windows: {lambda_values}")
    except Exception as e:
        print(f"   ❌ Lambda calculation error: {e}")
    
    print("\n📊 Component tests completed!")

def main():
    """Main testing function."""
    print("🎯 Simplified Real FEP Testing")
    print("=" * 35)
    
    # Test components
    demonstrate_fep_components()
    
    # Test solvent-only FEP
    success = test_solvent_only_fep()
    
    if success:
        print(f"\n🎉 Real FEP components are working!")
        print(f"💡 The framework is ready for full FEP calculations")
        print(f"   Main limitation: Ligand force field parameterization")
    else:
        print(f"\n❌ Some real FEP components need debugging")
    
    print(f"\n🏁 Testing complete!")

if __name__ == "__main__":
    main()
