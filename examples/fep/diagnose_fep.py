#!/usr/bin/env python3
"""
Diagnostic test for FEP setup
"""

import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def diagnose_fep_setup():
    """Diagnose FEP setup step by step."""
    print("🔍 FEP Setup Diagnostics")
    print("=" * 40)
    
    # Check files
    config_file = "fep_test_config.json"
    protein_file = "data/targets/7jvr/7jvr_system.pdb"
    
    print("1. Checking files...")
    if os.path.exists(config_file):
        print(f"   ✅ Config: {config_file}")
        with open(config_file) as f:
            config = json.load(f)
        print(f"   Mock mode: {config['fep']['mock_mode']}")
    else:
        print(f"   ❌ Config missing: {config_file}")
        return False
    
    if os.path.exists(protein_file):
        print(f"   ✅ Protein: {protein_file}")
        with open(protein_file) as f:
            lines = f.readlines()
        atom_lines = [l for l in lines if l.startswith('ATOM')]
        print(f"   Protein atoms: {len(atom_lines)}")
    else:
        print(f"   ❌ Protein missing: {protein_file}")
        return False
    
    # Check imports
    print("\n2. Checking imports...")
    try:
        from al_fep.oracles.fep_oracle import FEPOracle
        print("   ✅ FEPOracle import")
    except Exception as e:
        print(f"   ❌ FEPOracle import failed: {e}")
        return False
    
    try:
        import openmm
        print(f"   ✅ OpenMM {openmm.version.version}")
    except Exception as e:
        print(f"   ❌ OpenMM import failed: {e}")
        return False
    
    # Test oracle creation
    print("\n3. Testing oracle creation...")
    try:
        oracle = FEPOracle(
            target="7jvr",
            receptor_file=protein_file,
            config=config
        )
        print("   ✅ Oracle created successfully")
        print(f"   Mock mode: {oracle.mock_mode}")
        print(f"   Force field: {oracle.force_field}")
        print(f"   Lambda windows: {oracle.num_lambda_windows}")
        return True
    except Exception as e:
        print(f"   ❌ Oracle creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ligand_preparation():
    """Test ligand preparation separately."""
    print("\n4. Testing ligand preparation...")
    
    try:
        from al_fep.oracles.fep_oracle import FEPOracle
        
        config = {"fep": {"mock_mode": False}}
        oracle = FEPOracle(target="test", config=config)
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            temp_file = f.name
        
        # Test ligand prep
        success = oracle._prepare_ligand_for_fep("CCO", temp_file)
        
        if success and os.path.exists(temp_file):
            print("   ✅ Ligand preparation successful")
            with open(temp_file) as f:
                lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith('ATOM') or l.startswith('HETATM')]
            print(f"   Ligand atoms: {len(atom_lines)}")
            
            # Clean up
            os.unlink(temp_file)
            return True
        else:
            print("   ❌ Ligand preparation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Ligand preparation error: {e}")
        return False

def main():
    """Run all diagnostics."""
    print("🔬 FEP Real Test Diagnostics")
    print("=" * 50)
    
    # Run diagnostics
    setup_ok = diagnose_fep_setup()
    if not setup_ok:
        print("\n❌ Setup diagnostics failed")
        return
    
    ligand_ok = test_ligand_preparation()
    if not ligand_ok:
        print("\n❌ Ligand preparation failed")
        return
    
    print("\n✅ All diagnostics passed!")
    print("The FEP system should be ready for real calculations.")
    print("Note: Real FEP calculations may still take several minutes.")

if __name__ == "__main__":
    main()
