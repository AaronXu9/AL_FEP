#!/usr/bin/env python3
"""
Demo: Integration of molecular file utilities with BoltzOracle
Shows how to use PDB and SDF files to create Boltz predictions.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from al_fep.oracles.boltz_oracle import BoltzOracle
from al_fep.utils.molecular_file_utils import (
    extract_protein_sequence_from_pdb,
    extract_smiles_from_sdf,
    create_boltz_input_from_files,
    sdf_to_smiles_list
)


def demo_pdb_sdf_to_boltz_workflow():
    """Demo complete workflow from PDB/SDF files to Boltz predictions."""
    print("🚀 Complete PDB/SDF to Boltz Workflow Demo")
    print("=" * 70)
    
    # Available data files
    data_dir = project_root / "data"
    pdb_file = data_dir / "BZP_FEP_protein_model_8B5H.pdb"
    sdf_file = data_dir / "targets/BMC_FEP_validation_set_J_Med_Chem_2020.sdf"
    
    if not pdb_file.exists() or not sdf_file.exists():
        print("❌ Required data files not found")
        return False
    
    print(f"📄 PDB file: {pdb_file.name}")
    print(f"📄 SDF file: {sdf_file.name}")
    
    try:
        # Step 1: Extract protein sequence from PDB
        print(f"\n🧬 Step 1: Extract protein sequence")
        protein_sequences = extract_protein_sequence_from_pdb(str(pdb_file))
        
        for chain, seq in protein_sequences.items():
            print(f"   Chain {chain}: {len(seq)} residues")
            print(f"   First 50: {seq[:50]}...")
        
        # Step 2: Extract SMILES from SDF
        print(f"\n🧪 Step 2: Extract ligand SMILES")
        molecules = extract_smiles_from_sdf(str(sdf_file))
        print(f"   Found {len(molecules)} molecules")
        
        # Show first few molecules
        for i, mol in enumerate(molecules[:3]):
            name = mol.get('name', f'mol_{i+1}')
            smiles = mol['smiles']
            print(f"   {i+1}. {name}: {smiles}")
        
        # Step 3: Create Boltz YAML input file
        print(f"\n⚙️  Step 3: Create Boltz YAML input")
        temp_dir = project_root / "temp" / "boltz_demo"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        yaml_file = temp_dir / "demo_input.yaml"
        
        created_yaml = create_boltz_input_from_files(
            str(pdb_file),
            str(sdf_file),
            str(yaml_file),
            molecule_index=0  # Use first molecule
        )
        
        print(f"   ✅ YAML created: {yaml_file.name}")
        
        # Step 4: Use with BoltzOracle
        print(f"\n🎯 Step 4: Configure BoltzOracle")
        
        config = {
            "boltz": {
                "yaml_file_path": str(yaml_file),
                "preserve_yaml_files": True,
                "predict_affinity": True
            }
        }
        
        oracle = BoltzOracle(target="demo", config=config)
        print(f"   ✅ Oracle created: {oracle}")
        print(f"   📄 Using YAML: {oracle.yaml_file_path}")
        
        # Step 5: Test prediction setup (without running actual Boltz)
        print(f"\n🧪 Step 5: Prediction setup test")
        
        # Get SMILES from first molecule for testing
        test_smiles = molecules[0]['smiles']
        print(f"   Test molecule: {test_smiles}")
        
        # Test YAML file path resolution
        yaml_path, work_dir, cleanup = oracle._get_yaml_file_path(test_smiles)
        print(f"   YAML path: {yaml_path}")
        print(f"   Work dir: {work_dir}")
        print(f"   Cleanup: {cleanup}")
        
        if yaml_path == str(yaml_file):
            print(f"   ✅ Oracle correctly configured to use custom YAML file")
        else:
            print(f"   ❌ Oracle YAML path mismatch")
            return False
        
        print(f"\n💡 To run actual prediction:")
        print(f"   result = oracle.evaluate(['{test_smiles}'])")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in workflow: {e}")
        return False


def demo_batch_processing():
    """Demo batch processing multiple molecules from SDF file."""
    print(f"\n📦 Batch Processing Demo")
    print("-" * 50)
    
    # Use existing SDF file
    data_dir = project_root / "data"
    sdf_file = data_dir / "targets/BMC_FEP_validation_set_J_Med_Chem_2020.sdf"
    
    try:
        # Extract all SMILES
        smiles_list = sdf_to_smiles_list(str(sdf_file))
        print(f"   📊 Total molecules: {len(smiles_list)}")
        
        # Create BoltzOracle for batch processing
        oracle = BoltzOracle(target="batch_demo")
        
        # Test with first 3 molecules (without actually running predictions)
        test_smiles = smiles_list[:3]
        
        print(f"   🧪 Test molecules:")
        for i, smiles in enumerate(test_smiles):
            print(f"      {i+1}. {smiles}")
        
        print(f"\n   💡 Batch prediction would be:")
        print(f"      results = oracle.evaluate({test_smiles})")
        print(f"      # Returns list of {len(test_smiles)} prediction results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def demo_advanced_features():
    """Demo advanced features of the utility functions."""
    print(f"\n🔧 Advanced Features Demo")
    print("-" * 50)
    
    # Multi-chain PDB processing
    data_dir = project_root / "data"
    pdb_file = data_dir / "targets/7jvr/7jvr_system_prepared.pdb"
    
    if pdb_file.exists():
        print(f"   📄 Multi-chain PDB: {pdb_file.name}")
        
        try:
            # Extract all chains
            all_sequences = extract_protein_sequence_from_pdb(str(pdb_file))
            print(f"   🧬 Chains found: {list(all_sequences.keys())}")
            
            for chain, seq in all_sequences.items():
                print(f"      Chain {chain}: {len(seq)} residues")
            
            # Extract specific chain
            if 'A' in all_sequences:
                chain_a_seq = extract_protein_sequence_from_pdb(str(pdb_file), chain_id='A')
                print(f"   ✅ Specific chain extraction: Chain A has {len(chain_a_seq['A'])} residues")
            
        except Exception as e:
            print(f"   ❌ Multi-chain processing failed: {e}")
            return False
    
    # SDF with metadata processing
    sdf_file = data_dir / "targets/BMC_FEP_validation_set_J_Med_Chem_2020.sdf"
    
    try:
        print(f"\n   📄 SDF with metadata: {sdf_file.name}")
        molecules = extract_smiles_from_sdf(str(sdf_file))
        
        # Show molecule with all properties
        first_mol = molecules[0]
        print(f"   📋 Example molecule properties:")
        print(f"      Name: {first_mol.get('name', 'N/A')}")
        print(f"      SMILES: {first_mol['smiles']}")
        
        # Show experimental data if available
        exp_data = {}
        for key, value in first_mol.items():
            if any(exp_key in key.lower() for exp_key in ['pic50', 'exp', 'dg']):
                exp_data[key] = value
        
        if exp_data:
            print(f"      Experimental data:")
            for key, value in exp_data.items():
                print(f"         {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Advanced features failed: {e}")
        return False


def show_integration_summary():
    """Show summary of integration possibilities."""
    print(f"\n📋 Integration Summary")
    print("-" * 50)
    
    print(f"   🎯 Key Integration Points:")
    print(f"   1. PDB → Protein Sequence → BoltzOracle")
    print(f"   2. SDF → SMILES List → Batch Predictions")
    print(f"   3. PDB + SDF → Custom YAML → Specific Oracle Config")
    print(f"   4. Experimental Data from SDF → Validation")
    
    print(f"\n   💡 Workflow Options:")
    print(f"   Option A: Auto-generate (let BoltzOracle create YAML)")
    print(f"      oracle = BoltzOracle('target')")
    print(f"      results = oracle.evaluate(smiles_list)")
    
    print(f"\n   Option B: Custom YAML from files")
    print(f"      yaml_file = create_boltz_input_from_files(pdb, sdf, 'input.yaml')")
    print(f"      config = {{'boltz': {{'yaml_file_path': yaml_file}}}}")
    print(f"      oracle = BoltzOracle('target', config=config)")
    
    print(f"\n   Option C: Hybrid approach")
    print(f"      sequences = extract_protein_sequence_from_pdb(pdb_file)")
    print(f"      smiles_list = sdf_to_smiles_list(sdf_file)")
    print(f"      # Use extracted data with default BoltzOracle configuration")


def main():
    """Run all demos."""
    print("🧬 Molecular File Utils + BoltzOracle Integration Demo")
    print("=" * 80)
    
    demos = [
        demo_pdb_sdf_to_boltz_workflow,
        demo_batch_processing,
        demo_advanced_features
    ]
    
    passed = 0
    total = len(demos)
    
    for demo in demos:
        try:
            if demo():
                passed += 1
            else:
                print("   ❌ Demo failed")
        except Exception as e:
            print(f"   💥 Demo crashed: {e}")
    
    show_integration_summary()
    
    print(f"\n{'='*80}")
    print(f"📊 Demo Results: {passed}/{total} demos successful")
    
    if passed == total:
        print("🎉 Integration working perfectly!")
        print("\n🚀 Ready to use:")
        print("   # Extract data from your files")
        print("   sequences = extract_protein_sequence_from_pdb('protein.pdb')")
        print("   smiles_list = sdf_to_smiles_list('compounds.sdf')")
        print("   ")
        print("   # Create Boltz predictions")
        print("   oracle = BoltzOracle('your_target')")
        print("   results = oracle.evaluate(smiles_list)")
    else:
        print("❌ Some demos failed. Check the output above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
