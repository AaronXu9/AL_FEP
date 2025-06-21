#!/usr/bin/env python3
"""
Demo: BoltzOracle with Molecular File Utils - Direct Sequence Integration
Shows how to use extracted protein sequences directly with BoltzOracle.
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


def demo_direct_sequence_integration():
    """Demo using extracted protein sequences directly with BoltzOracle."""
    print("🚀 Direct Sequence Integration Demo")
    print("=" * 60)
    
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
        print(f"\n🧬 Step 1: Extract protein sequence from PDB")
        protein_sequences = extract_protein_sequence_from_pdb(str(pdb_file))
        
        # Use first chain
        chain_id = list(protein_sequences.keys())[0]
        protein_sequence = protein_sequences[chain_id]
        
        print(f"   Chain {chain_id}: {len(protein_sequence)} residues")
        print(f"   First 50: {protein_sequence[:50]}...")
        
        # Step 2: Extract SMILES from SDF
        print(f"\n🧪 Step 2: Extract SMILES from SDF")
        smiles_list = sdf_to_smiles_list(str(sdf_file))
        print(f"   Found {len(smiles_list)} molecules")
        print(f"   First 3 SMILES:")
        for i, smiles in enumerate(smiles_list[:3]):
            print(f"      {i+1}. {smiles}")
        
        # Step 3: Create BoltzOracle with direct sequence
        print(f"\n🎯 Step 3: Create BoltzOracle with direct protein sequence")
        
        config = {
            "boltz": {
                "protein_sequence": protein_sequence,  # Direct sequence input!
                "predict_affinity": True
            }
        }
        
        oracle = BoltzOracle(target="direct_demo", config=config)
        print(f"   ✅ Oracle created: {oracle}")
        print(f"   🧬 Protein sequence: {len(oracle.protein_sequence)} residues")
        
        # Step 4: Test prediction setup
        print(f"\n🧪 Step 4: Test prediction setup")
        
        test_smiles = smiles_list[0]
        print(f"   Test molecule: {test_smiles}")
        
        # Create a temporary YAML to verify sequence is used correctly
        temp_dir = project_root / "temp" / "direct_demo"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        yaml_file = temp_dir / "test_input.yaml"
        oracle._create_yaml_input(test_smiles, str(yaml_file))
        
        print(f"   ✅ YAML created: {yaml_file.name}")
        
        # Verify the YAML contains our protein sequence
        with open(yaml_file, 'r') as f:
            yaml_content = f.read()
        
        if protein_sequence in yaml_content:
            print(f"   ✅ YAML contains correct protein sequence")
        else:
            print(f"   ❌ YAML sequence mismatch")
            return False
        
        print(f"\n💡 Ready for prediction:")
        print(f"   result = oracle.evaluate(['{test_smiles}'])")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def demo_multi_chain_selection():
    """Demo selecting specific chains from multi-chain PDB files."""
    print(f"\n🔗 Multi-Chain Selection Demo")
    print("-" * 50)
    
    # Use the 7jvr multi-chain PDB
    data_dir = project_root / "data"
    pdb_file = data_dir / "targets/7jvr/7jvr_system_prepared.pdb"
    
    if not pdb_file.exists():
        print("   ⚠️  7jvr PDB file not found, skipping multi-chain demo")
        return True
    
    try:
        print(f"   📄 Multi-chain PDB: {pdb_file.name}")
        
        # Extract all chains
        all_sequences = extract_protein_sequence_from_pdb(str(pdb_file))
        print(f"   🧬 Available chains: {list(all_sequences.keys())}")
        
        for chain, seq in all_sequences.items():
            print(f"      Chain {chain}: {len(seq)} residues")
        
        # Select longest chain for demonstration
        longest_chain = max(all_sequences.keys(), key=lambda k: len(all_sequences[k]))
        selected_sequence = all_sequences[longest_chain]
        
        print(f"   ✅ Selected chain {longest_chain} ({len(selected_sequence)} residues)")
        
        # Create BoltzOracle with selected chain
        config = {
            "boltz": {
                "protein_sequence": selected_sequence,
                "predict_affinity": True
            }
        }
        
        oracle = BoltzOracle(target=f"7jvr_chain_{longest_chain}", config=config)
        print(f"   ✅ Oracle created with chain {longest_chain}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def demo_batch_processing_with_extracted_data():
    """Demo batch processing using extracted molecular data."""
    print(f"\n📦 Batch Processing with Extracted Data")
    print("-" * 50)
    
    try:
        # Extract data from files
        data_dir = project_root / "data"
        pdb_file = data_dir / "BZP_FEP_protein_model_8B5H.pdb"
        sdf_file = data_dir / "targets/BMC_FEP_validation_set_J_Med_Chem_2020.sdf"
        
        # Get protein sequence and SMILES list
        protein_sequences = extract_protein_sequence_from_pdb(str(pdb_file))
        protein_sequence = list(protein_sequences.values())[0]
        
        molecules = extract_smiles_from_sdf(str(sdf_file))
        
        print(f"   📊 Dataset summary:")
        print(f"      Protein: {len(protein_sequence)} residues")
        print(f"      Molecules: {len(molecules)} compounds")
        
        # Create oracle for batch processing
        config = {
            "boltz": {
                "protein_sequence": protein_sequence,
                "predict_affinity": True,
                "preserve_yaml_files": False  # Clean up temp files
            }
        }
        
        oracle = BoltzOracle(target="batch_demo", config=config)
        
        # Prepare batch data
        batch_smiles = [mol['smiles'] for mol in molecules[:5]]  # First 5 molecules
        batch_names = [mol.get('name', f'mol_{i}') for i, mol in enumerate(molecules[:5])]
        
        print(f"\n   🧪 Batch test molecules:")
        for name, smiles in zip(batch_names, batch_smiles):
            print(f"      {name}: {smiles}")
        
        print(f"\n   💡 Batch prediction would be:")
        print(f"      results = oracle.evaluate({batch_smiles})")
        print(f"      # Returns list of {len(batch_smiles)} prediction results")
        
        # Show how to access experimental data
        exp_data = []
        for mol in molecules[:5]:
            exp_info = {}
            for key, value in mol.items():
                if any(exp_key in key.lower() for exp_key in ['pic50', 'exp', 'dg']):
                    exp_info[key] = value
            exp_data.append(exp_info)
        
        if any(exp_data):
            print(f"\n   📊 Available experimental data for validation:")
            for i, (name, exp) in enumerate(zip(batch_names, exp_data)):
                if exp:
                    print(f"      {name}: {list(exp.keys())}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def demo_custom_yaml_with_extracted_data():
    """Demo creating custom YAML files with extracted data."""
    print(f"\n⚙️  Custom YAML with Extracted Data")
    print("-" * 50)
    
    try:
        data_dir = project_root / "data"
        pdb_file = data_dir / "BZP_FEP_protein_model_8B5H.pdb"
        sdf_file = data_dir / "targets/BMC_FEP_validation_set_J_Med_Chem_2020.sdf"
        
        # Create custom YAML using utility function
        temp_dir = project_root / "temp" / "custom_yaml_demo"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        yaml_file = temp_dir / "custom_input.yaml"
        
        created_yaml = create_boltz_input_from_files(
            str(pdb_file),
            str(sdf_file),
            str(yaml_file),
            molecule_index=0
        )
        
        print(f"   ✅ Custom YAML created: {yaml_file.name}")
        
        # Use the custom YAML with BoltzOracle
        config = {
            "boltz": {
                "yaml_file_path": str(yaml_file),
                "preserve_yaml_files": True
            }
        }
        
        oracle = BoltzOracle(target="custom_yaml_demo", config=config)
        print(f"   ✅ Oracle configured with custom YAML")
        print(f"   📄 YAML path: {oracle.yaml_file_path}")
        
        # Show YAML content preview
        with open(yaml_file, 'r') as f:
            content = f.read()
        
        print(f"\n   📋 YAML content preview:")
        lines = content.split('\n')
        for line in lines[:10]:
            print(f"      {line}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def show_integration_options():
    """Show all available integration options."""
    print(f"\n🎯 Integration Options Summary")
    print("-" * 50)
    
    print(f"   ✨ Option 1: Direct Sequence (Recommended)")
    print(f"      # Extract sequence from PDB, use directly")
    print(f"      sequences = extract_protein_sequence_from_pdb('protein.pdb')")
    print(f"      config = {{'boltz': {{'protein_sequence': sequences['A']}}}}")
    print(f"      oracle = BoltzOracle('target', config=config)")
    
    print(f"\n   ✨ Option 2: Custom YAML File")
    print(f"      # Create YAML from PDB+SDF, use as template")
    print(f"      yaml_file = create_boltz_input_from_files('protein.pdb', 'ligands.sdf', 'input.yaml')")
    print(f"      config = {{'boltz': {{'yaml_file_path': yaml_file}}}}")
    print(f"      oracle = BoltzOracle('target', config=config)")
    
    print(f"\n   ✨ Option 3: Hybrid Approach")
    print(f"      # Extract data, let BoltzOracle handle YAML generation")
    print(f"      sequences = extract_protein_sequence_from_pdb('protein.pdb')")
    print(f"      smiles_list = sdf_to_smiles_list('ligands.sdf')")
    print(f"      config = {{'boltz': {{'protein_sequence': sequences['A']}}}}")
    print(f"      oracle = BoltzOracle('target', config=config)")
    print(f"      results = oracle.evaluate(smiles_list)")
    
    print(f"\n   🎁 Benefits:")
    print(f"   • No need to create target-specific sequence files")
    print(f"   • Support for multi-chain PDB files")
    print(f"   • Automatic SMILES extraction from SDF files")
    print(f"   • Integration with experimental data")
    print(f"   • Flexible configuration options")


def main():
    """Run all demos."""
    print("🧬 BoltzOracle + Molecular Utils Integration")
    print("=" * 70)
    
    demos = [
        demo_direct_sequence_integration,
        demo_multi_chain_selection,
        demo_batch_processing_with_extracted_data,
        demo_custom_yaml_with_extracted_data
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
    
    show_integration_options()
    
    print(f"\n{'='*70}")
    print(f"📊 Demo Results: {passed}/{total} demos successful")
    
    if passed == total:
        print("🎉 Integration working perfectly!")
        print("\n🚀 You can now:")
        print("   ✓ Extract protein sequences from any PDB file")
        print("   ✓ Extract SMILES from any SDF file")
        print("   ✓ Use extracted data directly with BoltzOracle")
        print("   ✓ Process multi-chain proteins and large datasets")
        print("   ✓ Access experimental data for validation")
    else:
        print("❌ Some demos failed. Check the output above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
