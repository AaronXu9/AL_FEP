#!/usr/bin/env python3
"""
Test BoltzOracle with existing YAML file
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from al_fep.oracles.boltz_oracle import BoltzOracle


def test_boltz_oracle_with_yaml():
    """Test BoltzOracle using existing YAML file."""
    print("🧪 Testing BoltzOracle with Existing YAML File")
    print("=" * 60)
    
    # Get absolute path to the YAML file
    yaml_path = project_root / "examples" / "boltz" / "affinity.yaml"
    
    print(f"📄 YAML file: {yaml_path}")
    
    if not yaml_path.exists():
        print(f"❌ YAML file not found: {yaml_path}")
        return False
    
    # Configuration to use the existing YAML file
    config = {
        "boltz": {
            "yaml_file_path": str(yaml_path),
            "preserve_yaml_files": True
        }
    }
    
    try:
        print("🔧 Creating BoltzOracle...")
        oracle = BoltzOracle('test', config=config)
        print(f"✅ Oracle created: {oracle}")
        
        # Test molecule from the YAML file
        test_smiles = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"  # Tyrosine
        print(f"🧪 Test molecule: {test_smiles}")
        
        print("\n⚠️  To run actual prediction, uncomment the line below:")
        # print("   # result = oracle.evaluate([test_smiles])")
        # print("   # print(f'Result: {result}')")
        
        # Uncomment this line to run actual prediction:
        result = oracle.evaluate([test_smiles])
        print(f"📊 Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_boltz_oracle_with_BMC(pdb_path, sdf_path):
    from al_fep.utils.molecular_file_utils import (
        extract_protein_sequence_from_pdb,
        extract_smiles_from_sdf,
        create_boltz_input_from_files,
        pdb_to_fasta
    )
    fasta_seq = extract_protein_sequence_from_pdb(pdb_path)
    smiles_list = extract_smiles_from_sdf(sdf_path)
    
    yaml_file = create_boltz_input_from_files(pdb_path, sdf_path, f"examples/boltz/BMC.yaml")
    oracle = BoltzOracle('BMC', config={"boltz": {"yaml_file_path": yaml_file}})
    # oracle._parse_boltz_output("examples/boltz/boltz_results_BMC", "BMC")
    results = oracle.evaluate("CC(=O)O")  # Example SMILES for acetic acid
    print(f"Results for BMC: {results}")
    return oracle

if __name__ == "__main__":
    # success = test_boltz_oracle_with_yaml()
    success = test_boltz_oracle_with_BMC(
        "/home/aoxu/projects/AL_FEP/data/BMC_FEP_protein_model_6ZB1.pdb",
        "/home/aoxu/projects/AL_FEP/data/targets/BMC_FEP_validation_set_J_Med_Chem_2020.sdf"
    )
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)