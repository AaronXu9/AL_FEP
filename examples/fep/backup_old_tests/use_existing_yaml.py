#!/usr/bin/env python3
"""
Practical example: Using BoltzOracle with your existing affinity.yaml file
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from al_fep.oracles.boltz_oracle import BoltzOracle


def run_prediction_with_existing_yaml():
    """Run actual prediction using your existing YAML file."""
    print("🚀 Running BoltzOracle with Existing YAML File")
    print("=" * 60)
    
    # Path to your existing YAML file
    yaml_path = os.path.abspath("examples/boltz/affinity.yaml")
    print(f"📄 Using YAML file: {yaml_path}")
    
    # Configuration to use your existing YAML file
    config = {
        "boltz": {
            # Use your existing YAML file
            "yaml_file_path": yaml_path,
            # Keep the file (don't delete it)
            "preserve_yaml_files": True,
            # Boltz model settings
            "model": "boltz2",
            "diffusion_samples": 1,
            "recycling_steps": 3,
            "sampling_steps": 200,
            # Affinity prediction settings
            "predict_affinity": True,
            "diffusion_samples_affinity": 5,
            "affinity_mw_correction": False,
            # Output settings
            "output_format": "pdb",
            "use_msa_server": True,
            "use_potentials": False
        }
    }
    
    print("\n🔧 Creating BoltzOracle...")
    try:
        oracle = BoltzOracle(target="test", config=config)
        print(f"✅ Oracle created: {oracle}")
        
        # The SMILES from your YAML file
        test_smiles = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"  # Tyrosine
        print(f"🧪 Test molecule: {test_smiles}")
        
        print(f"\n📋 Oracle Configuration:")
        print(f"   Model: {oracle.model}")
        print(f"   Diffusion samples: {oracle.diffusion_samples}")
        print(f"   Affinity prediction: {oracle.predict_affinity}")
        print(f"   YAML file: {oracle.yaml_file_path}")
        print(f"   Preserve files: {oracle.preserve_yaml_files}")
        
        # NOTE: Uncomment the lines below to run actual prediction
        # This requires Boltz to be installed and may take time
        print(f"\n⚠️  To run actual prediction, uncomment the lines below:")
        result = oracle.evaluate([test_smiles])
        print(f'Result: {result}')

        # For demonstration, we'll just show what the prediction call would look like
        print(f"\n💡 Prediction call would be:")
        print(f"   result = oracle.evaluate(['{test_smiles}'])")
        
        # Show what the result structure would look like
        print(f"\n📊 Expected result format:")
        expected_result = {
            "score": "binding_affinity (converted to positive)",
            "binding_affinity": "predicted pIC50 or pKd value",
            "binding_probability": "probability of binding",
            "confidence_score": "model confidence",
            "ptm": "predicted TM-score",
            "iptm": "interface predicted TM-score",
            "plddt": "predicted LDDT score",
            "structure_file": "path to predicted structure",
            "method": "Boltz-2",
            "model": "boltz2",
            "samples": 1
        }
        
        print(f"{result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def show_yaml_analysis():
    """Analyze your YAML file and show what it contains."""
    print(f"\n🔍 Analyzing Your YAML File")
    print("-" * 40)
    
    yaml_path = os.path.abspath("examples/boltz/affinity.yaml")
    
    try:
        import yaml
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        print(f"📋 YAML Analysis:")
        print(f"   Version: {yaml_data.get('version', 'not specified')}")
        
        sequences = yaml_data.get("sequences", [])
        print(f"   Sequences: {len(sequences)} total")
        
        for i, seq in enumerate(sequences):
            if "protein" in seq:
                protein = seq["protein"]
                print(f"   Protein {i+1}:")
                print(f"      ID: {protein.get('id', 'not specified')}")
                print(f"      Length: {len(protein.get('sequence', ''))} residues")
                
            elif "ligand" in seq:
                ligand = seq["ligand"]
                print(f"   Ligand {i+1}:")
                print(f"      ID: {ligand.get('id', 'not specified')}")
                print(f"      SMILES: {ligand.get('smiles', 'not specified')}")
        
        properties = yaml_data.get("properties", [])
        if properties:
            print(f"   Properties: {len(properties)} configured")
            for prop in properties:
                if "affinity" in prop:
                    affinity = prop["affinity"]
                    print(f"      Affinity prediction for binder: {affinity.get('binder', 'not specified')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing YAML: {e}")
        return False


def show_alternatives():
    """Show alternative ways to use the YAML file."""
    print(f"\n🎯 Alternative Usage Patterns")
    print("-" * 40)
    
    yaml_path = os.path.abspath("examples/boltz/affinity.yaml")
    
    print(f"💡 Option 1: Use YAML file as-is")
    print(f"   config = {{'boltz': {{'yaml_file_path': '{yaml_path}'}}}}")
    print(f"   oracle = BoltzOracle('test', config=config)")
    print(f"   result = oracle.evaluate(['N[C@@H](Cc1ccc(O)cc1)C(=O)O'])")
    
    print(f"\n💡 Option 2: Use as template (copy to temp directory)")
    print(f"   config = {{'boltz': {{'yaml_template_dir': '/tmp/boltz_templates'}}}}")
    print(f"   oracle = BoltzOracle('test', config=config)")
    print(f"   # Oracle will copy YAML structure for each prediction")
    
    print(f"\n💡 Option 3: Extract info and use BoltzOracle's auto-generation")
    print(f"   # Let BoltzOracle create YAML files from protein sequence + SMILES")
    print(f"   config = {{'boltz': {{'predict_affinity': True}}}}")
    print(f"   oracle = BoltzOracle('test', config=config)")
    print(f"   result = oracle.evaluate(['N[C@@H](Cc1ccc(O)cc1)C(=O)O'])")


def main():
    """Main function to run all examples."""
    print("🧪 BoltzOracle Usage with Existing YAML File")
    print("=" * 60)
    
    success = True
    
    try:
        success &= run_prediction_with_existing_yaml()
        success &= show_yaml_analysis()
        show_alternatives()
        
        print(f"\n" + "=" * 60)
        if success:
            print("🎉 Your YAML file is ready to use with BoltzOracle!")
            print("\n🚀 Next steps:")
            print("1. Make sure Boltz is installed: pip install boltz")
            print("2. Uncomment the prediction lines in the code above")
            print("3. Run the prediction!")
        else:
            print("❌ Some issues were found. Please check the output above.")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
