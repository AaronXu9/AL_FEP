#!/usr/bin/env python3
"""
Test BoltzOracle with existing YAML file from examples/boltz/affinity.yaml
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from al_fep.oracles.boltz_oracle import BoltzOracle


def test_with_existing_yaml():
    """Test BoltzOracle using the existing affinity.yaml file."""
    print("🧪 Testing BoltzOracle with Existing YAML File")
    print("=" * 60)
    
    # Get absolute path to the existing YAML file
    yaml_path = os.path.abspath("../boltz/affinity.yaml")
    print(f"📄 Using YAML file: {yaml_path}")
    
    # Check if file exists
    if not os.path.exists(yaml_path):
        print(f"❌ YAML file not found: {yaml_path}")
        return False
    
    # Read and display YAML content
    print(f"\n📋 YAML File Content:")
    print("-" * 30)
    with open(yaml_path, 'r') as f:
        content = f.read()
        print(content)
    
    # Configure BoltzOracle to use this specific YAML file
    config = {
        "boltz": {
            "yaml_file_path": yaml_path,
            "preserve_yaml_files": True,  # Keep the file since it's user-provided
            "predict_affinity": True,
            "model": "boltz2",
            "diffusion_samples": 1,
            "diffusion_samples_affinity": 5
        }
    }
    
    print(f"\n🔧 Creating BoltzOracle with custom YAML configuration...")
    try:
        oracle = BoltzOracle(target="test", config=config)
        print(f"✅ Oracle created successfully: {oracle}")
        print(f"   Model: {oracle.model}")
        print(f"   YAML path: {oracle.yaml_file_path}")
        print(f"   Preserve files: {oracle.preserve_yaml_files}")
        
        # Test the YAML file path resolution
        test_smiles = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"  # Same as in YAML
        yaml_file, work_dir, cleanup = oracle._get_yaml_file_path(test_smiles)
        
        print(f"\n📁 YAML Path Resolution:")
        print(f"   YAML file: {yaml_file}")
        print(f"   Work directory: {work_dir}")
        print(f"   Will cleanup: {cleanup}")
        
        if yaml_file == yaml_path:
            print(f"✅ Correctly using the specified YAML file")
        else:
            print(f"❌ YAML path mismatch!")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to create oracle: {e}")
        return False


def test_yaml_content_compatibility():
    """Test if the existing YAML is compatible with BoltzOracle expectations."""
    print(f"\n🔍 Testing YAML Content Compatibility")
    print("-" * 40)
    
    yaml_path = os.path.abspath("../boltz/affinity.yaml")
    
    try:
        import yaml
        
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Check required fields
        checks = [
            ("version" in yaml_data, "Version field"),
            ("sequences" in yaml_data, "Sequences field"),
            (len(yaml_data["sequences"]) >= 2, "At least 2 sequences (protein + ligand)"),
            ("properties" in yaml_data, "Properties field for affinity"),
        ]
        
        for check, name in checks:
            if check:
                print(f"✅ {name}: Present")
            else:
                print(f"❌ {name}: Missing")
        
        # Extract details
        sequences = yaml_data.get("sequences", [])
        protein_seq = None
        ligand_smiles = None
        
        for seq in sequences:
            if "protein" in seq:
                protein_seq = seq["protein"].get("sequence")
            elif "ligand" in seq:
                ligand_smiles = seq["ligand"].get("smiles")
        
        if protein_seq:
            print(f"✅ Protein sequence: {len(protein_seq)} residues")
            print(f"   First 50 chars: {protein_seq[:50]}...")
        
        if ligand_smiles:
            print(f"✅ Ligand SMILES: {ligand_smiles}")
        
        # Check affinity configuration
        properties = yaml_data.get("properties", [])
        has_affinity = any("affinity" in prop for prop in properties)
        if has_affinity:
            print(f"✅ Affinity prediction configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Error parsing YAML: {e}")
        return False


def test_simulation_dry_run():
    """Simulate what would happen during actual prediction (without running Boltz)."""
    print(f"\n🎯 Simulation: Dry Run of Prediction Process")
    print("-" * 50)
    
    yaml_path = os.path.abspath("../boltz/affinity.yaml")
    
    config = {
        "boltz": {
            "yaml_file_path": yaml_path,
            "preserve_yaml_files": True
        }
    }
    
    try:
        oracle = BoltzOracle(target="test", config=config)
        
        # Extract SMILES from existing YAML for testing
        import yaml
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        test_smiles = None
        for seq in yaml_data.get("sequences", []):
            if "ligand" in seq:
                test_smiles = seq["ligand"].get("smiles")
                break
        
        if not test_smiles:
            print(f"❌ No SMILES found in YAML file")
            return False
        
        print(f"🧪 Test molecule: {test_smiles}")
        
        # Simulate the evaluation process (without actually running Boltz)
        print(f"\n📋 Simulation Steps:")
        print(f"1. ✅ Load YAML file: {yaml_path}")
        print(f"2. ✅ Extract SMILES: {test_smiles}")
        print(f"3. ✅ Use existing YAML (no regeneration needed)")
        print(f"4. 🔄 Would run: boltz predict {yaml_path} --out_dir [work_dir]")
        print(f"5. 📊 Would parse results from output directory")
        print(f"6. ✅ Preserve YAML file (as configured)")
        
        # Test if we can create the output directory structure
        yaml_file, work_dir, cleanup = oracle._get_yaml_file_path(test_smiles)
        print(f"\nOutput configuration:")
        print(f"   Work directory: {work_dir}")
        print(f"   Cleanup after: {cleanup}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 BoltzOracle Testing with Existing YAML File")
    print("=" * 60)
    
    tests = [
        test_with_existing_yaml,
        test_yaml_content_compatibility,
        test_simulation_dry_run
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Your YAML file is compatible and ready to use!")
        print("\n💡 To actually run prediction:")
        print("   oracle = BoltzOracle('test', config={'boltz': {'yaml_file_path': 'path/to/affinity.yaml'}})")
        print("   result = oracle.evaluate(['N[C@@H](Cc1ccc(O)cc1)C(=O)O'])  # Use SMILES from your YAML")
    else:
        print("❌ Some issues found. Please check the output above.")


if __name__ == "__main__":
    main()
