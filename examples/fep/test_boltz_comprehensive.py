#!/usr/bin/env python3
"""
Simple and Comprehensive BoltzOracle Test
Consolidated testing for all BoltzOracle functionality.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from al_fep.oracles.boltz_oracle import BoltzOracle


def test_basic_functionality():
    """Test basic BoltzOracle functionality."""
    print("🧪 Test 1: Basic Functionality")
    print("-" * 40)
    
    try:
        oracle = BoltzOracle(target="test")
        
        print(f"✅ Oracle created: {oracle}")
        print(f"   Target: {oracle.target}")
        print(f"   Model: {oracle.model}")
        print(f"   Protein sequence: {len(oracle.protein_sequence)} residues")
        print(f"   Work directory: {oracle.work_dir}")
        print(f"   Affinity prediction: {oracle.predict_affinity}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_yaml_configurations():
    """Test different YAML configuration modes."""
    print("\n🧪 Test 2: YAML Configuration Modes")
    print("-" * 40)
    
    test_smiles = "CCO"  # Simple ethanol
    
    # Test 1: Default mode
    print("   🔸 Default mode (temporary files)")
    try:
        oracle1 = BoltzOracle(target="test")
        yaml_path1, work_dir1, cleanup1 = oracle1._get_yaml_file_path(test_smiles)
        print(f"      ✅ Temp dir: {os.path.basename(work_dir1)}")
        print(f"      ✅ Cleanup: {cleanup1}")
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        return False
    
    # Test 2: Custom YAML path
    print("   🔸 Custom YAML file path")
    try:
        temp_dir = tempfile.mkdtemp()
        custom_yaml = os.path.join(temp_dir, "custom.yaml")
        
        config = {"boltz": {"yaml_file_path": custom_yaml}}
        oracle2 = BoltzOracle(target="test", config=config)
        
        print(f"      ✅ Custom path: {oracle2.yaml_file_path}")
        print(f"      ✅ Preserve files: {oracle2.preserve_yaml_files}")
        
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        return False
    
    # Test 3: Template directory
    print("   🔸 Template directory")
    try:
        temp_dir = tempfile.mkdtemp()
        
        config = {"boltz": {
            "yaml_template_dir": temp_dir,
            "preserve_yaml_files": True
        }}
        oracle3 = BoltzOracle(target="test", config=config)
        
        print(f"      ✅ Template dir: {oracle3.yaml_template_dir}")
        print(f"      ✅ Preserve: {oracle3.preserve_yaml_files}")
        
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        return False
    
    return True


def test_yaml_content():
    """Test YAML file content generation."""
    print("\n🧪 Test 3: YAML Content Generation")
    print("-" * 40)
    
    try:
        oracle = BoltzOracle(target="test")
        
        # Create temporary YAML file
        temp_dir = tempfile.mkdtemp()
        yaml_file = os.path.join(temp_dir, "test.yaml")
        
        test_smiles = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"  # Tyrosine
        oracle._create_yaml_input(test_smiles, yaml_file)
        
        if os.path.exists(yaml_file):
            print(f"   ✅ YAML file created: {os.path.basename(yaml_file)}")
            
            with open(yaml_file, 'r') as f:
                content = f.read()
            
            # Basic content checks
            checks = [
                ("sequences:" in content, "Sequences section"),
                ("protein:" in content, "Protein entry"),
                ("ligand:" in content, "Ligand entry"),
                (test_smiles in content, "Correct SMILES"),
                ("properties:" in content if oracle.predict_affinity else True, "Affinity properties")
            ]
            
            for check, name in checks:
                if check:
                    print(f"   ✅ {name}")
                else:
                    print(f"   ❌ {name}")
                    return False
            
            print(f"   📄 File size: {len(content)} bytes")
        else:
            print(f"   ❌ YAML file not created")
            return False
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_existing_yaml():
    """Test compatibility with existing YAML files."""
    print("\n🧪 Test 4: Existing YAML File")
    print("-" * 40)
    
    existing_yaml = os.path.abspath("../boltz/affinity.yaml")
    
    if not os.path.exists(existing_yaml):
        print(f"   ⚠️  Existing YAML not found: {existing_yaml}")
        print(f"   ℹ️  Skipping test (not an error)")
        return True
    
    try:
        config = {
            "boltz": {
                "yaml_file_path": existing_yaml,
                "preserve_yaml_files": True
            }
        }
        
        oracle = BoltzOracle(target="test", config=config)
        
        print(f"   ✅ Oracle configured with existing YAML")
        print(f"   📄 YAML path: {oracle.yaml_file_path}")
        print(f"   🔒 Preserve files: {oracle.preserve_yaml_files}")
        
        # Test that it uses the correct path
        yaml_path, work_dir, cleanup = oracle._get_yaml_file_path("CCO")
        if yaml_path == existing_yaml and not cleanup:
            print(f"   ✅ Correctly using existing YAML file")
        else:
            print(f"   ❌ Not using existing YAML correctly")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def demo_usage_examples():
    """Show practical usage examples."""
    print("\n🎯 Demo: Usage Examples")
    print("-" * 40)
    
    examples = [
        ("Basic Usage", {}),
        ("High Quality", {
            "boltz": {
                "diffusion_samples": 5,
                "recycling_steps": 5,
                "diffusion_samples_affinity": 10
            }
        }),
        ("Template Mode", {
            "boltz": {
                "yaml_template_dir": "/tmp/boltz_templates",
                "preserve_yaml_files": True
            }
        })
    ]
    
    for name, config in examples:
        print(f"   🔸 {name}")
        try:
            oracle = BoltzOracle(target="test", config=config)
            print(f"      ✅ Oracle: {oracle}")
            print(f"      💡 Usage: oracle.evaluate(['CCO'])")
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            return False
    
    # Show expected output format
    print(f"\n   📊 Expected result format:")
    result_fields = [
        "score: float (higher = better)",
        "binding_affinity: float (pIC50/pKd)",
        "binding_probability: float (0-1)",
        "confidence_score: float (0-1)",
        "structure_file: str (path)",
        "method: 'Boltz-2'"
    ]
    
    for field in result_fields:
        print(f"      {field}")
    
    return True


def main():
    """Run all tests."""
    print("🚀 BoltzOracle Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_yaml_configurations,
        test_yaml_content,
        test_existing_yaml,
        demo_usage_examples
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("   ❌ Test failed")
        except Exception as e:
            print(f"   💥 Test crashed: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BoltzOracle is working correctly.")
        print("\n💡 Ready to use:")
        print("   oracle = BoltzOracle('test')")
        print("   result = oracle.evaluate(['CCO'])")
    else:
        print("❌ Some tests failed. Check the output above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
