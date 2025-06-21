#!/usr/bin/env python3
"""
Comprehensive test of BoltzOracle implementation.
Tests all YAML configuration modes and functionality.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from al_fep.oracles.boltz_oracle import BoltzOracle


def test_basic_import():
    """Test basic import and initialization."""
    print("1. Testing basic import and initialization...")
    
    try:
        oracle = BoltzOracle(target="test")
        print(f"   ✓ BoltzOracle created: {oracle}")
        print(f"   ✓ Target: {oracle.target}")
        print(f"   ✓ Model: {oracle.model}")
        print(f"   ✓ Work directory: {oracle.work_dir}")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_yaml_modes():
    """Test different YAML file configuration modes."""
    print("\n2. Testing YAML file configuration modes...")
    
    # Test 1: Default (temporary files)
    print("   Testing default mode (temporary files)...")
    try:
        oracle = BoltzOracle(target="test")
        yaml_path, work_dir, cleanup = oracle._get_yaml_file_path("CCO")
        print(f"   ✓ Default mode - YAML: {yaml_path}, Work dir: {work_dir}, Cleanup: {cleanup}")
        
        # Create the YAML file to test
        oracle._create_yaml_input("CCO", yaml_path)
        if os.path.exists(yaml_path):
            print(f"   ✓ YAML file created successfully")
            # Clean up
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
        else:
            print(f"   ✗ YAML file not created")
            return False
    except Exception as e:
        print(f"   ✗ Default mode failed: {e}")
        return False
    
    # Test 2: Custom YAML file path
    print("   Testing custom YAML file path...")
    try:
        temp_dir = tempfile.mkdtemp()
        custom_yaml = os.path.join(temp_dir, "custom_input.yaml")
        
        config = {
            "boltz": {
                "yaml_file_path": custom_yaml
            }
        }
        oracle = BoltzOracle(target="test", config=config)
        yaml_path, work_dir, cleanup = oracle._get_yaml_file_path("CCO")
        
        if yaml_path == custom_yaml and work_dir == temp_dir and not cleanup:
            print(f"   ✓ Custom path mode working correctly")
            # Create the YAML file to test
            oracle._create_yaml_input("CCO", yaml_path)
            if os.path.exists(yaml_path):
                print(f"   ✓ Custom YAML file created successfully")
            else:
                print(f"   ✗ Custom YAML file not created")
                return False
        else:
            print(f"   ✗ Custom path mode parameters incorrect")
            return False
        
        # Clean up
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"   ✗ Custom path mode failed: {e}")
        return False
    
    # Test 3: Template directory
    print("   Testing template directory mode...")
    try:
        temp_dir = tempfile.mkdtemp()
        
        config = {
            "boltz": {
                "yaml_template_dir": temp_dir,
                "preserve_yaml_files": True
            }
        }
        oracle = BoltzOracle(target="test", config=config)
        yaml_path, work_dir, cleanup = oracle._get_yaml_file_path("CCO")
        
        if work_dir == temp_dir and not cleanup:
            print(f"   ✓ Template directory mode working correctly")
            # Create the YAML file to test
            oracle._create_yaml_input("CCO", yaml_path)
            if os.path.exists(yaml_path):
                print(f"   ✓ Template YAML file created successfully")
            else:
                print(f"   ✗ Template YAML file not created")
                return False
        else:
            print(f"   ✗ Template directory mode parameters incorrect")
            return False
        
        # Clean up
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"   ✗ Template directory mode failed: {e}")
        return False
    
    return True


def test_yaml_content():
    """Test YAML file content generation."""
    print("\n3. Testing YAML content generation...")
    
    try:
        oracle = BoltzOracle(target="test")
        
        # Create a temporary YAML file
        temp_dir = tempfile.mkdtemp()
        yaml_file = os.path.join(temp_dir, "test_input.yaml")
        
        # Test basic YAML creation
        oracle._create_yaml_input("CCO", yaml_file)
        
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as f:
                content = f.read()
            print(f"   ✓ YAML file created with content:")
            print("   " + "\n   ".join(content.split('\n')[:10]))  # Show first 10 lines
            
            # Check for required elements
            if "sequences:" in content and "protein:" in content and "ligand:" in content:
                print(f"   ✓ YAML contains required sequences")
            else:
                print(f"   ✗ YAML missing required sequences")
                return False
                
            if "smiles: CCO" in content:
                print(f"   ✓ YAML contains correct SMILES")
            else:
                print(f"   ✗ YAML missing or incorrect SMILES")
                return False
                
            if oracle.predict_affinity and "properties:" in content:
                print(f"   ✓ YAML contains affinity prediction configuration")
            else:
                print(f"   ✓ YAML without affinity prediction (as configured)")
        else:
            print(f"   ✗ YAML file not created")
            return False
        
        # Clean up
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"   ✗ YAML content test failed: {e}")
        return False


def test_sequence_loading():
    """Test protein sequence loading."""
    print("\n4. Testing protein sequence loading...")
    
    try:
        oracle = BoltzOracle(target="test")
        
        if oracle.protein_sequence:
            print(f"   ✓ Protein sequence loaded: {len(oracle.protein_sequence)} residues")
            print(f"   ✓ First 50 chars: {oracle.protein_sequence[:50]}...")
            return True
        else:
            print(f"   ✗ No protein sequence loaded")
            return False
            
    except Exception as e:
        print(f"   ✗ Sequence loading failed: {e}")
        return False


def test_config_integration():
    """Test configuration integration."""
    print("\n5. Testing configuration integration...")
    
    try:
        # Test with custom configuration
        config = {
            "boltz": {
                "model": "custom_model",
                "diffusion_samples": 10,
                "predict_affinity": False,
                "preserve_yaml_files": True,
                "yaml_template_dir": "/tmp/boltz_test"
            }
        }
        
        oracle = BoltzOracle(target="test", config=config)
        
        # Check that configuration was applied
        checks = [
            (oracle.model == "custom_model", "Model configuration"),
            (oracle.diffusion_samples == 10, "Diffusion samples configuration"),
            (oracle.predict_affinity == False, "Affinity prediction configuration"),
            (oracle.preserve_yaml_files == True, "Preserve YAML files configuration"),
            (oracle.yaml_template_dir == "/tmp/boltz_test", "YAML template directory configuration"),
        ]
        
        for check, name in checks:
            if check:
                print(f"   ✓ {name} applied correctly")
            else:
                print(f"   ✗ {name} not applied correctly")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ✗ Configuration integration failed: {e}")
        return False


def main():
    """Run all tests."""
    print("BoltzOracle Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_import,
        test_yaml_modes,
        test_yaml_content,
        test_sequence_loading,
        test_config_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"   Test failed!")
        except Exception as e:
            print(f"   Test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BoltzOracle is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
