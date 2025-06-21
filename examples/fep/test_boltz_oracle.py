#!/usr/bin/env python3
"""
Test script for Boltz Oracle
"""

import sys
import os
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_boltz_oracle_basic():
    """Test basic Boltz oracle functionality."""
    print("🧪 Basic Boltz Oracle Test")
    print("=" * 40)
    
    try:
        # Import
        from al_fep.oracles.boltz_oracle import BoltzOracle
        from al_fep.utils.config import load_config
        print("✓ Import successful")
        
        # Load configuration
        config_path = os.path.join(os.getcwd(), "config", "targets", "boltz_config.yaml")
        try:
            config = load_config(config_path)
            # Check if config actually loaded
            if config and 'boltz' in config:
                print("✓ Config loaded from file")
            else:
                # Config file exists but empty or missing boltz section
                raise KeyError("Config missing boltz section")
        except Exception:
            # Use default config if file not found or invalid
            config = {
                "boltz": {
                    "model": "boltz2",
                    "diffusion_samples": 1,
                    "use_msa_server": True,  # Enable MSA for testing
                    "predict_affinity": True,
                    "output_format": "pdb",
                    "work_dir": "temp/boltz_test"  # Use temp directory
                },
                "cache": True
            }
            print("✓ Using default config")
        
        print(f"  Model: {config['boltz']['model']}")
        print(f"  Samples: {config['boltz']['diffusion_samples']}")
        print(f"  Affinity prediction: {config['boltz']['predict_affinity']}")
        
        # Create oracle
        oracle = BoltzOracle(target="test", config=config)
        print(f"✓ Oracle created: {oracle}")
        
        # Test evaluation with a simple molecule
        test_smiles = "CCO"  # Ethanol
        print(f"\nTesting evaluation with {test_smiles}...")
        print("Note: This will take several minutes as Boltz runs ML prediction...")
        
        result = oracle.evaluate(test_smiles)
        print(f"✓ Evaluation completed")
        print(f"  Result type: {type(result)}")
        
        if isinstance(result, list) and len(result) > 0:
            result = result[0]  # Get first result from list
        
        if isinstance(result, dict):
            print(f"  Keys: {list(result.keys())}")
            if 'error' in result and result['error']:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  ✅ Success!")
                print(f"    Score: {result.get('score')}")
                print(f"    Binding Affinity: {result.get('binding_affinity')}")
                print(f"    Binding Probability: {result.get('binding_probability')}")
                print(f"    Confidence Score: {result.get('confidence_score')}")
                print(f"    Method: {result.get('method')}")
                if result.get('structure_file'):
                    print(f"    Structure file: {result.get('structure_file')}")
        else:
            print(f"  Unexpected result format: {result}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure Boltz-2 is installed: pip install boltz")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n🎉 Test completed!")
    return True

def test_boltz_oracle_config():
    """Test Boltz oracle configuration options."""
    print("\n🧪 Boltz Oracle Configuration Test")
    print("=" * 40)
    
    try:
        from al_fep.oracles.boltz_oracle import BoltzOracle
        
        # Test different configurations
        configs = [
            {
                "name": "Fast Mode",
                "config": {
                    "boltz": {
                        "diffusion_samples": 1,
                        "use_msa_server": True,
                        "predict_affinity": False
                    }
                }
            },
            {
                "name": "High Quality Mode", 
                "config": {
                    "boltz": {
                        "diffusion_samples": 5,
                        "recycling_steps": 5,
                        "use_potentials": True,
                        "predict_affinity": True,
                        "diffusion_samples_affinity": 10
                    }
                }
            }
        ]
        
        for test_config in configs:
            print(f"\n📋 Testing {test_config['name']}:")
            try:
                oracle = BoltzOracle(target="test", config=test_config["config"])
                print(f"  ✓ Oracle created: {oracle}")
                print(f"  ✓ Model: {oracle.model}")
                print(f"  ✓ Samples: {oracle.diffusion_samples}")
                print(f"  ✓ Affinity prediction: {oracle.predict_affinity}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Configuration test error: {e}")

def test_yaml_generation():
    """Test YAML file generation for Boltz."""
    print("\n🧪 YAML Generation Test")
    print("=" * 40)
    
    try:
        from al_fep.oracles.boltz_oracle import BoltzOracle
        import tempfile
        
        config = {
            "boltz": {
                "model": "boltz2",
                "predict_affinity": True,
                "work_dir": "data/boltz_workspace"
            }
        }
        
        oracle = BoltzOracle(target="test", config=config)
        
        # Test YAML creation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_file = oracle._create_yaml_input("CCO", f.name)
            
        print(f"✓ YAML file created: {yaml_file}")
        
        # Read and display the YAML content
        with open(yaml_file, 'r') as f:
            content = f.read()
            print("📄 YAML Content:")
            print(content)
        
        # Clean up
        os.unlink(yaml_file)
        print("✓ Temporary file cleaned up")
        
    except Exception as e:
        print(f"❌ YAML generation test error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Boltz Oracle Tests")
    print("=" * 50)
    
    # Run tests
    test_boltz_oracle_basic()
    test_boltz_oracle_config()
    test_yaml_generation()
    
    print("\n" + "=" * 50)
    print("🏁 All tests completed!")
