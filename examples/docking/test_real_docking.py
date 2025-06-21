#!/usr/bin/env python3
"""
Test Real Docking Oracle with 7JVR
This script tests the AutoDock Vina integration with the prepared 7JVR receptor.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from al_fep import DockingOracle, setup_logging, load_config


def test_real_docking():
    """Test real docking oracle with 7JVR."""
    
    setup_logging(level="INFO")
    
    print("🎯 Testing Real Docking Oracle with 7JVR")
    print("=" * 50)
    
    # Load configuration
    config_dir = Path('config')
    config = load_config(
        config_dir / 'targets' / '7jvr.yaml',
        config_dir / 'default.yaml'
    )
    
    print(f"✓ Configuration loaded")
    print(f"  - Target: {config['target_info']['name']}")
    print(f"  - Receptor: {config['docking']['receptor_file']}")
    print(f"  - Mock mode: {config['docking'].get('mock_mode', True)}")
    
    # Check if receptor file exists
    receptor_file = Path(config['docking']['receptor_file'])
    if not receptor_file.exists():
        print(f"❌ Receptor file not found: {receptor_file}")
        return False
    
    print(f"✓ Receptor file found: {receptor_file}")
    print(f"  - File size: {receptor_file.stat().st_size / 1024:.1f} KB")
    
    # Initialize docking oracle
    print("\n🔬 Initializing Docking Oracle...")
    try:
        docking_oracle = DockingOracle(target="7jvr", config=config)
        print(f"✓ Docking oracle initialized: {docking_oracle}")
    except Exception as e:
        print(f"❌ Failed to initialize docking oracle: {e}")
        return False
    
    # Test molecules (start with simple ones)
    test_molecules = [
        "CCO",  # Ethanol (simple test)
        "CC(C)NCC(COC1=CC=CC2=C1C=CN2)O",  # Propranolol-like
        "CC1(C)SC2C(NC(=O)C(NC(=O)OC(C)(C)C)C3CCC4=C(c5ccccc5)C3N4)C(=O)N2C1C(=O)O",  # Nirmatrelvir
    ]
    
    print(f"\n⚗️  Testing {len(test_molecules)} molecules:")
    
    results = []
    for i, smiles in enumerate(test_molecules, 1):
        mol_name = ["Ethanol", "Propranolol-like", "Nirmatrelvir"][i-1]
        print(f"\n{i}. Testing {mol_name}: {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
        
        try:
            result = docking_oracle.evaluate(smiles)
            
            if result.get('error'):
                print(f"   ❌ Error: {result['error']}")
                results.append({'molecule': mol_name, 'success': False, 'error': result['error']})
            else:
                score = result.get('score', 'N/A')
                affinity = result.get('binding_affinity', 'N/A')
                method = result.get('method', 'N/A')
                
                print(f"   ✓ Success!")
                print(f"   - Score: {score}")
                print(f"   - Binding Affinity: {affinity} kcal/mol")
                print(f"   - Method: {method}")
                
                results.append({
                    'molecule': mol_name,
                    'success': True,
                    'score': score,
                    'affinity': affinity,
                    'method': method
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({'molecule': mol_name, 'success': False, 'error': str(e)})
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"=" * 30)
    successful = sum(1 for r in results if r['success'])
    print(f"✓ Successful dockings: {successful}/{len(results)}")
    print(f"❌ Failed dockings: {len(results) - successful}/{len(results)}")
    
    if successful > 0:
        print(f"\n🏆 Best scoring molecule:")
        successful_results = [r for r in results if r['success']]
        best = min(successful_results, key=lambda x: x['score'])
        print(f"  - Molecule: {best['molecule']}")
        print(f"  - Score: {best['score']}")
        print(f"  - Affinity: {best['affinity']} kcal/mol")
    
    # Oracle statistics
    print(f"\n📈 Oracle Statistics:")
    stats = docking_oracle.get_statistics()
    print(f"  - Total calls: {stats['call_count']}")
    print(f"  - Average time: {stats.get('average_time', 0.0):.2f}s")
    
    return successful > 0


def main():
    """Main function."""
    # Check if we're in the right environment
    try:
        import subprocess
        result = subprocess.run(['which', 'vina'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ AutoDock Vina not found in PATH")
            print("Please ensure you're in the PoseBench environment")
            return
        else:
            print(f"✓ AutoDock Vina found: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Error checking Vina: {e}")
        return
    
    # Run the test
    success = test_real_docking()
    
    if success:
        print(f"\n🎉 Real docking oracle test completed successfully!")
        print(f"✅ Ready for production use with AutoDock Vina")
    else:
        print(f"\n❌ Real docking oracle test failed")
        print(f"Please check the configuration and try again")


if __name__ == "__main__":
    main()
