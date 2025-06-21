#!/usr/bin/env python3
"""
Test both Vina and GNINA docking oracles with 7JVR
"""

import os
import sys
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from al_fep.utils.config import load_config
from al_fep.oracles.docking_oracle import DockingOracle


def test_vina_oracle():
    """Test the Vina docking oracle."""
    print("🧪 Testing Vina Docking Oracle")
    print("=" * 50)
    
    # Load config using the correct function
    config = load_config("config/targets/7jvr.yaml", "config/default.yaml")
    
    # Override to use Vina
    if "docking" not in config:
        config["docking"] = {}
    config["docking"]["engine"] = "vina"
    
    # Initialize oracle
    oracle = DockingOracle(target="7jvr", config=config)
    
    # Test molecules
    test_molecules = [
        ("Ethanol", "CCO"),
        ("Propranolol-like", "CC(C)NCC(COC1=CC=CC2=C1C=CN2)O"),
    ]
    
    print(f"✓ Oracle initialized: {oracle}")
    print(f"✓ Engine: {oracle.engine.upper()}")
    print(f"✓ Receptor: {oracle.receptor_file}")
    
    results = []
    for name, smiles in test_molecules:
        print(f"\n🔬 Testing {name}: {smiles}")
        start_time = time.time()
        
        try:
            result = oracle.evaluate(smiles)
            end_time = time.time()
            
            if result.get('error'):
                print(f"   ❌ Error: {result['error']}")
            else:
                score = result.get('score', 'N/A')
                affinity = result.get('binding_affinity', 'N/A')
                print(f"   ✅ Success!")
                print(f"   - Score: {score}")
                print(f"   - Binding Affinity: {affinity} kcal/mol")
                print(f"   - Time: {end_time - start_time:.2f}s")
                
            results.append((name, result, end_time - start_time))
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append((name, {"error": str(e)}, 0))
    
    return results


def test_gnina_oracle():
    """Test the GNINA docking oracle."""
    print("\n🧪 Testing GNINA Docking Oracle")
    print("=" * 50)
    
    # Load config using the correct function
    config = load_config("config/targets/7jvr.yaml", "config/default.yaml")
    
    # Override to use GNINA
    if "docking" not in config:
        config["docking"] = {}
    config["docking"]["engine"] = "gnina"
    
    # Initialize oracle
    oracle = DockingOracle(target="7jvr", config=config)
    
    # Test molecules (start with simple one)
    test_molecules = [
        ("Ethanol", "CCO"),
    ]
    
    print(f"✓ Oracle initialized: {oracle}")
    print(f"✓ Engine: {oracle.engine.upper()}")
    print(f"✓ Receptor: {oracle.receptor_file}")
    
    results = []
    for name, smiles in test_molecules:
        print(f"\n🔬 Testing {name}: {smiles}")
        start_time = time.time()
        
        try:
            result = oracle.evaluate(smiles)
            end_time = time.time()
            
            if result.get('error'):
                print(f"   ❌ Error: {result['error']}")
            else:
                score = result.get('score', 'N/A')
                affinity = result.get('binding_affinity', 'N/A')
                print(f"   ✅ Success!")
                print(f"   - Score: {score}")
                print(f"   - Binding Affinity: {affinity} kcal/mol")
                print(f"   - Time: {end_time - start_time:.2f}s")
                
            results.append((name, result, end_time - start_time))
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append((name, {"error": str(e)}, 0))
    
    return results


def compare_results(vina_results, gnina_results):
    """Compare Vina and GNINA results."""
    print("\n📊 Comparison: Vina vs GNINA")
    print("=" * 50)
    
    print(f"{'Molecule':<15} {'Vina Score':<12} {'GNINA Score':<12} {'Vina Time':<10} {'GNINA Time':<12}")
    print("-" * 70)
    
    for i, (name, _, _) in enumerate(vina_results):
        if i < len(gnina_results):
            vina_result = vina_results[i][1]
            gnina_result = gnina_results[i][1]
            vina_time = vina_results[i][2]
            gnina_time = gnina_results[i][2]
            
            vina_score = vina_result.get('binding_affinity', 'Error')
            gnina_score = gnina_result.get('binding_affinity', 'Error')
            
            print(f"{name:<15} {str(vina_score):<12} {str(gnina_score):<12} {vina_time:<10.2f} {gnina_time:<12.2f}")


def main():
    """Main function to test both oracles."""
    print("🎯 Testing Docking Oracles: Vina vs GNINA")
    print("=" * 60)
    
    # Test Vina
    try:
        vina_results = test_vina_oracle()
    except Exception as e:
        print(f"❌ Vina test failed: {e}")
        vina_results = []
    
    # Test GNINA
    try:
        gnina_results = test_gnina_oracle()
    except Exception as e:
        print(f"❌ GNINA test failed: {e}")
        gnina_results = []
    
    # Compare results
    if vina_results and gnina_results:
        compare_results(vina_results, gnina_results)
    
    print(f"\n🎉 Testing completed!")
    if vina_results:
        successful_vina = sum(1 for _, result, _ in vina_results if not result.get('error'))
        print(f"✅ Vina: {successful_vina}/{len(vina_results)} successful")
    
    if gnina_results:
        successful_gnina = sum(1 for _, result, _ in gnina_results if not result.get('error'))
        print(f"✅ GNINA: {successful_gnina}/{len(gnina_results)} successful")


if __name__ == "__main__":
    main()