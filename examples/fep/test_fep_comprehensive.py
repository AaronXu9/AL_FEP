#!/usr/bin/env python3
"""
Comprehensive FEP Oracle testing
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_fep_comprehensive():
    """Test FEP oracle with various molecules."""
    print("🧪 Comprehensive FEP Oracle Test")
    print("=" * 40)
    
    from al_fep.oracles.fep_oracle import FEPOracle
    
    # Create oracle
    config = {"fep": {"mock_mode": True}}
    oracle = FEPOracle(target="test", config=config)
    
    # Test various molecules
    test_molecules = [
        ("Ethanol", "CCO"),
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Propranolol", "CC(C)NCC(COC1=CC=CC2=C1C=CN2)O"),
        ("Water", "O"),
        ("Benzene", "C1=CC=CC=C1"),
        ("Invalid SMILES", "INVALID_MOLECULE"),
    ]
    
    print(f"Testing {len(test_molecules)} molecules...\n")
    
    results = []
    for name, smiles in test_molecules:
        print(f"🔬 {name}: {smiles}")
        
        try:
            result = oracle.evaluate(smiles)
            
            if result.get('error'):
                print(f"   ❌ Error: {result['error']}")
                results.append((name, "Error", result['error']))
            else:
                score = result.get('score', 'N/A')
                fep_score = result.get('fep_score', 'N/A')
                print(f"   ✅ Score: {score:.3f}, FEP: {fep_score:.3f} kcal/mol")
                results.append((name, "Success", fep_score))
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append((name, "Exception", str(e)))
    
    # Summary
    print(f"\n📊 Results Summary:")
    print("-" * 50)
    successful = sum(1 for _, status, _ in results if status == "Success")
    print(f"Successful evaluations: {successful}/{len(test_molecules)}")
    
    for name, status, value in results:
        if status == "Success":
            print(f"  ✅ {name:<15}: {value:.3f} kcal/mol")
        else:
            print(f"  ❌ {name:<15}: {status}")
    
    # Test caching
    print(f"\n🔄 Testing caching...")
    print("First evaluation:")
    result1 = oracle.evaluate("CCO")
    print("Second evaluation (should be cached):")
    result2 = oracle.evaluate("CCO")
    
    if result1 == result2:
        print("✅ Caching works correctly!")
    else:
        print("❌ Caching issue detected")
    
    # Check statistics
    stats = oracle.get_statistics()
    print(f"\n📈 Oracle Statistics:")
    print(f"  Total calls: {stats['call_count']}")
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Average time: {stats['avg_time']:.4f}s")
    
    print(f"\n🎉 Comprehensive test completed!")

if __name__ == "__main__":
    test_fep_comprehensive()
