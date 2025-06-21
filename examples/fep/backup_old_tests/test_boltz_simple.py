#!/usr/bin/env python3
"""
Simple Boltz Oracle test
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def main():
    try:
        from al_fep.oracles.boltz_oracle import BoltzOracle
        print("✅ BoltzOracle import successful")
        
        # Create a simple config
        config = {
            'boltz': {
                'model': 'boltz2',
                'diffusion_samples': 1,
                'predict_affinity': True,
                'use_msa_server': False,
                'work_dir': 'temp/boltz_test'
            }
        }
        
        # Create oracle
        oracle = BoltzOracle(target='test', config=config)
        print(f"✅ Oracle created: {oracle}")
        print(f"   Model: {oracle.model}")
        print(f"   Samples: {oracle.diffusion_samples}")
        print(f"   Predict affinity: {oracle.predict_affinity}")
        print(f"   Work dir: {oracle.work_dir}")
        print(f"   Protein sequence: {len(oracle.protein_sequence)} residues")
        
        # Get statistics
        stats = oracle.get_statistics()
        print(f"✅ Statistics: {stats}")
        
        print("🎉 Boltz Oracle implementation is working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
