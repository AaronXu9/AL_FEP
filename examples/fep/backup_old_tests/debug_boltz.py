#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join('..', '..', 'src'))

try:
    from al_fep.oracles.boltz_oracle import BoltzOracle
    print('✓ Import successful')
    
    config = {
        'boltz': {
            'model': 'boltz2',
            'diffusion_samples': 1,
            'use_msa_server': False,
            'predict_affinity': True,
            'output_format': 'pdb',
            'work_dir': 'temp/boltz_test'
        }
    }
    
    print('Creating oracle...')
    oracle = BoltzOracle(target='test', config=config)
    print('✓ Oracle created successfully!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
