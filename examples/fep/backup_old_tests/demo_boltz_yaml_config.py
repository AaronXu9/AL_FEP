#!/usr/bin/env python3
"""
Demonstration of Boltz Oracle YAML file path configuration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def demo_yaml_configurations():
    """Demonstrate different YAML file path configurations."""
    print("🧪 Boltz Oracle YAML Configuration Demo")
    print("=" * 50)
    
    from al_fep.oracles.boltz_oracle import BoltzOracle
    
    # Demo 1: Default behavior (temporary files)
    print("\n📁 Demo 1: Default Configuration (Temporary Files)")
    config1 = {
        'boltz': {
            'diffusion_samples': 1,
            'use_msa_server': False,
            'predict_affinity': True
        }
    }
    
    oracle1 = BoltzOracle(target='test', config=config1)  # Use 'test' target which has default sequence
    yaml_file1, work_dir1, cleanup1 = oracle1._get_yaml_file_path('CCO')
    print(f"  YAML file: {yaml_file1}")
    print(f"  Work dir: {work_dir1}")
    print(f"  Will cleanup: {cleanup1}")
    
    # Demo 2: Custom YAML file path
    print("\n📁 Demo 2: Custom YAML File Path")
    config2 = {
        'boltz': {
            'yaml_file_path': '/tmp/demo_custom_input.yaml',
            'diffusion_samples': 1,
            'use_msa_server': False,
            'predict_affinity': True
        }
    }
    
    oracle2 = BoltzOracle(target='test', config=config2)
    yaml_file2, work_dir2, cleanup2 = oracle2._get_yaml_file_path('CCO')
    print(f"  YAML file: {yaml_file2}")
    print(f"  Work dir: {work_dir2}")
    print(f"  Will cleanup: {cleanup2}")
    
    # Demo 3: Template directory with preserved files
    print("\n📁 Demo 3: Template Directory (Preserved Files)")
    config3 = {
        'boltz': {
            'yaml_template_dir': '/tmp/demo_boltz_templates',
            'preserve_yaml_files': True,
            'diffusion_samples': 1,
            'use_msa_server': False,
            'predict_affinity': True
        }
    }
    
    oracle3 = BoltzOracle(target='test', config=config3)
    yaml_file3, work_dir3, cleanup3 = oracle3._get_yaml_file_path('CCO')
    print(f"  YAML file: {yaml_file3}")
    print(f"  Work dir: {work_dir3}")
    print(f"  Will cleanup: {cleanup3}")
    
    # Demo 4: Template directory with cleanup
    print("\n📁 Demo 4: Template Directory (Temporary Files)")
    config4 = {
        'boltz': {
            'yaml_template_dir': '/tmp/demo_boltz_temp',
            'preserve_yaml_files': False,  # Will be cleaned up
            'diffusion_samples': 1,
            'use_msa_server': False,
            'predict_affinity': True
        }
    }
    
    oracle4 = BoltzOracle(target='test', config=config4)
    yaml_file4, work_dir4, cleanup4 = oracle4._get_yaml_file_path('CCO')
    print(f"  YAML file: {yaml_file4}")
    print(f"  Work dir: {work_dir4}")
    print(f"  Will cleanup: {cleanup4}")
    
    # Actually create and show YAML content
    print("\n🧬 Creating Sample YAML Files")
    print("-" * 30)
    
    for i, (oracle, yaml_file) in enumerate([(oracle2, yaml_file2), (oracle3, yaml_file3)], 1):
        oracle._create_yaml_input('CCO', yaml_file)
        print(f"✅ Created YAML file {i}: {yaml_file}")
        
        # Show file size
        if os.path.exists(yaml_file):
            size = os.path.getsize(yaml_file)
            print(f"   File size: {size} bytes")
            
            # Show first few lines
            with open(yaml_file, 'r') as f:
                lines = f.readlines()[:5]
                print(f"   First lines:")
                for line in lines:
                    print(f"     {line.strip()}")
    
    print("\n🎉 YAML Configuration Demo Complete!")
    print("\n📋 Configuration Options Summary:")
    print("   1. Default: Uses temporary directories (auto-cleanup)")
    print("   2. yaml_file_path: Use specific file path (no cleanup)")
    print("   3. yaml_template_dir + preserve_yaml_files=True: Use template dir (no cleanup)")
    print("   4. yaml_template_dir + preserve_yaml_files=False: Use template dir (cleanup)")

if __name__ == "__main__":
    demo_yaml_configurations()
