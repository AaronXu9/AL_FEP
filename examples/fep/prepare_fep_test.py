#!/usr/bin/env python3
"""
Prepare data for real FEP testing
"""

import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def prepare_protein_for_fep():
    """Prepare protein structure for FEP calculations."""
    print("🔧 Preparing Protein Structure for FEP")
    print("=" * 40)
    
    # Source and target paths
    source_pdb = "data/targets/7jvr/7JVR.pdb"
    target_dir = "data/targets/7jvr"
    target_pdb = os.path.join(target_dir, "7jvr_system.pdb")
    
    if not os.path.exists(source_pdb):
        print(f"❌ Source PDB not found: {source_pdb}")
        return False
    
    try:
        # For FEP, we need a clean protein structure
        # The 7JVR.pdb should work, but let's clean it up
        
        print(f"Processing {source_pdb}...")
        
        with open(source_pdb, 'r') as f:
            lines = f.readlines()
        
        # Clean PDB: keep only ATOM records for protein
        clean_lines = []
        for line in lines:
            if line.startswith('ATOM'):
                # Keep protein atoms (not water, ions, etc.)
                if line[17:20].strip() in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                                          'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                                          'THR', 'TRP', 'TYR', 'VAL']:
                    clean_lines.append(line)
            elif line.startswith(('HEADER', 'TITLE', 'COMPND', 'SOURCE', 'REMARK')):
                clean_lines.append(line)
        
        # Add END record
        clean_lines.append('END\n')
        
        # Write cleaned PDB
        with open(target_pdb, 'w') as f:
            f.writelines(clean_lines)
        
        print(f"✅ Clean protein structure saved: {target_pdb}")
        
        # Check the structure
        atom_count = sum(1 for line in clean_lines if line.startswith('ATOM'))
        print(f"   Protein atoms: {atom_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to prepare protein: {e}")
        return False

def create_test_config():
    """Create FEP test configuration."""
    print("\n🔧 Creating FEP Test Configuration")
    print("=" * 40)
    
    config = {
        "fep": {
            "mock_mode": False,  # Real FEP calculations
            "force_field": "amber14",
            "water_model": "tip3p",
            "num_lambda_windows": 5,  # Reduced for testing (normally 11-21)
            "simulation_time": 0.01,  # Very short: 10 ps for testing (normally 1-5 ns)
            "temperature": 298.15,
            "pressure": 1.0,
            "equilibration_steps": 100,  # Very short (normally 50,000)
            "production_steps": 500,     # Very short (normally 250,000+)
            "output_frequency": 50       # Save every 50 steps
        }
    }
    
    print("Test configuration:")
    for key, value in config["fep"].items():
        print(f"  {key}: {value}")
    
    return config

def check_dependencies():
    """Check if all required dependencies are available."""
    print("\n🔍 Checking Dependencies")
    print("=" * 40)
    
    missing_deps = []
    
    # Check OpenMM
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        print(f"✅ OpenMM {openmm.version.version}")
    except ImportError:
        print("❌ OpenMM not found")
        missing_deps.append("openmm")
    
    # Check RDKit
    try:
        from rdkit import Chem
        print("✅ RDKit")
    except ImportError:
        print("❌ RDKit not found")
        missing_deps.append("rdkit")
    
    # Check if we can create OpenMM systems
    try:
        platform_names = []
        for i in range(openmm.Platform.getNumPlatforms()):
            platform_names.append(openmm.Platform.getPlatform(i).getName())
        print(f"✅ OpenMM Platforms: {', '.join(platform_names)}")
        
        # Check for GPU acceleration
        if 'CUDA' in platform_names:
            print("✅ CUDA GPU acceleration available")
        elif 'OpenCL' in platform_names:
            print("✅ OpenCL GPU acceleration available")
        else:
            print("⚠️  Only CPU platform available (will be slow)")
            
    except Exception as e:
        print(f"❌ OpenMM platform check failed: {e}")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with:")
        for dep in missing_deps:
            if dep == "openmm":
                print("  conda install -c conda-forge openmm")
            elif dep == "rdkit":
                print("  conda install -c conda-forge rdkit")
        return False
    
    return True

def estimate_computation_time():
    """Estimate computation time for FEP test."""
    print("\n⏱️  Computation Time Estimate")
    print("=" * 40)
    
    # Test configuration
    lambda_windows = 5
    production_steps = 500
    
    print(f"Test configuration:")
    print(f"  Lambda windows: {lambda_windows}")
    print(f"  Steps per window: {production_steps}")
    print(f"  Total MD steps: {lambda_windows * production_steps * 2} (complex + solvent)")
    
    print(f"\nEstimated time:")
    print(f"  CPU: ~5-10 minutes")
    print(f"  GPU: ~1-2 minutes")
    print(f"  (Real production FEP: 2-24 hours)")

def main():
    """Main preparation function."""
    print("🧪 FEP Test Data Preparation")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Cannot proceed without required dependencies")
        return False
    
    # Prepare protein structure
    if not prepare_protein_for_fep():
        print("\n❌ Protein preparation failed")
        return False
    
    # Create test configuration
    config = create_test_config()
    
    # Show computation estimates
    estimate_computation_time()
    
    print(f"\n🎯 Data Preparation Complete!")
    print(f"Ready to run real FEP test with:")
    print(f"  Protein: data/targets/7jvr/7jvr_system.pdb")
    print(f"  Configuration: Test settings (very fast)")
    print(f"  Next step: Run test_real_fep_prepared.py")
    
    return True, config

if __name__ == "__main__":
    success, config = main()
    
    if success:
        # Save config for the test
        import json
        with open("fep_test_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Test configuration saved to fep_test_config.json")
