#!/usr/bin/env python3
"""
Prepare 7JVR receptor for AutoDock Vina docking.
This script converts the PDB file to PDBQT format required by Vina.
"""

import os
import subprocess
import sys
from pathlib import Path


def prepare_receptor(pdb_file, output_file, remove_waters=True, add_hydrogens=True):
    """
    Prepare receptor PDB file for docking.
    
    Args:
        pdb_file: Input PDB file path
        output_file: Output PDBQT file path
        remove_waters: Remove water molecules
        add_hydrogens: Add polar hydrogens
    """
    print(f"Preparing receptor: {pdb_file}")
    print(f"Output file: {output_file}")
    
    # Ensure input file exists
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Clean the PDB file and convert to PDBQT
    cmd = [
        "obabel", 
        str(pdb_file), 
        "-O", str(output_file),
        "-p"  # Add hydrogens
    ]
    
    if remove_waters:
        cmd.extend(["-d"])  # Remove all hydrogens first, then add back polar ones
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Receptor preparation successful!")
        print(f"Output saved to: {output_file}")
        
        # Check if file was created and has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"✓ Output file size: {os.path.getsize(output_file)} bytes")
            return True
        else:
            print("✗ Output file is empty or not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Error preparing receptor: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def validate_receptor(pdbqt_file):
    """Validate the prepared PDBQT file."""
    if not os.path.exists(pdbqt_file):
        return False
    
    with open(pdbqt_file, 'r') as f:
        content = f.read()
    
    # Check for essential PDBQT elements
    has_atoms = "ATOM" in content or "HETATM" in content
    has_end = "END" in content or "ENDMDL" in content
    
    if has_atoms:
        print("✓ PDBQT file contains atom records")
    else:
        print("✗ PDBQT file missing atom records")
    
    # Count atoms
    atom_lines = [line for line in content.split('\n') if line.startswith(('ATOM', 'HETATM'))]
    print(f"✓ Found {len(atom_lines)} atom records")
    
    return has_atoms and len(atom_lines) > 0


def main():
    """Main function to prepare 7JVR receptor."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    pdb_file = project_root / "data" / "targets" / "7jvr" / "7JVR.pdb"
    pdbqt_file = project_root / "data" / "targets" / "7jvr" / "7jvr_prepared.pdbqt"
    
    print("🔬 Preparing 7JVR Receptor for AutoDock Vina")
    print("=" * 50)
    
    # Check if input file exists
    if not pdb_file.exists():
        print(f"✗ PDB file not found: {pdb_file}")
        sys.exit(1)
    
    print(f"📁 Input PDB: {pdb_file}")
    print(f"📁 Output PDBQT: {pdbqt_file}")
    
    # Prepare receptor
    success = prepare_receptor(str(pdb_file), str(pdbqt_file))
    
    if success:
        # Validate the output
        print("\n🔍 Validating prepared receptor...")
        if validate_receptor(str(pdbqt_file)):
            print("\n✅ Receptor preparation completed successfully!")
            print(f"🎯 Ready for docking with: {pdbqt_file}")
        else:
            print("\n❌ Receptor validation failed!")
            sys.exit(1)
    else:
        print("\n❌ Receptor preparation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
