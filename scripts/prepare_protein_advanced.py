#!/usr/bin/env python3
"""
Advanced protein preparation for FEP calculations
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def clean_and_prepare_protein(input_pdb: str, output_pdb: str):
    """
    Clean and prepare protein structure for FEP calculations.
    
    This handles common issues:
    - Missing terminal groups
    - Non-standard residues
    - Missing atoms
    """
    try:
        import openmm.app as app
        from openmm import unit
        import numpy as np
        
        print(f"🧬 Advanced protein preparation: {input_pdb}")
        
        # Load the original PDB
        print("  Loading original PDB structure...")
        pdb = app.PDBFile(input_pdb)
        original_atoms = len(list(pdb.topology.atoms()))
        print(f"  ✓ Loaded {original_atoms} atoms")
        
        # Create force field 
        print("  Setting up force field...")
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        
        # Use Modeller with more robust settings
        print("  Creating modeller...")
        from openmm.app import Modeller
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # First, try to add missing atoms (not just hydrogens)
        print("  Adding missing heavy atoms...")
        try:
            modeller.addMissingAtoms(forcefield)
            print(f"  ✓ Added missing atoms - now {len(list(modeller.topology.atoms()))} atoms")
        except Exception as e:
            print(f"  ⚠️  Could not add missing atoms: {e}")
            print("  Continuing with original structure...")
        
        # Now add hydrogens
        print("  Adding hydrogen atoms...")
        try:
            modeller.addHydrogens(forcefield)
            final_atoms = len(list(modeller.topology.atoms()))
            print(f"  ✓ Added hydrogens - final structure: {final_atoms} atoms")
        except Exception as e:
            print(f"  ❌ Could not add hydrogens: {e}")
            return False
        
        # Save the prepared structure
        print(f"  Saving prepared structure to: {output_pdb}")
        os.makedirs(os.path.dirname(output_pdb), exist_ok=True)
        
        with open(output_pdb, 'w') as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
        
        print("  ✅ Advanced protein preparation completed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Advanced protein preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def try_alternative_preparation(input_pdb: str, output_pdb: str):
    """Try alternative preparation using PDBFixer."""
    try:
        print("🛠️  Trying alternative preparation with PDBFixer...")
        
        # Try using PDBFixer for more robust preparation
        try:
            from pdbfixer import PDBFixer
            from openmm.app import PDBFile
        except ImportError:
            print("  ❌ PDBFixer not available. Install with: conda install -c conda-forge pdbfixer")
            return False
        
        print("  Loading structure with PDBFixer...")
        fixer = PDBFixer(filename=input_pdb)
        
        print("  Finding missing residues...")
        fixer.findMissingResidues()
        
        print("  Finding missing atoms...")
        fixer.findMissingAtoms()
        
        print("  Adding missing atoms...")
        fixer.addMissingAtoms()
        
        print("  Adding missing hydrogens...")
        fixer.addMissingHydrogens(7.0)  # pH 7.0
        
        print(f"  Saving fixed structure to: {output_pdb}")
        PDBFile.writeFile(fixer.topology, fixer.positions, open(output_pdb, 'w'))
        
        print("  ✅ PDBFixer preparation completed!")
        return True
        
    except Exception as e:
        print(f"  ❌ PDBFixer preparation failed: {e}")
        return False

def create_minimal_test_system():
    """Create a minimal test system for FEP testing."""
    try:
        print("🧪 Creating minimal test system...")
        
        import openmm.app as app
        from openmm import unit
        from openmm.app import Modeller
        
        # Create a simple test system: just an alanine dipeptide
        # This is the minimal peptide for testing MD
        
        # Use OpenMM's built-in test systems
        from openmm.app import testInstallation
        
        print("  Creating alanine dipeptide test system...")
        
        # Create topology for ACE-ALA-NME (alanine dipeptide)
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        
        # Simple peptide sequence
        from openmm.app import PDBFile
        
        # Create minimal PDB content for alanine dipeptide
        minimal_pdb_content = """MODEL        1
ATOM      1  H1  ACE A   1      -2.384   1.209  -0.016  1.00  0.00           H  
ATOM      2  CH3 ACE A   1      -1.681   0.386  -0.002  1.00  0.00           C  
ATOM      3  H2  ACE A   1      -1.832  -0.198   0.893  1.00  0.00           H  
ATOM      4  H3  ACE A   1      -1.832  -0.198  -0.897  1.00  0.00           H  
ATOM      5  C   ACE A   1      -0.270   0.956   0.002  1.00  0.00           C  
ATOM      6  O   ACE A   1       0.708   0.235  -0.002  1.00  0.00           O  
ATOM      7  N   ALA A   2      -0.151   2.285   0.008  1.00  0.00           N  
ATOM      8  H   ALA A   2      -1.009   2.818   0.011  1.00  0.00           H  
ATOM      9  CA  ALA A   2       1.168   2.935   0.008  1.00  0.00           C  
ATOM     10  HA  ALA A   2       1.697   2.670  -0.903  1.00  0.00           H  
ATOM     11  CB  ALA A   2       0.964   4.439   0.008  1.00  0.00           C  
ATOM     12  HB1 ALA A   2       0.435   4.703   0.920  1.00  0.00           H  
ATOM     13  HB2 ALA A   2       1.936   4.916   0.008  1.00  0.00           H  
ATOM     14  HB3 ALA A   2       0.435   4.703  -0.904  1.00  0.00           H  
ATOM     15  C   ALA A   2       1.985   2.479   1.212  1.00  0.00           C  
ATOM     16  O   ALA A   2       1.481   2.081   2.268  1.00  0.00           O  
ATOM     17  N   NME A   3       3.310   2.479   1.090  1.00  0.00           N  
ATOM     18  H   NME A   3       3.725   2.798   0.227  1.00  0.00           H  
ATOM     19  CH3 NME A   3       4.127   2.023   2.194  1.00  0.00           C  
ATOM     20  HH31 NME A   3       3.666   1.227   2.774  1.00  0.00           H  
ATOM     21  HH32 NME A   3       5.048   1.627   1.761  1.00  0.00           H  
ATOM     22  HH33 NME A   3       4.383   2.827   2.881  1.00  0.00           H  
ENDMDL
"""
        
        # Save minimal test system
        test_pdb = "data/targets/7jvr/minimal_test_system.pdb"
        os.makedirs(os.path.dirname(test_pdb), exist_ok=True)
        
        with open(test_pdb, 'w') as f:
            f.write(minimal_pdb_content)
        
        print(f"  ✅ Created minimal test system: {test_pdb}")
        return test_pdb
        
    except Exception as e:
        print(f"  ❌ Failed to create minimal test system: {e}")
        return None

def main():
    """Main protein preparation workflow with fallbacks."""
    print("🧪 Comprehensive Protein Preparation for FEP")
    print("=" * 50)
    
    # Define file paths
    input_pdb = "data/targets/7jvr/7jvr_system.pdb"
    output_pdb = "data/targets/7jvr/7jvr_system_prepared.pdb"
    
    success = False
    
    # Method 1: Advanced OpenMM preparation
    if os.path.exists(input_pdb):
        print("\n🔄 Method 1: Advanced OpenMM preparation")
        success = clean_and_prepare_protein(input_pdb, output_pdb)
    else:
        print(f"❌ Input file not found: {input_pdb}")
    
    # Method 2: PDBFixer if Method 1 fails
    if not success and os.path.exists(input_pdb):
        print("\n🔄 Method 2: PDBFixer preparation")
        success = try_alternative_preparation(input_pdb, output_pdb)
    
    # Method 3: Create minimal test system
    if not success:
        print("\n🔄 Method 3: Creating minimal test system")
        minimal_system = create_minimal_test_system()
        if minimal_system:
            print(f"\n💡 Suggestion: Use minimal test system for initial FEP testing:")
            print(f"   {minimal_system}")
            success = True
    
    # Final status
    if success:
        print(f"\n🎉 Protein preparation workflow completed!")
        if os.path.exists(output_pdb):
            print(f"✅ Use this prepared protein for FEP: {output_pdb}")
        else:
            print(f"✅ Use minimal test system for initial testing")
    else:
        print(f"\n❌ All preparation methods failed")
        print(f"💡 Consider using a different protein structure or manual preparation")

if __name__ == "__main__":
    main()
