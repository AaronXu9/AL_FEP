#!/usr/bin/env python3
"""
Protein preparation for FEP calculations
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def prepare_protein_for_fep(input_pdb: str, output_pdb: str):
    """
    Prepare protein structure for FEP calculations by adding hydrogens.
    
    Args:
        input_pdb: Input PDB file (e.g., from X-ray structure)
        output_pdb: Output PDB file with hydrogens added
    """
    try:
        import openmm.app as app
        from openmm import unit
        
        print(f"🧬 Preparing protein: {input_pdb}")
        
        # Load the original PDB
        print("  Loading original PDB structure...")
        pdb = app.PDBFile(input_pdb)
        print(f"  ✓ Loaded {len(list(pdb.topology.atoms()))} atoms")
        
        # Create force field 
        print("  Setting up force field...")
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        
        # Use Modeller to add hydrogens
        print("  Adding hydrogen atoms...")
        from openmm.app import Modeller
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # Add missing hydrogens
        modeller.addHydrogens(forcefield)
        print(f"  ✓ Added hydrogens - now {len(list(modeller.topology.atoms()))} atoms")
        
        # Save the prepared structure
        print(f"  Saving prepared structure to: {output_pdb}")
        with open(output_pdb, 'w') as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
        
        print("  ✅ Protein preparation completed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Protein preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_prepared_protein(pdb_file: str):
    """Validate that the prepared protein is suitable for FEP."""
    try:
        import openmm.app as app
        
        print(f"🔍 Validating prepared protein: {pdb_file}")
        
        # Load PDB
        pdb = app.PDBFile(pdb_file)
        
        # Count atoms and residues
        atoms = list(pdb.topology.atoms())
        residues = list(pdb.topology.residues())
        
        print(f"  Structure stats:")
        print(f"    Atoms: {len(atoms)}")
        print(f"    Residues: {len(residues)}")
        
        # Check for hydrogen atoms
        hydrogen_count = sum(1 for atom in atoms if atom.element.symbol == 'H')
        print(f"    Hydrogen atoms: {hydrogen_count}")
        
        if hydrogen_count == 0:
            print("  ❌ No hydrogen atoms found!")
            return False
        
        # Try to create a system (this will fail if residues are not recognized)
        print("  Testing force field compatibility...")
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        
        try:
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.NoCutoff,  # Simple test
                constraints=None
            )
            print("  ✅ Force field compatible!")
            return True
            
        except Exception as e:
            print(f"  ❌ Force field compatibility failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Main protein preparation workflow."""
    print("🧪 Protein Preparation for FEP")
    print("=" * 40)
    
    # Define file paths
    input_pdb = "data/targets/7jvr/7jvr_system.pdb"
    output_pdb = "data/targets/7jvr/7jvr_system_prepared.pdb"
    
    # Check if input exists
    if not os.path.exists(input_pdb):
        print(f"❌ Input file not found: {input_pdb}")
        return
    
    # Prepare protein
    success = prepare_protein_for_fep(input_pdb, output_pdb)
    
    if success:
        # Validate the prepared protein
        if validate_prepared_protein(output_pdb):
            print(f"\n🎉 Protein preparation successful!")
            print(f"Use this file for FEP calculations: {output_pdb}")
        else:
            print(f"\n❌ Prepared protein failed validation")
    else:
        print(f"\n❌ Protein preparation failed")

if __name__ == "__main__":
    main()
