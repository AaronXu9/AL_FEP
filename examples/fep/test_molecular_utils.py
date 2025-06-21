#!/usr/bin/env python3
"""
Test molecular file utility functions with AL_FEP data files.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from al_fep.utils.molecular_file_utils import (
    extract_protein_sequence_from_pdb,
    extract_smiles_from_sdf,
    create_boltz_input_from_files,
    pdb_to_fasta,
    sdf_to_smiles_list,
    write_fasta_file
)


def test_pdb_sequence_extraction():
    """Test extracting protein sequences from PDB files."""
    print("🧬 Test 1: PDB Sequence Extraction")
    print("-" * 50)
    
    # Look for available PDB files in the data directory
    data_dir = project_root / "data"
    pdb_files = list(data_dir.glob("**/*.pdb"))
    
    if not pdb_files:
        print("   ⚠️  No PDB files found in data directory")
        return False
    
    for pdb_file in pdb_files[:2]:  # Test first 2 PDB files
        print(f"\n   📄 Processing: {pdb_file.name}")
        try:
            sequences = extract_protein_sequence_from_pdb(str(pdb_file))
            
            for chain_id, sequence in sequences.items():
                print(f"      Chain {chain_id}: {len(sequence)} residues")
                print(f"      First 50 residues: {sequence[:50]}...")
            
            # Create FASTA file
            fasta_file = project_root / "temp" / f"{pdb_file.stem}_sequences.fasta"
            fasta_file.parent.mkdir(exist_ok=True)
            
            write_fasta_file(sequences, str(fasta_file), f"from {pdb_file.name}")
            print(f"      ✅ FASTA created: {fasta_file.name}")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            return False
    
    return True


def test_sdf_smiles_extraction():
    """Test extracting SMILES from SDF files."""
    print("\n🧪 Test 2: SDF SMILES Extraction")
    print("-" * 50)
    
    # Look for available SDF files
    data_dir = project_root / "data"
    sdf_files = list(data_dir.glob("**/*.sdf"))
    
    if not sdf_files:
        print("   ⚠️  No SDF files found in data directory")
        return False
    
    for sdf_file in sdf_files[:2]:  # Test first 2 SDF files
        print(f"\n   📄 Processing: {sdf_file.name}")
        try:
            molecules = extract_smiles_from_sdf(str(sdf_file))
            
            print(f"      Found {len(molecules)} molecules")
            
            # Show first few molecules
            for i, mol in enumerate(molecules[:3]):
                name = mol.get('name', f'mol_{i+1}')
                smiles = mol['smiles']
                print(f"      {i+1}. {name}: {smiles}")
                
                # Show additional properties if available
                other_props = {k: v for k, v in mol.items() if k not in ['name', 'smiles']}
                if other_props:
                    print(f"         Properties: {list(other_props.keys())}")
            
            if len(molecules) > 3:
                print(f"      ... and {len(molecules) - 3} more molecules")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            return False
    
    return True


def test_boltz_yaml_generation():
    """Test creating Boltz YAML files from PDB and SDF files."""
    print("\n⚙️  Test 3: Boltz YAML Generation")
    print("-" * 50)
    
    # Find PDB and SDF files
    data_dir = project_root / "data"
    pdb_files = list(data_dir.glob("**/*.pdb"))
    sdf_files = list(data_dir.glob("**/*.sdf"))
    
    if not pdb_files or not sdf_files:
        print("   ⚠️  Need both PDB and SDF files for this test")
        print(f"      PDB files found: {len(pdb_files)}")
        print(f"      SDF files found: {len(sdf_files)}")
        return False
    
    # Use first available PDB and SDF files
    pdb_file = pdb_files[0]
    sdf_file = sdf_files[0]
    
    print(f"   📄 PDB: {pdb_file.name}")
    print(f"   📄 SDF: {sdf_file.name}")
    
    try:
        # Create output directory
        output_dir = project_root / "temp" / "boltz_inputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate YAML file
        yaml_file = output_dir / f"{pdb_file.stem}_{sdf_file.stem}_input.yaml"
        
        created_yaml = create_boltz_input_from_files(
            str(pdb_file),
            str(sdf_file),
            str(yaml_file),
            molecule_index=0
        )
        
        print(f"   ✅ YAML created: {yaml_file.name}")
        
        # Show YAML content
        with open(created_yaml, 'r') as f:
            content = f.read()
        
        print(f"   📋 YAML content preview:")
        lines = content.split('\n')
        for line in lines[:15]:  # Show first 15 lines
            print(f"      {line}")
        
        if len(lines) > 15:
            print(f"      ... ({len(lines) - 15} more lines)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_convenience_functions():
    """Test convenience functions."""
    print("\n🛠️  Test 4: Convenience Functions")
    print("-" * 50)
    
    data_dir = project_root / "data"
    pdb_files = list(data_dir.glob("**/*.pdb"))
    sdf_files = list(data_dir.glob("**/*.sdf"))
    
    if pdb_files:
        pdb_file = pdb_files[0]
        print(f"   📄 Testing pdb_to_fasta with: {pdb_file.name}")
        
        try:
            temp_dir = project_root / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            fasta_file = temp_dir / f"{pdb_file.stem}_conv.fasta"
            pdb_to_fasta(str(pdb_file), str(fasta_file))
            
            print(f"      ✅ FASTA conversion successful: {fasta_file.name}")
            
        except Exception as e:
            print(f"      ❌ pdb_to_fasta failed: {e}")
            return False
    
    if sdf_files:
        sdf_file = sdf_files[0]
        print(f"   📄 Testing sdf_to_smiles_list with: {sdf_file.name}")
        
        try:
            smiles_list = sdf_to_smiles_list(str(sdf_file))
            print(f"      ✅ Extracted {len(smiles_list)} SMILES strings")
            
            # Show first few SMILES
            for i, smiles in enumerate(smiles_list[:3]):
                print(f"         {i+1}. {smiles}")
            
        except Exception as e:
            print(f"      ❌ sdf_to_smiles_list failed: {e}")
            return False
    
    return True


def show_data_directory_structure():
    """Show what molecular files are available in the data directory."""
    print("\n📁 Available Data Files")
    print("-" * 50)
    
    data_dir = project_root / "data"
    
    if not data_dir.exists():
        print("   ❌ Data directory not found")
        return
    
    # Find molecular files
    file_types = {
        "PDB files": "**/*.pdb",
        "SDF files": "**/*.sdf", 
        "FASTA files": "**/*.fasta",
        "MOL files": "**/*.mol",
        "MOL2 files": "**/*.mol2"
    }
    
    for file_type, pattern in file_types.items():
        files = list(data_dir.glob(pattern))
        print(f"   {file_type}: {len(files)} found")
        
        for file_path in files[:3]:  # Show first 3 files
            rel_path = file_path.relative_to(project_root)
            print(f"      - {rel_path}")
        
        if len(files) > 3:
            print(f"      ... and {len(files) - 3} more")


def main():
    """Run all tests."""
    print("🧬 Molecular File Utilities Test Suite")
    print("=" * 70)
    
    # Show available data files first
    show_data_directory_structure()
    
    # Run tests
    tests = [
        test_pdb_sequence_extraction,
        test_sdf_smiles_extraction,
        test_boltz_yaml_generation,
        test_convenience_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("   ❌ Test failed")
        except Exception as e:
            print(f"   💥 Test crashed: {e}")
    
    print(f"\n{'='*70}")
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All utility functions working correctly!")
        print("\n💡 Usage examples:")
        print("   from al_fep.utils.molecular_file_utils import *")
        print("   sequences = extract_protein_sequence_from_pdb('protein.pdb')")
        print("   smiles_list = sdf_to_smiles_list('compounds.sdf')")
        print("   create_boltz_input_from_files('protein.pdb', 'ligands.sdf', 'input.yaml')")
    else:
        print("❌ Some tests failed. Check the output above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
