"""
Molecular file utility functions for AL_FEP.

This module provides utilities for extracting molecular data from various file formats:
- Extract protein sequences from PDB files
- Extract ligand SMILES from SDF files
- Convert between molecular formats
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Amino acid one-letter to three-letter code mapping
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O'  # Non-standard amino acids
}


def extract_protein_sequence_from_pdb(
    pdb_file: str, 
    chain_id: Optional[str] = None,
    include_hetero: bool = False
) -> Dict[str, str]:
    """
    Extract protein sequence(s) from a PDB file.
    
    Args:
        pdb_file: Path to the PDB file
        chain_id: Specific chain ID to extract (if None, extract all chains)
        include_hetero: Whether to include hetero atoms/residues
        
    Returns:
        Dictionary mapping chain IDs to sequences
        
    Raises:
        FileNotFoundError: If PDB file doesn't exist
        ValueError: If no valid sequences found
    """
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    sequences = {}
    current_chain = None
    current_sequence = []
    last_residue_num = None
    
    logger.info(f"Extracting protein sequences from PDB: {pdb_file}")
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                # Parse ATOM records
                if line.startswith('ATOM') or (include_hetero and line.startswith('HETATM')):
                    # Extract fields from PDB format
                    chain = line[21:22].strip()
                    residue_name = line[17:20].strip()
                    residue_num = int(line[22:26].strip())
                    atom_name = line[12:16].strip()
                    
                    # Only process CA atoms for protein backbone
                    if atom_name != 'CA':
                        continue
                    
                    # Skip if not the requested chain
                    if chain_id and chain != chain_id:
                        continue
                    
                    # Convert 3-letter to 1-letter amino acid code
                    aa_1letter = AA_3TO1.get(residue_name)
                    if not aa_1letter:
                        if residue_name in ['HOH', 'WAT']:  # Water molecules
                            continue
                        logger.warning(f"Unknown residue: {residue_name} in chain {chain}")
                        continue
                    
                    # Handle chain changes
                    if current_chain != chain:
                        # Save previous chain sequence
                        if current_chain and current_sequence:
                            sequences[current_chain] = ''.join(current_sequence)
                        
                        # Start new chain
                        current_chain = chain
                        current_sequence = []
                        last_residue_num = None
                    
                    # Handle missing residues (gaps in numbering)
                    if last_residue_num and residue_num != last_residue_num + 1:
                        gap_size = residue_num - last_residue_num - 1
                        if gap_size > 0:
                            logger.warning(f"Gap detected in chain {chain}: {gap_size} residues missing")
                            # You could add 'X' for unknown residues: current_sequence.extend(['X'] * gap_size)
                    
                    # Add residue if not already added (avoid duplicates)
                    if last_residue_num != residue_num:
                        current_sequence.append(aa_1letter)
                        last_residue_num = residue_num
        
        # Save final chain sequence
        if current_chain and current_sequence:
            sequences[current_chain] = ''.join(current_sequence)
    
    except Exception as e:
        raise ValueError(f"Error parsing PDB file {pdb_file}: {str(e)}")
    
    if not sequences:
        raise ValueError(f"No valid protein sequences found in {pdb_file}")
    
    # Log results
    for chain, seq in sequences.items():
        logger.info(f"Chain {chain}: {len(seq)} residues")
    
    return sequences


def extract_smiles_from_sdf(sdf_file: str) -> List[Dict[str, str]]:
    """
    Extract SMILES strings and metadata from an SDF file.
    
    Args:
        sdf_file: Path to the SDF file
        
    Returns:
        List of dictionaries containing SMILES and metadata for each molecule
        
    Raises:
        FileNotFoundError: If SDF file doesn't exist
        ValueError: If no valid molecules found
    """
    if not os.path.exists(sdf_file):
        raise FileNotFoundError(f"SDF file not found: {sdf_file}")
    
    molecules = []
    current_molecule = {}
    in_properties = False
    
    logger.info(f"Extracting SMILES from SDF: {sdf_file}")
    
    try:
        with open(sdf_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Start of new molecule (molecule name)
            if i == 0 or (i > 0 and lines[i-1].strip() == '$$$$'):
                if current_molecule and 'smiles' in current_molecule:
                    molecules.append(current_molecule)
                
                current_molecule = {'name': line if line else f'mol_{len(molecules)+1}'}
                in_properties = False
            
            # Properties block starts after M  END
            elif line.startswith('M  END'):
                in_properties = True
            
            # Extract properties
            elif in_properties and line.startswith('>'):
                # Property name
                prop_match = re.match(r'<([^>]+)>', line)
                if prop_match:
                    prop_name = prop_match.group(1)
                    i += 1
                    if i < len(lines):
                        prop_value = lines[i].strip()
                        current_molecule[prop_name] = prop_value
                        
                        # Check if this is a SMILES property
                        if prop_name.lower() in ['smiles', 'canonical_smiles', 'smiles_string']:
                            current_molecule['smiles'] = prop_value
            
            # End of molecule
            elif line == '$$$$':
                if current_molecule and 'smiles' in current_molecule:
                    molecules.append(current_molecule)
                current_molecule = {}
                in_properties = False
            
            i += 1
        
        # Add final molecule if exists
        if current_molecule and 'smiles' in current_molecule:
            molecules.append(current_molecule)
    
    except Exception as e:
        raise ValueError(f"Error parsing SDF file {sdf_file}: {str(e)}")
    
    # Try to generate SMILES using RDKit if available and no SMILES found
    if not molecules or not any('smiles' in mol for mol in molecules):
        molecules = _extract_smiles_with_rdkit(sdf_file)
    
    if not molecules:
        raise ValueError(f"No valid molecules with SMILES found in {sdf_file}")
    
    logger.info(f"Extracted {len(molecules)} molecules from SDF")
    return molecules


def _extract_smiles_with_rdkit(sdf_file: str) -> List[Dict[str, str]]:
    """
    Extract SMILES using RDKit (fallback method).
    
    Args:
        sdf_file: Path to the SDF file
        
    Returns:
        List of dictionaries with SMILES and metadata
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        molecules = []
        supplier = Chem.SDMolSupplier(sdf_file)
        
        for i, mol in enumerate(supplier):
            if mol is None:
                continue
            
            smiles = Chem.MolToSmiles(mol)
            mol_dict = {
                'name': mol.GetProp('_Name') if mol.HasProp('_Name') else f'mol_{i+1}',
                'smiles': smiles,
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol)
            }
            
            # Add any existing properties
            for prop_name in mol.GetPropNames():
                mol_dict[prop_name] = mol.GetProp(prop_name)
            
            molecules.append(mol_dict)
        
        logger.info(f"RDKit extracted {len(molecules)} molecules")
        return molecules
        
    except ImportError:
        logger.warning("RDKit not available, cannot extract SMILES from SDF structure")
        return []
    except Exception as e:
        logger.error(f"RDKit extraction failed: {str(e)}")
        return []


def write_fasta_file(sequences: Dict[str, str], output_file: str, description: str = ""):
    """
    Write protein sequences to a FASTA file.
    
    Args:
        sequences: Dictionary mapping sequence names to sequences
        output_file: Path to output FASTA file
        description: Optional description for sequences
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for name, sequence in sequences.items():
            desc = f" {description}" if description else ""
            f.write(f">{name}{desc}\n")
            
            # Write sequence in 80-character lines
            for i in range(0, len(sequence), 80):
                f.write(f"{sequence[i:i+80]}\n")
    
    logger.info(f"Wrote {len(sequences)} sequences to {output_file}")


def create_boltz_input_from_files(
    pdb_file: str,
    sdf_file: str,
    output_yaml: str,
    chain_id: Optional[str] = None,
    molecule_index: int = 0
) -> str:
    """
    Create a Boltz YAML input file from PDB and SDF files.
    
    Args:
        pdb_file: Path to PDB file containing protein structure
        sdf_file: Path to SDF file containing ligand(s)
        output_yaml: Path to output YAML file
        chain_id: Specific protein chain to use (if None, use first chain)
        molecule_index: Index of molecule to use from SDF file
        
    Returns:
        Path to created YAML file
    """
    import yaml
    
    # Extract protein sequence
    protein_sequences = extract_protein_sequence_from_pdb(pdb_file, chain_id)
    
    # Use first chain if no specific chain requested
    if chain_id:
        if chain_id not in protein_sequences:
            raise ValueError(f"Chain {chain_id} not found in PDB file")
        protein_sequence = protein_sequences[chain_id]
        protein_chain_id = chain_id
    else:
        protein_chain_id = list(protein_sequences.keys())[0]
        protein_sequence = protein_sequences[protein_chain_id]
    
    # Extract ligand SMILES
    molecules = extract_smiles_from_sdf(sdf_file)
    
    if molecule_index >= len(molecules):
        raise ValueError(f"Molecule index {molecule_index} out of range (0-{len(molecules)-1})")
    
    ligand_smiles = molecules[molecule_index]['smiles']
    ligand_name = molecules[molecule_index].get('name', f'ligand_{molecule_index}')
    
    # Create YAML content
    yaml_content = {
        'version': 1,
        'sequences': [
            {
                'protein': {
                    'id': protein_chain_id,
                    'sequence': protein_sequence
                }
            },
            {
                'ligand': {
                    'id': 'L',
                    'smiles': ligand_smiles
                }
            }
        ],
        'properties': [
            {
                'affinity': {
                    'binder': 'L'
                }
            }
        ]
    }
    
    # Write YAML file
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
    with open(output_yaml, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    logger.info(f"Created Boltz input file: {output_yaml}")
    logger.info(f"Protein: Chain {protein_chain_id}, {len(protein_sequence)} residues")
    logger.info(f"Ligand: {ligand_name}, SMILES: {ligand_smiles}")
    
    return output_yaml


# Convenience functions for common use cases
def pdb_to_fasta(pdb_file: str, fasta_file: str, chain_id: Optional[str] = None):
    """Convert PDB file to FASTA format."""
    sequences = extract_protein_sequence_from_pdb(pdb_file, chain_id)
    write_fasta_file(sequences, fasta_file, f"from {os.path.basename(pdb_file)}")


def sdf_to_smiles_list(sdf_file: str) -> List[str]:
    """Extract just the SMILES strings from an SDF file."""
    molecules = extract_smiles_from_sdf(sdf_file)
    return [mol['smiles'] for mol in molecules]


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Molecular file utilities")
    parser.add_argument("--pdb", help="PDB file to process")
    parser.add_argument("--sdf", help="SDF file to process")
    parser.add_argument("--chain", help="Specific chain ID to extract")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if args.pdb and args.output:
        sequences = extract_protein_sequence_from_pdb(args.pdb, args.chain)
        for chain, seq in sequences.items():
            print(f"Chain {chain}: {seq}")
        
        if args.output.endswith('.fasta'):
            write_fasta_file(sequences, args.output)
            print(f"FASTA written to {args.output}")
    
    if args.sdf:
        molecules = extract_smiles_from_sdf(args.sdf)
        for mol in molecules:
            print(f"{mol.get('name', 'Unknown')}: {mol['smiles']}")
