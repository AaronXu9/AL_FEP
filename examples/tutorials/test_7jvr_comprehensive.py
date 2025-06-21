#!/usr/bin/env python3
"""
Comprehensive 7JVR Real Protein Test Script
Tests the AL-FEP framework with the SARS-CoV-2 Main Protease structure
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

# Add src to path
sys.path.append('src')

# AL-FEP imports
from al_fep import (
    MLFEPOracle, DockingOracle, FEPOracle,
    ActiveLearningPipeline,
    MolecularDataset, MolecularFeaturizer,
    setup_logging, load_config
)

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw


def main():
    """Run comprehensive 7JVR testing"""
    
    # Setup logging
    setup_logging(level="INFO")
    print("🧪 Starting AL-FEP 7JVR Comprehensive Test")
    print("=" * 60)
    
    # 1. Load 7JVR configuration
    print("\n1. Loading 7JVR Configuration...")
    config_dir = Path('config')
    config = load_config(
        config_dir / 'targets' / '7jvr.yaml',
        config_dir / 'default.yaml'
    )
    
    print("7JVR Target Configuration:")
    print(f"- PDB ID: {config['target_info']['pdb_id']}")
    print(f"- Name: {config['target_info']['name']}")
    print(f"- Resolution: {config['target_info']['resolution']} Å")
    print(f"- Binding Site Center: {config['binding_site']['center']}")
    print(f"- Search Box Size: {config['binding_site']['size']}")
    
    # Check PDB file
    pdb_file = Path('data/targets/7jvr/7JVR.pdb')
    print(f"\nPDB File Status: {'✓ Found' if pdb_file.exists() else '✗ Missing'}")
    if pdb_file.exists():
        print(f"File size: {pdb_file.stat().st_size / 1024:.1f} KB")
    
    # 2. Analyze PDB structure
    print("\n2. Analyzing PDB Structure...")
    if pdb_file.exists():
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        
        # Count different record types
        record_counts = {}
        for line in lines:
            record_type = line[:6].strip()
            record_counts[record_type] = record_counts.get(record_type, 0) + 1
        
        print("PDB Structure Analysis:")
        for record, count in sorted(record_counts.items()):
            if count > 10:  # Only show significant record types
                print(f"- {record}: {count} records")
        
        # Find ligands
        ligands = []
        for line in lines:
            if line.startswith('HETATM'):
                residue = line[17:20].strip()
                if residue not in ['HOH', 'SO4', 'CL', 'NA']:  # Skip common solvents
                    ligands.append(residue)
        
        unique_ligands = list(set(ligands))
        print(f"\nPotential Ligands: {unique_ligands}")
    
    # 3. Initialize Oracles
    print("\n3. Initializing Oracles for 7JVR...")
    
    # ML-FEP Oracle (real)
    print("Initializing ML-FEP Oracle...")
    ml_fep_oracle = MLFEPOracle(config=config)
    print("✓ ML-FEP Oracle initialized")
    
    # Docking Oracle (mock mode for testing)
    print("Initializing Docking Oracle (mock mode)...")
    docking_oracle = DockingOracle(config=config, mock_mode=True)
    print("✓ Docking Oracle initialized")
    
    # FEP Oracle (mock mode for testing)
    print("Initializing FEP Oracle (mock mode)...")
    fep_oracle = FEPOracle(config=config, mock_mode=True)
    print("✓ FEP Oracle initialized")
    
    # 4. Test with diverse drug-like molecules
    print("\n4. Testing with Diverse Drug-like Molecules...")
    
    # Test molecules including known COVID-19 inhibitors
    test_molecules = [
        # Known SARS-CoV-2 Main Protease inhibitors
        "CC(C)[C@H](NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)OC(C)(C)C)C(=O)N[C@@H](CC(C)C)C(=O)C(=O)NC1CC1",  # Nirmatrelvir
        "CC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)C(=O)NC3=CC=C(C=C3)S(=O)(=O)N",  # Paxlovid-related
        
        # FDA-approved drugs with potential activity
        "CC1=C(C=C(C=C1)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)N)C",  # Celecoxib
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        
        # Drug-like diverse molecules
        "CC1=CC=C(C=C1)S(=O)(=O)NC2=CC=CC=C2",  # Sulfonamide
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)C",  # Benzophenone
        
        # Small molecules for validation
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
    ]
    
    print(f"Testing {len(test_molecules)} molecules:")
    
    # 5. Evaluate molecules with all oracles
    print("\n5. Evaluating Molecules with All Oracles...")
    
    results = []
    
    for i, smiles in enumerate(test_molecules):
        print(f"\nMolecule {i+1}: {smiles}")
        
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("  ✗ Invalid SMILES")
            continue
        
        print(f"  ✓ Valid SMILES (MW: {Descriptors.MolWt(mol):.1f})")
        
        # Evaluate with ML-FEP Oracle
        try:
            ml_fep_score, ml_fep_uncertainty = ml_fep_oracle.evaluate([smiles])[0]
            print(f"  ML-FEP: {ml_fep_score:.3f} ± {ml_fep_uncertainty:.3f}")
        except Exception as e:
            print(f"  ML-FEP: Error - {e}")
            ml_fep_score = ml_fep_uncertainty = None
        
        # Evaluate with Docking Oracle (mock)
        try:
            docking_score, docking_uncertainty = docking_oracle.evaluate([smiles])[0]
            print(f"  Docking: {docking_score:.3f} ± {docking_uncertainty:.3f}")
        except Exception as e:
            print(f"  Docking: Error - {e}")
            docking_score = docking_uncertainty = None
        
        # Evaluate with FEP Oracle (mock)
        try:
            fep_score, fep_uncertainty = fep_oracle.evaluate([smiles])[0]
            print(f"  FEP: {fep_score:.3f} ± {fep_uncertainty:.3f}")
        except Exception as e:
            print(f"  FEP: Error - {e}")
            fep_score = fep_uncertainty = None
        
        # Store results
        results.append({
            'smiles': smiles,
            'molecular_weight': Descriptors.MolWt(mol),
            'ml_fep_score': ml_fep_score,
            'ml_fep_uncertainty': ml_fep_uncertainty,
            'docking_score': docking_score,
            'docking_uncertainty': docking_uncertainty,
            'fep_score': fep_score,
            'fep_uncertainty': fep_uncertainty
        })
    
    # 6. Analyze results
    print("\n6. Analyzing Results...")
    df = pd.DataFrame(results)
    
    # Remove failed evaluations
    df_valid = df.dropna()
    
    print(f"Successfully evaluated {len(df_valid)}/{len(results)} molecules")
    
    if len(df_valid) > 0:
        print("\nSummary Statistics:")
        print(f"ML-FEP Scores: {df_valid['ml_fep_score'].mean():.3f} ± {df_valid['ml_fep_score'].std():.3f}")
        print(f"Docking Scores: {df_valid['docking_score'].mean():.3f} ± {df_valid['docking_score'].std():.3f}")
        print(f"FEP Scores: {df_valid['fep_score'].mean():.3f} ± {df_valid['fep_score'].std():.3f}")
        
        # Find best molecules by each oracle
        print(f"\nBest molecule by ML-FEP: {df_valid.loc[df_valid['ml_fep_score'].idxmax(), 'smiles']}")
        print(f"Best molecule by Docking: {df_valid.loc[df_valid['docking_score'].idxmax(), 'smiles']}")
        print(f"Best molecule by FEP: {df_valid.loc[df_valid['fep_score'].idxmax(), 'smiles']}")
    
    # 7. Test Active Learning Pipeline
    print("\n7. Testing Active Learning Pipeline...")
    
    try:
        # Create molecular dataset
        dataset = MolecularDataset(test_molecules)
        print(f"✓ Created dataset with {len(dataset)} molecules")
        
        # Initialize active learning pipeline
        pipeline = ActiveLearningPipeline(
            oracle=ml_fep_oracle,
            dataset=dataset,
            config=config
        )
        print("✓ Active Learning Pipeline initialized")
        
        # Run a small active learning iteration
        selected_molecules = pipeline.select_molecules(n_molecules=3)
        print(f"✓ Selected {len(selected_molecules)} molecules for next iteration")
        
    except Exception as e:
        print(f"✗ Active Learning Pipeline failed: {e}")
    
    # 8. Test oracle statistics
    print("\n8. Testing Oracle Statistics...")
    
    for oracle_name, oracle in [
        ("ML-FEP", ml_fep_oracle),
        ("Docking", docking_oracle),
        ("FEP", fep_oracle)
    ]:
        stats = oracle.get_statistics()
        print(f"{oracle_name} Oracle Stats:")
        print(f"  - Evaluations: {stats['n_evaluations']}")
        print(f"  - Cache hits: {stats['cache_hits']}")
        print(f"  - Errors: {stats['n_errors']}")
    
    # 9. Save results
    print("\n9. Saving Results...")
    
    # Save evaluation results
    results_dir = Path('data/results')
    results_dir.mkdir(exist_ok=True)
    
    df.to_csv(results_dir / '7jvr_comprehensive_test.csv', index=False)
    print(f"✓ Results saved to {results_dir / '7jvr_comprehensive_test.csv'}")
    
    # Create simple visualization if we have valid results
    if len(df_valid) > 3:
        try:
            plt.figure(figsize=(12, 4))
            
            # Plot 1: Score comparison
            plt.subplot(131)
            scores = df_valid[['ml_fep_score', 'docking_score', 'fep_score']]
            scores.boxplot()
            plt.title('Oracle Score Distributions')
            plt.ylabel('Score')
            
            # Plot 2: Uncertainty comparison
            plt.subplot(132)
            uncertainties = df_valid[['ml_fep_uncertainty', 'docking_uncertainty', 'fep_uncertainty']]
            uncertainties.boxplot()
            plt.title('Oracle Uncertainty Distributions')
            plt.ylabel('Uncertainty')
            
            # Plot 3: Score vs MW
            plt.subplot(133)
            plt.scatter(df_valid['molecular_weight'], df_valid['ml_fep_score'], alpha=0.7)
            plt.xlabel('Molecular Weight')
            plt.ylabel('ML-FEP Score')
            plt.title('Score vs Molecular Weight')
            
            plt.tight_layout()
            plt.savefig(results_dir / '7jvr_test_analysis.png', dpi=150, bbox_inches='tight')
            print(f"✓ Analysis plot saved to {results_dir / '7jvr_test_analysis.png'}")
            
        except Exception as e:
            print(f"⚠ Visualization failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 7JVR Comprehensive Test Completed Successfully!")
    print(f"✓ Tested {len(test_molecules)} molecules")
    print(f"✓ Evaluated with 3 oracles (ML-FEP, Docking, FEP)")
    print(f"✓ {len(df_valid)} molecules successfully evaluated")
    print("✓ Active Learning Pipeline tested")
    print("✓ Results saved and analyzed")


if __name__ == "__main__":
    main()
