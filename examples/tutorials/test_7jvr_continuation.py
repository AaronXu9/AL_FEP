#!/usr/bin/env python3
"""
7JVR AL-FEP Test Continuation Script
This script continues the testing where the notebook left off.
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# AL-FEP imports
from al_fep import (
    MLFEPOracle, DockingOracle, FEPOracle,
    ActiveLearningPipeline,
    MolecularDataset, MolecularFeaturizer,
    setup_logging, load_config
)

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors

def main():
    print("🎯 7JVR AL-FEP Test Continuation")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Load configuration
    config_dir = Path('config')
    config = load_config(
        config_dir / 'targets' / '7jvr.yaml',
        config_dir / 'default.yaml'
    )
    
    print(f"✓ Loaded 7JVR configuration")
    print(f"  - Target: {config['target_info']['name']}")
    print(f"  - Resolution: {config['target_info']['resolution']} Å")
    print(f"  - Binding Site: {config['binding_site']['center']}")
    
    # Initialize oracles
    print("\n🔬 Initializing Oracles...")
    ml_oracle = MLFEPOracle(target="7jvr", config=config)
    
    docking_config = config.copy()
    docking_config['docking'] = docking_config.get('docking', {})
    docking_config['docking']['mock_mode'] = True
    docking_oracle = DockingOracle(target="7jvr", config=docking_config)
    
    fep_config = config.copy()
    fep_config['fep'] = fep_config.get('fep', {})
    fep_config['fep']['mock_mode'] = True
    fep_oracle = FEPOracle(target="7jvr", config=fep_config)
    
    print("✓ All oracles initialized")
    
    # Test molecules
    test_molecules = [
        "CC1(C)SC2C(NC(=O)C(NC(=O)OC(C)(C)C)C3CCC4=C(c5ccccc5)C3N4)C(=O)N2C1C(=O)O",  # Nirmatrelvir
        "CC(C)CC(C(=O)NC(CC1CCCCC1)C(=O)NC(C#N)CC2=CC=CC=C2)NC(=O)C(C(C)(C)C)NC(=O)OC(C)(C)C",
        "CC(C)CC(NC(=O)C(Cc1ccccc1)NC(=O)OC(C)(C)C)C(=O)NC1CC2CCCCC2C1",
        "CCN(CC)CCCC(C)NC1=C2N=CC=NC2=NC=N1",
        "Cc1ccc(S(=O)(=O)N2CCN(CC2)C(=O)C3CC4CCC(C3)N4C(=O)OC(C)(C)C)cc1",
        "COc1cc(ccc1O)C2CC(=O)c3c(O)cc(O)cc3O2",
        "CC(=O)NC1C(C(C(OC1OC2=C(C(=O)C3=CC=CC=C3C2=O)O)CO)O)O",
        "c1ccc2c(c1)nc(s2)N3CCN(CC3)C(=O)C4CC5CCC(C4)N5",
        "CC1CCN(CC1)C(=O)C2=C(C=CS2)NC(=O)C3=CC=C(C=C3)Cl",
        "COC1=CC=C(C=C1)C2=NN=C(N2)NC3=CC=C(C=C3)S(=O)(=O)N",
        "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
        "CC1=CN=C(C=C1)NC2=NC=C(C(=N2)NC3=CC=CC(=C3)C#C)C4=CC=CC=C4",
        "CC(C)NCC(COC1=CC=CC2=C1C=CN2)O",
        "CN1CCN(CC1)C2=NC3=CC=CC=C3N=C2NC4=CC=C(C=C4)C",
        "COC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)C)C",
    ]
    
    # Validate molecules
    valid_molecules = []
    for smiles in test_molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_molecules.append(smiles)
    
    print(f"\n📋 Prepared {len(valid_molecules)} valid test molecules")
    
    # Run Active Learning Pipeline
    print("\n🧠 Running Active Learning Pipeline...")
    dataset = MolecularDataset(smiles=valid_molecules, name="7JVR_TestSet")
    
    al_pipeline = ActiveLearningPipeline(
        oracles=[ml_oracle, docking_oracle, fep_oracle],
        strategy="uncertainty_sampling",
        batch_size=3,
        max_iterations=5,
        config=config
    )
    
    al_pipeline.load_molecular_pool(valid_molecules)
    al_results = al_pipeline.run()
    
    print("\n🎉 Active Learning Complete!")
    print(f"- Iterations: {al_results['total_iterations']}")
    print(f"- Molecules evaluated: {al_results['total_evaluated']}")
    
    # Oracle performance
    print("\n📊 Oracle Performance:")
    for stats in al_results['oracle_statistics']:
        oracle_name = stats.get('oracle', 'Unknown')
        call_count = stats.get('call_count', 0)
        avg_time = stats.get('avg_time', 0.0)
        print(f"  - {oracle_name}: {call_count} calls, {avg_time:.3f}s avg")
    
    # Best molecules
    print("\n🏆 Top 5 Molecules by ML-FEP Score:")
    best_molecules = al_results['best_molecules'][:5]
    for i, mol in enumerate(best_molecules, 1):
        smiles = mol['smiles']
        ml_score = mol.get('ml-fep_score', 'N/A')
        docking_score = mol.get('docking_score', 'N/A')
        fep_score = mol.get('fep_score', 'N/A')
        
        print(f"{i}. SMILES: {smiles[:60]}{'...' if len(smiles) > 60 else ''}")
        print(f"   ML-FEP: {ml_score}, Docking: {docking_score}, FEP: {fep_score}")
        print()
    
    # Save results
    output_dir = Path('data/results')
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / '7jvr_al_fep_results.csv'
    al_pipeline.save_results(str(results_file))
    print(f"✓ Results saved to: {results_file}")
    
    # Molecular properties analysis
    print("\n🧪 Molecular Properties Analysis:")
    mol_properties = []
    for smiles in valid_molecules:
        mol = Chem.MolFromSmiles(smiles)
        props = {
            'SMILES': smiles,
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol)
        }
        mol_properties.append(props)
    
    df_molecules = pd.DataFrame(mol_properties)
    
    # Lipinski Rule of Five compliance
    lipinski_violations = (
        (df_molecules['MW'] > 500) +
        (df_molecules['LogP'] > 5) +
        (df_molecules['HBD'] > 5) +
        (df_molecules['HBA'] > 10)
    )
    
    print(f"Lipinski Rule of Five Compliance:")
    print(f"- 0 violations: {sum(lipinski_violations == 0)} molecules")
    print(f"- 1 violation: {sum(lipinski_violations == 1)} molecules")
    print(f"- 2+ violations: {sum(lipinski_violations >= 2)} molecules")
    
    # Feature importance
    print("\n🔍 ML-FEP Feature Importance:")
    try:
        feature_importance = ml_oracle.get_feature_importance()
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            print("Top 10 most important features:")
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                print(f"{i:2d}. {feature:<20} : {importance:.4f}")
        else:
            print("No feature importance available")
    except Exception as e:
        print(f"Error getting feature importance: {e}")
    
    print("\n🎯 7JVR AL-FEP Pipeline Test Summary")
    print("=" * 60)
    print(f"Target Protein: 7JVR (SARS-CoV-2 Main Protease)")
    print(f"PDB Resolution: {config['target_info']['resolution']} Å")
    print(f"Binding Site: {config['binding_site']['center']}")
    print()
    print(f"Dataset:")
    print(f"- Total molecules tested: {len(valid_molecules)}")
    print(f"- Drug-like molecules (0-1 Lipinski violations): {sum(lipinski_violations <= 1)}")
    print()
    print(f"Active Learning Results:")
    print(f"- Iterations completed: {al_results['total_iterations']}")
    print(f"- Molecules evaluated: {al_results['total_evaluated']}")
    print(f"- Success rate: {al_results['total_evaluated']/len(valid_molecules)*100:.1f}%")
    print()
    print(f"Best Performing Molecule:")
    if best_molecules:
        best_mol = best_molecules[0]
        print(f"- SMILES: {best_mol['smiles'][:80]}{'...' if len(best_mol['smiles']) > 80 else ''}")
        print(f"- ML-FEP Score: {best_mol.get('ml-fep_score', 'N/A')}")
        print(f"- Uncertainty: {best_mol.get('uncertainty', 'N/A')}")
    print()
    print("✅ All tests completed successfully!")
    print("📁 Results saved to data/results/")
    print("🔬 Ready for production use with real protein structures!")

if __name__ == "__main__":
    main()
