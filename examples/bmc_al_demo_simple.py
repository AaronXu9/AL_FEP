#!/usr/bin/env python3
"""
Simple Active Learning Demo for BMC using GNINA Oracle

This is a simplified version that demonstrates the core active learning workflow.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rdkit import Chem
from rdkit.Chem import Descriptors

# Import AL_FEP modules
from al_fep.oracles.docking_oracle import DockingOracle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_bmc_molecules(sdf_file):
    """Load molecules from BMC SDF file."""
    logger.info(f"Loading molecules from {sdf_file}")
    
    supplier = Chem.SDMolSupplier(sdf_file)
    molecules = []
    
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
            
        mol_data = {
            'mol_id': i,
            'entry_name': mol.GetProp('s_m_entry_name') if mol.HasProp('s_m_entry_name') else f'mol_{i}',
            'smiles': Chem.MolToSmiles(mol),
            'pic50_exp': float(mol.GetProp('PIC50_MEAN')) if mol.HasProp('PIC50_MEAN') else None,
            'mw': Descriptors.MolWt(mol),
            'selected_round': None,
            'gnina_score': None,
            'uncertainty': 0.0
        }
        molecules.append(mol_data)
    
    logger.info(f"Loaded {len(molecules)} molecules")
    return molecules


def setup_gnina_oracle(protein_file):
    """Setup GNINA docking oracle."""
    logger.info("Setting up GNINA oracle")
    
    config = {
        'docking': {
            'engine': 'gnina',
            'receptor_file': protein_file,
            'scoring_function': 'default',
            'cnn_scoring': True,
            'exhaustiveness': 4,  # Reduced for demo
            'num_poses': 1
        }
    }
    
    oracle = DockingOracle(target='BMC', config=config)
    return oracle


def evaluate_molecules(oracle, molecules):
    """Evaluate molecules with GNINA."""
    logger.info(f"Evaluating {len(molecules)} molecules")
    
    smiles_list = [mol['smiles'] for mol in molecules]
    
    try:
        results = oracle.evaluate(smiles_list)
        
        for i, (mol, result) in enumerate(zip(molecules, results)):
            if result:
                mol['gnina_score'] = result.get('score', None)
                mol['gnina_cnn_score'] = result.get('cnn_score', None)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {i + 1}/{len(molecules)} molecules")
                
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        # Set default values on error
        for mol in molecules:
            mol['gnina_score'] = None
            mol['gnina_cnn_score'] = None
    
    return molecules


def calculate_uncertainty(molecules):
    """Simple uncertainty calculation."""
    scores = [mol['gnina_score'] for mol in molecules if mol['gnina_score'] is not None]
    
    if not scores:
        return molecules
    
    score_mean = np.mean(scores)
    score_std = np.std(scores) + 1e-6
    
    for mol in molecules:
        if mol['gnina_score'] is not None:
            mol['uncertainty'] = abs(mol['gnina_score'] - score_mean) / score_std
    
    return molecules


def run_simple_active_learning():
    """Run a simplified active learning demo."""
    # File paths
    base_dir = Path(__file__).parent.parent
    sdf_file = base_dir / "data/targets/BMC_FEP_validation_set_J_Med_Chem_2020.sdf"
    protein_file = base_dir / "data/BMC_FEP_protein_model_6ZB1.pdb"
    output_dir = Path("results/bmc_al_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    initial_size = 1
    batch_size = 1
    max_rounds = 5
    
    logger.info("Starting BMC Active Learning Demo")
    logger.info(f"Initial size: {initial_size}, Batch size: {batch_size}, Max rounds: {max_rounds}")
    
    # Load molecules
    molecules = load_bmc_molecules(str(sdf_file))
    
    # Setup oracle
    oracle = setup_gnina_oracle(str(protein_file))
    
    # Initial random selection
    np.random.seed(42)
    available_indices = list(range(len(molecules)))
    initial_indices = np.random.choice(available_indices, size=initial_size, replace=False)
    
    selected_molecules = []
    all_results = []
    
    # Round 0: Initial selection
    logger.info("Round 0: Initial selection")
    initial_batch = [molecules[i] for i in initial_indices]
    for mol in initial_batch:
        mol['selected_round'] = 0
    
    selected_molecules.extend(initial_batch)
    initial_batch = evaluate_molecules(oracle, initial_batch)
    all_results.extend(initial_batch)
    
    # Active learning rounds
    for round_num in range(1, max_rounds + 1):
        logger.info(f"\nRound {round_num}: Active selection")
        
        # Get unselected molecules
        unselected = [mol for mol in molecules if mol['selected_round'] is None]
        
        if len(unselected) == 0:
            logger.info("No more molecules to select")
            break
        
        # Calculate uncertainty for current selected set
        calculate_uncertainty(selected_molecules)
        
        # Simple selection: highest uncertainty (diversity proxy)
        # In practice, would use more sophisticated methods
        unselected_sample = np.random.choice(
            len(unselected), 
            size=min(batch_size * 3, len(unselected)), 
            replace=False
        )
        candidate_batch = [unselected[i] for i in unselected_sample]
        
        # Evaluate candidates
        candidate_batch = evaluate_molecules(oracle, candidate_batch)
        candidate_batch = calculate_uncertainty(candidate_batch)
        
        # Select top candidates by uncertainty
        candidate_batch.sort(key=lambda x: x['uncertainty'], reverse=True)
        selected_batch = candidate_batch[:batch_size]
        
        # Mark as selected
        for mol in selected_batch:
            mol['selected_round'] = round_num
        
        selected_molecules.extend(selected_batch)
        all_results.extend(selected_batch)
        
        logger.info(f"Selected {len(selected_batch)} molecules in round {round_num}")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'mol_id': mol['mol_id'],
            'entry_name': mol['entry_name'],
            'smiles': mol['smiles'],
            'selected_round': mol['selected_round'],
            'gnina_score': mol['gnina_score'],
            'gnina_cnn_score': mol.get('gnina_cnn_score'),
            'uncertainty': mol['uncertainty'],
            'pic50_exp': mol['pic50_exp'],
            'mw': mol['mw']
        }
        for mol in all_results
    ])
    
    results_file = output_dir / "bmc_al_demo_results.csv"
    results_df.to_csv(results_file, index=False)
    
    # Print summary
    print(f"\nDemo completed!")
    print(f"Total selected molecules: {len(selected_molecules)}")
    print(f"Results saved to: {results_file}")
    
    return results_df


if __name__ == "__main__":
    try:
        results = run_simple_active_learning()
        print("Active learning demo completed successfully!")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
