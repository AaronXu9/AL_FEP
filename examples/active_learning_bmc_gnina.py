#!/usr/bin/env python3
"""
Active Learning Pipeline for BMC FEP Validation Set using GNINA Oracle

This script demonstrates an active learning workflow for molecular discovery
using the BMC FEP validation set as the molecular pool and GNINA as the oracle.

Features:
- Loads molecules from SDF file with experimental data
- Uses GNINA docking oracle for molecular evaluation
- Implements uncertainty sampling for active learning
- Tracks molecules by selection round
- Saves comprehensive results with metadata
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcTPSA

# Import AL_FEP modules
from al_fep.oracles.docking_oracle import DockingOracle
from al_fep.active_learning.pipeline import ActiveLearningPipeline
from al_fep.active_learning.uncertainty_sampling import UncertaintySampling
from al_fep.utils.molecular_file_utils import extract_smiles_from_sdf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BMCActiveLearningExperiment:
    """
    Active Learning experiment for BMC FEP validation set.
    """
    
    def __init__(
        self,
        sdf_file: str,
        protein_file: str,
        output_dir: str = "results/bmc_al_gnina",
        batch_size: int = 10,
        max_rounds: int = 20,
        initial_size: int = 50,
        random_seed: int = 42
    ):
        """
        Initialize the BMC Active Learning experiment.
        
        Args:
            sdf_file: Path to BMC SDF file with molecules
            protein_file: Path to protein PDB file
            output_dir: Directory to save results
            batch_size: Number of molecules to select per round
            max_rounds: Maximum number of AL rounds
            initial_size: Size of initial training set
            random_seed: Random seed for reproducibility
        """
        self.sdf_file = sdf_file
        self.protein_file = protein_file
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.initial_size = initial_size
        self.random_seed = random_seed
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.molecules_pool = []
        self.experimental_data = {}
        self.selected_molecules = []
        self.round_results = []
        
        # Set random seed
        np.random.seed(random_seed)
        
        logger.info(f"Initialized BMC AL experiment with:")
        logger.info(f"  SDF file: {sdf_file}")
        logger.info(f"  Protein file: {protein_file}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Max rounds: {max_rounds}")
    
    def load_molecules(self):
        """Load molecules from SDF file with experimental data."""
        logger.info("Loading molecules from SDF file...")
        
        # Read SDF file
        supplier = Chem.SDMolSupplier(self.sdf_file)
        molecules = []
        
        for i, mol in enumerate(supplier):
            if mol is None:
                continue
                
            # Extract molecular data
            mol_data = {
                'mol_id': i,
                'mol': mol,
                'smiles': Chem.MolToSmiles(mol),
                'entry_name': mol.GetProp('s_m_entry_name') if mol.HasProp('s_m_entry_name') else f'mol_{i}',
                'pic50_exp': float(mol.GetProp('PIC50_MEAN')) if mol.HasProp('PIC50_MEAN') else None,
                'exp_dg': float(mol.GetProp('r_fepplus_exp_dg')) if mol.HasProp('r_fepplus_exp_dg') else None,
                'selected_round': None,  # Will be set when molecule is selected
                'gnina_score': None,     # Will be set when scored by GNINA
                'gnina_cnn_score': None, # Will be set when scored by GNINA
                'uncertainty': None      # Will be set during AL
            }
            
            # Add molecular descriptors
            mol_data.update(self._calculate_descriptors(mol))
            
            molecules.append(mol_data)
            
        self.molecules_pool = molecules
        logger.info(f"Loaded {len(self.molecules_pool)} molecules from SDF file")
        
        # Store experimental data for analysis
        self.experimental_data = {
            mol['mol_id']: {
                'entry_name': mol['entry_name'],
                'pic50_exp': mol['pic50_exp'],
                'exp_dg': mol['exp_dg']
            }
            for mol in self.molecules_pool
        }
    
    def _calculate_descriptors(self, mol):
        """Calculate molecular descriptors."""
        try:
            return {
                'mw': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': Lipinski.NumHDonors(mol),
                'hba': Lipinski.NumHAcceptors(mol),
                'tpsa': CalcTPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol)
            }
        except Exception as e:
            logger.warning(f"Error calculating descriptors: {e}")
            return {
                'mw': None, 'logp': None, 'hbd': None, 'hba': None,
                'tpsa': None, 'rotatable_bonds': None, 'aromatic_rings': None
            }
    
    def setup_gnina_oracle(self):
        """Setup GNINA docking oracle."""
        logger.info("Setting up GNINA oracle...")
        
        # GNINA oracle configuration
        docking_config = {
            'docking': {
                'engine': 'gnina',
                'receptor_file': self.protein_file,
                'scoring_function': 'default',
                'cnn_scoring': True,
                'search_space': {
                    'center_x': 0.0,
                    'center_y': 0.0, 
                    'center_z': 0.0,
                    'size_x': 20.0,
                    'size_y': 20.0,
                    'size_z': 20.0
                },
                'exhaustiveness': 8,
                'num_poses': 1
            }
        }
        
        self.oracle = DockingOracle(
            target='BMC',
            receptor_file=self.protein_file,
            config=docking_config
        )
        
        logger.info("GNINA oracle setup complete")
    
    def run_initial_selection(self):
        """Select initial molecules randomly."""
        logger.info(f"Selecting {self.initial_size} initial molecules randomly...")
        
        # Random selection for initial set
        available_indices = list(range(len(self.molecules_pool)))
        initial_indices = np.random.choice(
            available_indices, 
            size=min(self.initial_size, len(available_indices)), 
            replace=False
        )
        
        # Mark molecules as selected in round 0
        for idx in initial_indices:
            self.molecules_pool[idx]['selected_round'] = 0
            self.selected_molecules.append(self.molecules_pool[idx])
        
        logger.info(f"Selected {len(initial_indices)} molecules for initial training set")
        
    def evaluate_molecules(self, molecules: List[Dict], round_num: int):
        """Evaluate molecules using GNINA oracle."""
        logger.info(f"Evaluating {len(molecules)} molecules with GNINA (Round {round_num})...")
        
        evaluated_molecules = []
        
        for i, mol_data in enumerate(molecules):
            try:
                # Evaluate with oracle
                smiles = mol_data['smiles']
                result = self.oracle.evaluate([smiles])
                
                # Extract scores
                if result and len(result) > 0:
                    mol_data['gnina_score'] = result[0].get('score', None)
                    mol_data['gnina_cnn_score'] = result[0].get('cnn_score', None)
                else:
                    mol_data['gnina_score'] = None
                    mol_data['gnina_cnn_score'] = None
                
                evaluated_molecules.append(mol_data)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Evaluated {i + 1}/{len(molecules)} molecules")
                    
            except Exception as e:
                logger.error(f"Error evaluating molecule {mol_data['entry_name']}: {e}")
                mol_data['gnina_score'] = None
                mol_data['gnina_cnn_score'] = None
                evaluated_molecules.append(mol_data)
        
        logger.info(f"Completed evaluation of {len(evaluated_molecules)} molecules")
        return evaluated_molecules
    
    def calculate_uncertainty(self, molecules: List[Dict]):
        """Calculate uncertainty for molecule selection."""
        # Simple uncertainty based on score variance
        # In a real scenario, this would use model ensemble or prediction intervals
        
        scores = [mol.get('gnina_score', 0) for mol in molecules if mol.get('gnina_score') is not None]
        if not scores:
            return molecules
        
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        
        for mol in molecules:
            if mol.get('gnina_score') is not None:
                # Uncertainty based on distance from mean (simplified)
                uncertainty = abs(mol['gnina_score'] - score_mean) / (score_std + 1e-6)
                mol['uncertainty'] = uncertainty
            else:
                mol['uncertainty'] = 0.0
        
        return molecules
    
    def select_next_batch(self, round_num: int):
        """Select next batch of molecules using uncertainty sampling."""
        logger.info(f"Selecting next batch for round {round_num}...")
        
        # Get unselected molecules
        unselected = [mol for mol in self.molecules_pool if mol['selected_round'] is None]
        
        if len(unselected) == 0:
            logger.info("No more molecules to select")
            return []
        
        # Calculate uncertainty for unselected molecules
        unselected = self.calculate_uncertainty(unselected)
        
        # Sort by uncertainty (highest first)
        unselected.sort(key=lambda x: x.get('uncertainty', 0), reverse=True)
        
        # Select top molecules
        batch_size = min(self.batch_size, len(unselected))
        selected_batch = unselected[:batch_size]
        
        # Mark as selected
        for mol in selected_batch:
            mol['selected_round'] = round_num
            self.selected_molecules.append(mol)
        
        logger.info(f"Selected {batch_size} molecules for round {round_num}")
        return selected_batch
    
    def run_active_learning(self):
        """Run the complete active learning pipeline."""
        logger.info("Starting active learning pipeline...")
        
        # Initial selection and evaluation
        self.run_initial_selection()
        initial_molecules = [mol for mol in self.molecules_pool if mol['selected_round'] == 0]
        self.evaluate_molecules(initial_molecules, 0)
        
        # Save initial results
        self.save_round_results(0, initial_molecules)
        
        # Active learning rounds
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"\n--- Active Learning Round {round_num} ---")
            
            # Select next batch
            batch = self.select_next_batch(round_num)
            
            if len(batch) == 0:
                logger.info("No more molecules to select. Stopping AL.")
                break
            
            # Evaluate batch
            self.evaluate_molecules(batch, round_num)
            
            # Save round results
            self.save_round_results(round_num, batch)
            
            # Check stopping criteria
            if len(batch) < self.batch_size:
                logger.info("Batch size smaller than requested. Stopping AL.")
                break
        
        logger.info("Active learning pipeline completed!")
    
    def save_round_results(self, round_num: int, molecules: List[Dict]):
        """Save results for a specific round."""
        round_data = {
            'round': round_num,
            'num_molecules': len(molecules),
            'timestamp': datetime.now().isoformat(),
            'molecules': []
        }
        
        for mol in molecules:
            mol_result = {
                'mol_id': mol['mol_id'],
                'entry_name': mol['entry_name'],
                'smiles': mol['smiles'],
                'selected_round': mol['selected_round'],
                'gnina_score': mol['gnina_score'],
                'gnina_cnn_score': mol['gnina_cnn_score'],
                'uncertainty': mol.get('uncertainty'),
                'pic50_exp': mol['pic50_exp'],
                'exp_dg': mol['exp_dg'],
                # Molecular descriptors
                'mw': mol.get('mw'),
                'logp': mol.get('logp'),
                'hbd': mol.get('hbd'),
                'hba': mol.get('hba'),
                'tpsa': mol.get('tpsa'),
                'rotatable_bonds': mol.get('rotatable_bonds'),
                'aromatic_rings': mol.get('aromatic_rings')
            }
            round_data['molecules'].append(mol_result)
        
        self.round_results.append(round_data)
        
        # Save individual round file
        round_file = self.output_dir / f"round_{round_num:02d}_results.json"
        with open(round_file, 'w') as f:
            json.dump(round_data, f, indent=2)
        
        logger.info(f"Saved round {round_num} results to {round_file}")
    
    def save_final_results(self):
        """Save comprehensive final results."""
        logger.info("Saving final results...")
        
        # Create comprehensive results DataFrame
        all_results = []
        for mol in self.molecules_pool:
            result = {
                'mol_id': mol['mol_id'],
                'entry_name': mol['entry_name'],
                'smiles': mol['smiles'],
                'selected_round': mol['selected_round'],
                'gnina_score': mol['gnina_score'],
                'gnina_cnn_score': mol['gnina_cnn_score'],
                'uncertainty': mol.get('uncertainty'),
                'pic50_exp': mol['pic50_exp'],
                'exp_dg': mol['exp_dg'],
                'selected': mol['selected_round'] is not None,
                # Molecular descriptors
                'mw': mol.get('mw'),
                'logp': mol.get('logp'),
                'hbd': mol.get('hbd'),
                'hba': mol.get('hba'),
                'tpsa': mol.get('tpsa'),
                'rotatable_bonds': mol.get('rotatable_bonds'),
                'aromatic_rings': mol.get('aromatic_rings')
            }
            all_results.append(result)
        
        # Save as CSV
        df = pd.DataFrame(all_results)
        results_file = self.output_dir / "bmc_al_gnina_complete_results.csv"
        df.to_csv(results_file, index=False)
        
        # Save experiment metadata
        metadata = {
            'experiment_name': 'BMC_AL_GNINA',
            'sdf_file': str(self.sdf_file),
            'protein_file': str(self.protein_file),
            'parameters': {
                'batch_size': self.batch_size,
                'max_rounds': self.max_rounds,
                'initial_size': self.initial_size,
                'random_seed': self.random_seed
            },
            'results_summary': {
                'total_molecules': len(self.molecules_pool),
                'selected_molecules': len(self.selected_molecules),
                'completed_rounds': len(self.round_results),
                'selection_percentage': len(self.selected_molecules) / len(self.molecules_pool) * 100
            },
            'files': {
                'complete_results': str(results_file.name),
                'round_results': [f"round_{i:02d}_results.json" for i in range(len(self.round_results))]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = self.output_dir / "experiment_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved complete results to {results_file}")
        logger.info(f"Saved experiment metadata to {metadata_file}")
        
        # Print summary
        self.print_experiment_summary()
    
    def print_experiment_summary(self):
        """Print experiment summary."""
        print("\n" + "="*60)
        print("BMC ACTIVE LEARNING EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Total molecules in pool: {len(self.molecules_pool)}")
        print(f"Selected molecules: {len(self.selected_molecules)}")
        print(f"Selection percentage: {len(self.selected_molecules)/len(self.molecules_pool)*100:.1f}%")
        print(f"Completed rounds: {len(self.round_results)}")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)


def main():
    """Main function to run the active learning experiment."""
    parser = argparse.ArgumentParser(description="BMC Active Learning with GNINA Oracle")
    
    parser.add_argument(
        "--sdf_file", 
        default="data/targets/BMC_FEP_validation_set_J_Med_Chem_2020.sdf",
        help="Path to BMC SDF file"
    )
    parser.add_argument(
        "--protein_file",
        default="data/BMC_FEP_protein_model_6ZB1.pdb", 
        help="Path to protein PDB file"
    )
    parser.add_argument(
        "--output_dir",
        default="results/bmc_al_gnina",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of molecules to select per round"
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=20,
        help="Maximum number of AL rounds"
    )
    parser.add_argument(
        "--initial_size",
        type=int,
        default=50,
        help="Size of initial training set"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    base_dir = Path(__file__).parent.parent
    sdf_file = base_dir / args.sdf_file
    protein_file = base_dir / args.protein_file
    
    # Check if files exist
    if not sdf_file.exists():
        logger.error(f"SDF file not found: {sdf_file}")
        sys.exit(1)
    
    if not protein_file.exists():
        logger.error(f"Protein file not found: {protein_file}")
        sys.exit(1)
    
    # Run experiment
    try:
        experiment = BMCActiveLearningExperiment(
            sdf_file=str(sdf_file),
            protein_file=str(protein_file),
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_rounds=args.max_rounds,
            initial_size=args.initial_size,
            random_seed=args.random_seed
        )
        
        # Load molecules
        experiment.load_molecules()
        
        # Setup oracle
        experiment.setup_gnina_oracle()
        
        # Run active learning
        experiment.run_active_learning()
        
        # Save final results
        experiment.save_final_results()
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
