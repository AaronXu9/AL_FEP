#!/usr/bin/env python3
"""
Quick start script for AL-FEP framework
"""

import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from al_fep import (
    MLFEPOracle, ActiveLearningPipeline, 
    MolecularDataset, setup_logging, load_config
)


def main():
    """Run a simple active learning experiment."""
    
    # Setup logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting AL-FEP quick start demo")
    
    # Load configuration
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    config = load_config(
        os.path.join(config_dir, 'targets', '7jvr.yaml'),
        os.path.join(config_dir, 'default.yaml')
    )
    
    # Initialize ML-FEP oracle
    oracle = MLFEPOracle(target="7jvr", config=config)
    logger.info(f"Initialized oracle: {oracle}")
    
    # Example molecules
    smiles_list = [
        "CCO", "CCN", "CCC", "CCCC", "CCCCC",
        "c1ccccc1", "c1cccnc1", "c1ccncc1", 
        "CC(C)O", "CC(=O)O", "c1ccc(O)cc1"
    ]
    
    # Create dataset
    dataset = MolecularDataset(smiles=smiles_list, name="QuickStart")
    logger.info(f"Created dataset with {len(dataset)} molecules")
    
    # Setup active learning
    al_pipeline = ActiveLearningPipeline(
        oracles=[oracle],
        strategy="uncertainty_sampling",
        batch_size=3,
        max_iterations=3,
        config=config
    )
    
    # Load molecular pool
    al_pipeline.load_molecular_pool(smiles_list)
    
    # Run active learning
    logger.info("Running active learning...")
    results = al_pipeline.run()
    
    # Display results
    logger.info(f"Active learning completed!")
    logger.info(f"Total iterations: {results['total_iterations']}")
    logger.info(f"Total evaluated: {results['total_evaluated']}")
    
    # Show best molecules
    best_molecules = results['best_molecules']
    logger.info("\nTop 3 molecules:")
    for i, mol in enumerate(best_molecules[:3], 1):
        score = mol.get('ml-fep_score', 0)
        smiles = mol['smiles']
        logger.info(f"{i}. Score: {score:.3f} - {smiles}")
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 'quickstart_results.csv')
    al_pipeline.save_results(results_file)
    logger.info(f"Results saved to {results_file}")
    
    logger.info("Quick start demo completed successfully!")


if __name__ == "__main__":
    main()
