"""
Active Learning Pipeline for molecular discovery
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from rdkit import Chem

from ..oracles.base_oracle import BaseOracle
from .uncertainty_sampling import UncertaintySampling

logger = logging.getLogger(__name__)


class ActiveLearningPipeline:
    """
    Main active learning pipeline for molecular discovery.
    """
    
    def __init__(
        self,
        oracles: List[BaseOracle],
        strategy: str = "uncertainty_sampling",
        initial_pool_size: int = 1000,
        batch_size: int = 10,
        max_iterations: int = 50,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the active learning pipeline.
        
        Args:
            oracles: List of oracle objects for evaluation
            strategy: Active learning strategy
            initial_pool_size: Size of initial molecular pool
            batch_size: Number of molecules to evaluate per iteration
            max_iterations: Maximum number of AL iterations
            config: Configuration dictionary
        """
        self.oracles = oracles
        self.strategy = strategy
        self.initial_pool_size = initial_pool_size
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.config = config or {}
        
        # Initialize strategy
        self.al_strategy = self._initialize_strategy()
        
        # Data storage
        self.molecular_pool = []
        self.evaluated_molecules = []
        self.results_history = []
        self.iteration = 0
        
        logger.info(f"ActiveLearningPipeline initialized with {len(oracles)} oracles")
    
    def _initialize_strategy(self):
        """Initialize the active learning strategy."""
        if self.strategy == "uncertainty_sampling":
            return UncertaintySampling(config=self.config)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def load_molecular_pool(self, smiles_list: List[str]):
        """
        Load molecular pool for active learning.
        
        Args:
            smiles_list: List of SMILES strings
        """
        # Validate and canonicalize SMILES
        valid_smiles = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol)
                valid_smiles.append(canonical_smiles)
        
        self.molecular_pool = valid_smiles
        logger.info(f"Loaded {len(self.molecular_pool)} molecules into pool")
    
    def generate_initial_pool(self):
        """Generate initial molecular pool using random sampling or enumeration."""
        # For demonstration, create a simple pool
        # In practice, this would use chemical databases like ChEMBL, ZINC, etc.
        
        logger.info("Generating initial molecular pool...")
        
        # Some example SMILES for demonstration
        example_smiles = [
            "CCO", "CCN", "CCC", "CCCC", "CCCCC",
            "c1ccccc1", "c1cccnc1", "c1ccncc1", "c1cncnc1",
            "CC(C)O", "CC(C)N", "CC(=O)O", "CC(=O)N",
            "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(C)cc1",
            "CCc1ccccc1", "CCc1ccncc1", "CCc1cncnc1"
        ]
        
        # Expand with variations
        self.molecular_pool = example_smiles * (self.initial_pool_size // len(example_smiles) + 1)
        self.molecular_pool = self.molecular_pool[:self.initial_pool_size]
        
        logger.info(f"Generated initial pool with {len(self.molecular_pool)} molecules")
    
    def evaluate_molecules(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate molecules using all oracles.
        
        Args:
            smiles_list: List of SMILES to evaluate
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for smiles in smiles_list:
            molecule_results = {"smiles": smiles}
            
            # Evaluate with each oracle
            for oracle in self.oracles:
                oracle_result = oracle.evaluate(smiles)
                
                # Add oracle-specific results
                oracle_name = oracle.name.lower()
                molecule_results[f"{oracle_name}_score"] = oracle_result.get("score")
                molecule_results[f"{oracle_name}_result"] = oracle_result
            
            results.append(molecule_results)
        
        return results
    
    def select_batch(self) -> List[str]:
        """
        Select next batch of molecules using active learning strategy.
        
        Returns:
            List of selected SMILES
        """
        # Get available molecules (not yet evaluated)
        evaluated_smiles = {mol["smiles"] for mol in self.evaluated_molecules}
        available_pool = [
            smiles for smiles in self.molecular_pool 
            if smiles not in evaluated_smiles
        ]
        
        if len(available_pool) == 0:
            logger.warning("No more molecules available in pool")
            return []
        
        # Use strategy to select batch
        selected = self.al_strategy.select_batch(
            available_pool, 
            self.evaluated_molecules,
            self.batch_size
        )
        
        logger.info(f"Selected {len(selected)} molecules for evaluation")
        return selected
    
    def run_iteration(self) -> Dict[str, Any]:
        """
        Run a single active learning iteration.
        
        Returns:
            Iteration results
        """
        start_time = time.time()
        
        # Select batch
        selected_molecules = self.select_batch()
        
        if len(selected_molecules) == 0:
            logger.info("No molecules selected, stopping")
            return {"status": "no_molecules", "iteration": self.iteration}
        
        # Evaluate selected molecules
        logger.info(f"Evaluating {len(selected_molecules)} molecules...")
        evaluation_results = self.evaluate_molecules(selected_molecules)
        
        # Update evaluated molecules
        self.evaluated_molecules.extend(evaluation_results)
        
        # Calculate iteration statistics
        iteration_time = time.time() - start_time
        
        # Get best molecules so far
        best_molecules = self.get_best_molecules(n=5)
        
        iteration_result = {
            "iteration": self.iteration,
            "n_evaluated": len(selected_molecules),
            "total_evaluated": len(self.evaluated_molecules),
            "iteration_time": iteration_time,
            "best_molecules": best_molecules,
            "status": "completed"
        }
        
        self.results_history.append(iteration_result)
        
        logger.info(f"Iteration {self.iteration} completed in {iteration_time:.2f}s")
        return iteration_result
    
    def get_best_molecules(self, n: int = 10, oracle_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get best molecules evaluated so far.
        
        Args:
            n: Number of top molecules to return
            oracle_name: Oracle to use for ranking (default: first oracle)
            
        Returns:
            List of best molecules with their scores
        """
        if len(self.evaluated_molecules) == 0:
            return []
        
        # Choose oracle for ranking
        if oracle_name is None:
            oracle_name = self.oracles[0].name.lower()
        
        score_key = f"{oracle_name}_score"
        
        # Filter molecules with valid scores
        valid_molecules = [
            mol for mol in self.evaluated_molecules
            if mol.get(score_key) is not None
        ]
        
        if len(valid_molecules) == 0:
            return []
        
        # Sort by score (higher is better)
        sorted_molecules = sorted(
            valid_molecules,
            key=lambda x: x[score_key],
            reverse=True
        )
        
        return sorted_molecules[:n]
    
    def run(self, budget: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete active learning pipeline.
        
        Args:
            budget: Total evaluation budget (overrides max_iterations)
            
        Returns:
            Final results summary
        """
        logger.info("Starting active learning pipeline...")
        
        # Initialize molecular pool if empty
        if len(self.molecular_pool) == 0:
            self.generate_initial_pool()
        
        # Determine stopping criteria
        if budget is not None:
            max_evaluations = budget
        else:
            max_evaluations = self.max_iterations * self.batch_size
        
        # Run iterations
        while (
            self.iteration < self.max_iterations and
            len(self.evaluated_molecules) < max_evaluations
        ):
            iteration_result = self.run_iteration()
            
            if iteration_result["status"] == "no_molecules":
                break
            
            self.iteration += 1
        
        # Final summary
        final_results = {
            "total_iterations": self.iteration,
            "total_evaluated": len(self.evaluated_molecules),
            "best_molecules": self.get_best_molecules(n=10),
            "oracle_statistics": [oracle.get_statistics() for oracle in self.oracles],
            "results_history": self.results_history
        }
        
        logger.info(f"Active learning completed after {self.iteration} iterations")
        logger.info(f"Total molecules evaluated: {len(self.evaluated_molecules)}")
        
        return final_results
    
    def save_results(self, filepath: str):
        """Save results to file."""
        # Convert to DataFrame for easy saving
        df = pd.DataFrame(self.evaluated_molecules)
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get results as pandas DataFrame."""
        return pd.DataFrame(self.evaluated_molecules)
