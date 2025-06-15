"""
Uncertainty sampling strategy for active learning
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)


class UncertaintySampling:
    """
    Uncertainty sampling strategy for active learning.
    
    Selects molecules with highest prediction uncertainty.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize uncertainty sampling strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.diversity_weight = self.config.get("diversity_weight", 0.1)
        
    def select_batch(
        self,
        candidate_pool: List[str],
        evaluated_molecules: List[Dict[str, Any]],
        batch_size: int
    ) -> List[str]:
        """
        Select batch of molecules using uncertainty sampling.
        
        Args:
            candidate_pool: Available SMILES to choose from
            evaluated_molecules: Previously evaluated molecules
            batch_size: Number of molecules to select
            
        Returns:
            List of selected SMILES
        """
        if len(candidate_pool) == 0:
            return []
        
        if len(candidate_pool) <= batch_size:
            return candidate_pool
        
        # If no ML-FEP oracle available, use random sampling
        if not self._has_uncertainty_oracle(evaluated_molecules):
            logger.info("No uncertainty information available, using random sampling")
            return self._random_sampling(candidate_pool, batch_size)
        
        # Calculate uncertainty scores for candidates
        uncertainty_scores = self._calculate_uncertainty_scores(
            candidate_pool, evaluated_molecules
        )
        
        # Add diversity component
        if self.diversity_weight > 0:
            diversity_scores = self._calculate_diversity_scores(
                candidate_pool, evaluated_molecules
            )
            
            # Combine uncertainty and diversity
            combined_scores = (
                (1 - self.diversity_weight) * uncertainty_scores +
                self.diversity_weight * diversity_scores
            )
        else:
            combined_scores = uncertainty_scores
        
        # Select top candidates
        top_indices = np.argsort(combined_scores)[-batch_size:]
        selected = [candidate_pool[i] for i in top_indices]
        
        logger.info(f"Selected {len(selected)} molecules using uncertainty sampling")
        return selected
    
    def _has_uncertainty_oracle(self, evaluated_molecules: List[Dict[str, Any]]) -> bool:
        """Check if any oracle provides uncertainty information."""
        if len(evaluated_molecules) == 0:
            return False
        
        # Look for ML-FEP results with uncertainty
        sample_mol = evaluated_molecules[0]
        return any(
            key.endswith("_result") and 
            isinstance(sample_mol.get(key), dict) and
            "uncertainty" in sample_mol.get(key, {})
            for key in sample_mol.keys()
        )
    
    def _random_sampling(self, candidate_pool: List[str], batch_size: int) -> List[str]:
        """Random sampling fallback."""
        indices = np.random.choice(len(candidate_pool), size=batch_size, replace=False)
        return [candidate_pool[i] for i in indices]
    
    def _calculate_uncertainty_scores(
        self,
        candidate_pool: List[str],
        evaluated_molecules: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Calculate uncertainty scores for candidate molecules.
        
        This is a simplified implementation. In practice, you would:
        1. Use a trained ML model to predict uncertainty
        2. Consider model ensemble disagreement
        3. Use Bayesian approaches for uncertainty quantification
        """
        # For demonstration, create mock uncertainty scores
        # In practice, use ML model predictions
        
        scores = []
        for smiles in candidate_pool:
            # Mock uncertainty based on molecular complexity
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0.0)
                continue
            
            # Simple heuristic: more complex molecules have higher uncertainty
            complexity = (
                mol.GetNumAtoms() +
                mol.GetNumBonds() +
                len(Chem.GetSymmSSSR(mol))  # number of rings
            )
            
            # Add some randomness
            uncertainty = complexity / 50.0 + np.random.normal(0, 0.1)
            scores.append(max(0, uncertainty))
        
        return np.array(scores)
    
    def _calculate_diversity_scores(
        self,
        candidate_pool: List[str],
        evaluated_molecules: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Calculate diversity scores to promote chemical space exploration.
        """
        try:
            from rdkit.Chem import rdMolDescriptors
            from sklearn.metrics.pairwise import pairwise_distances
            
            # Calculate fingerprints for candidates
            candidate_fps = []
            for smiles in candidate_pool:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
                    candidate_fps.append(list(fp))
                else:
                    candidate_fps.append([0] * 2048)
            
            candidate_fps = np.array(candidate_fps)
            
            # Calculate fingerprints for evaluated molecules
            evaluated_fps = []
            for mol_data in evaluated_molecules:
                mol = Chem.MolFromSmiles(mol_data["smiles"])
                if mol is not None:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
                    evaluated_fps.append(list(fp))
            
            if len(evaluated_fps) == 0:
                # No evaluated molecules yet, return uniform scores
                return np.ones(len(candidate_pool))
            
            evaluated_fps = np.array(evaluated_fps)
            
            # Calculate distances to evaluated molecules
            distances = pairwise_distances(candidate_fps, evaluated_fps, metric='jaccard')
            
            # Diversity score is minimum distance to any evaluated molecule
            diversity_scores = np.min(distances, axis=1)
            
            return diversity_scores
            
        except ImportError:
            logger.warning("Scikit-learn not available for diversity calculation")
            return np.ones(len(candidate_pool))
        except Exception as e:
            logger.warning(f"Error calculating diversity scores: {e}")
            return np.ones(len(candidate_pool))
