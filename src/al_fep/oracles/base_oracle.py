"""
Base oracle class for molecular evaluation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
import time
from rdkit import Chem
import pandas as pd

logger = logging.getLogger(__name__)


class BaseOracle(ABC):
    """
    Abstract base class for molecular oracles.
    
    An oracle evaluates molecules and returns scores/properties.
    """
    
    def __init__(
        self,
        name: str,
        target: str,
        config: Optional[Dict[str, Any]] = None,
        cache: bool = True
    ):
        """
        Initialize the oracle.
        
        Args:
            name: Name of the oracle
            target: Target identifier (e.g., "7jvr")
            config: Configuration dictionary
            cache: Whether to cache results
        """
        self.name = name
        self.target = target
        self.config = config or {}
        self.cache = cache
        self.call_count = 0
        self.total_time = 0.0
        
        if self.cache:
            self._cache = {}
        
        logger.info(f"Initialized {self.name} oracle for target {self.target}")
    
    @abstractmethod
    def _evaluate_single(self, smiles: str) -> Dict[str, Any]:
        """
        Evaluate a single molecule.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary containing evaluation results
        """
        pass
    
    def _evaluate_batch(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of molecules.
        
        Default implementation evaluates molecules one by one.
        Override this method in subclasses that support efficient batch processing.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of evaluation results
        """
        results = []
        for smiles in smiles_list:
            result = self._evaluate_single(smiles)
            results.append(result)
        return results
    
    def supports_batch_processing(self) -> bool:
        """
        Check if this oracle supports efficient batch processing.
        
        Returns:
            True if the oracle overrides _evaluate_batch for efficiency
        """
        # Check if _evaluate_batch has been overridden
        return type(self)._evaluate_batch is not BaseOracle._evaluate_batch
    
    def evaluate(self, molecules: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluate one or more molecules.
        
        Args:
            molecules: SMILES string or list of SMILES strings
            
        Returns:
            Evaluation results (single dict or list of dicts)
        """
        if isinstance(molecules, str):
            return self._evaluate_with_cache(molecules)
        
        # For lists, decide between batch processing and individual evaluation
        if len(molecules) > 1 and self.supports_batch_processing():
            return self._evaluate_batch_with_cache(molecules)
        else:
            # Fall back to individual evaluation
            results = []
            for smiles in molecules:
                result = self._evaluate_with_cache(smiles)
                results.append(result)
            return results
    
    def _evaluate_with_cache(self, smiles: str) -> Dict[str, Any]:
        """
        Evaluate a molecule with caching support.
        """
        # Check cache first
        if self.cache and smiles in self._cache:
            logger.debug(f"Cache hit for {smiles}")
            return self._cache[smiles]
        
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return {
                "smiles": smiles,
                "score": None,
                "error": "Invalid SMILES",
                "oracle": self.name
            }
        
        # Canonicalize SMILES
        canonical_smiles = Chem.MolToSmiles(mol)
        
        # Check cache with canonical SMILES
        if self.cache and canonical_smiles in self._cache:
            logger.debug(f"Cache hit for canonical {canonical_smiles}")
            return self._cache[canonical_smiles]
        
        # Evaluate
        start_time = time.time()
        try:
            result = self._evaluate_single(canonical_smiles)
            result["smiles"] = canonical_smiles
            result["oracle"] = self.name
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Error evaluating {canonical_smiles}: {str(e)}")
            result = {
                "smiles": canonical_smiles,
                "score": None,
                "error": str(e),
                "oracle": self.name,
                "success": False
            }
        
        # Update statistics
        evaluation_time = time.time() - start_time
        result["evaluation_time"] = evaluation_time
        self.call_count += 1
        self.total_time += evaluation_time
        
        # Cache result
        if self.cache:
            self._cache[canonical_smiles] = result
        
        logger.debug(f"Evaluated {canonical_smiles} in {evaluation_time:.2f}s")
        return result
    
    def _evaluate_batch_with_cache(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of molecules with caching support.
        
        This method handles cache checking and SMILES validation before
        calling the oracle's batch evaluation method.
        """
        # Validate and canonicalize all SMILES first
        validated_smiles = []
        results = []
        indices_to_evaluate = []
        
        for i, smiles in enumerate(smiles_list):
            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                result = {
                    "smiles": smiles,
                    "score": None,
                    "error": "Invalid SMILES",
                    "oracle": self.name,
                    "success": False,
                    "evaluation_time": 0.0
                }
                results.append(result)
                validated_smiles.append(None)
                continue
            
            # Canonicalize SMILES
            canonical_smiles = Chem.MolToSmiles(mol)
            validated_smiles.append(canonical_smiles)
            
            # Check cache
            if self.cache and canonical_smiles in self._cache:
                logger.debug(f"Cache hit for {canonical_smiles}")
                results.append(self._cache[canonical_smiles])
            else:
                # Mark for batch evaluation
                results.append(None)  # Placeholder
                indices_to_evaluate.append(i)
        
        # Batch evaluate molecules not in cache
        if indices_to_evaluate:
            smiles_to_evaluate = [validated_smiles[i] for i in indices_to_evaluate]
            
            start_time = time.time()
            try:
                batch_results = self._evaluate_batch(smiles_to_evaluate)
                evaluation_time = time.time() - start_time
                
                # Post-process batch results
                for j, (original_idx, result) in enumerate(zip(indices_to_evaluate, batch_results)):
                    canonical_smiles = validated_smiles[original_idx]
                    
                    # Add metadata to result
                    result["smiles"] = canonical_smiles
                    result["oracle"] = self.name
                    result["success"] = True
                    result["evaluation_time"] = evaluation_time / len(batch_results)  # Approximate per-molecule time
                    
                    # Cache result
                    if self.cache:
                        self._cache[canonical_smiles] = result
                    
                    # Store in results
                    results[original_idx] = result
                
                # Update statistics
                self.call_count += len(batch_results)
                self.total_time += evaluation_time
                
                logger.debug(f"Batch evaluated {len(batch_results)} molecules in {evaluation_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in batch evaluation: {str(e)}")
                # Fill in error results for molecules that failed
                for original_idx in indices_to_evaluate:
                    canonical_smiles = validated_smiles[original_idx]
                    error_result = {
                        "smiles": canonical_smiles,
                        "score": None,
                        "error": str(e),
                        "oracle": self.name,
                        "success": False,
                        "evaluation_time": 0.0
                    }
                    results[original_idx] = error_result
        
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get oracle usage statistics.
        """
        return {
            "oracle": self.name,
            "target": self.target,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "avg_time": self.total_time / max(1, self.call_count),
            "cache_size": len(self._cache) if self.cache else 0
        }
    
    def clear_cache(self):
        """Clear the cache."""
        if self.cache:
            self._cache.clear()
            logger.info(f"Cleared cache for {self.name} oracle")
    
    def save_cache(self, filepath: str):
        """Save cache to file."""
        if self.cache and self._cache:
            df = pd.DataFrame(list(self._cache.values()))
            df.to_csv(filepath, index=False)
            logger.info(f"Saved cache to {filepath}")
    
    def load_cache(self, filepath: str):
        """Load cache from file."""
        if self.cache:
            try:
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    smiles = row["smiles"]
                    self._cache[smiles] = row.to_dict()
                logger.info(f"Loaded cache from {filepath}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}")
    
    def __str__(self):
        return f"{self.name}Oracle(target={self.target}, calls={self.call_count})"
    
    def __repr__(self):
        return self.__str__()
