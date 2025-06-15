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
