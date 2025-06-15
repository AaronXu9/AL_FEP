"""
Molecular dataset class for handling SMILES and molecular data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging

logger = logging.getLogger(__name__)


class MolecularDataset:
    """
    Dataset class for handling molecular data and SMILES.
    """
    
    def __init__(
        self,
        smiles: Optional[List[str]] = None,
        properties: Optional[Dict[str, List]] = None,
        name: str = "MolecularDataset"
    ):
        """
        Initialize molecular dataset.
        
        Args:
            smiles: List of SMILES strings
            properties: Dictionary of molecular properties
            name: Dataset name
        """
        self.name = name
        self.data = pd.DataFrame()
        
        if smiles is not None:
            self.add_molecules(smiles, properties)
        
        logger.info(f"MolecularDataset '{name}' initialized")
    
    def add_molecules(
        self,
        smiles: List[str],
        properties: Optional[Dict[str, List]] = None
    ):
        """
        Add molecules to the dataset.
        
        Args:
            smiles: List of SMILES strings
            properties: Dictionary of additional properties
        """
        # Validate and canonicalize SMILES
        valid_data = []
        
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol)
                
                row_data = {"smiles": canonical_smiles}
                
                # Add properties if provided
                if properties:
                    for prop_name, prop_values in properties.items():
                        if i < len(prop_values):
                            row_data[prop_name] = prop_values[i]
                
                valid_data.append(row_data)
            else:
                logger.warning(f"Invalid SMILES skipped: {smi}")
        
        # Add to dataframe
        new_df = pd.DataFrame(valid_data)
        self.data = pd.concat([self.data, new_df], ignore_index=True)
        
        # Remove duplicates
        self.data = self.data.drop_duplicates(subset=['smiles']).reset_index(drop=True)
        
        logger.info(f"Added {len(valid_data)} molecules to dataset")
    
    def calculate_descriptors(self, descriptor_names: Optional[List[str]] = None):
        """
        Calculate molecular descriptors for all molecules.
        
        Args:
            descriptor_names: List of descriptor names to calculate
        """
        if descriptor_names is None:
            descriptor_names = [
                'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
                'NumRotatableBonds', 'TPSA', 'NumAromaticRings'
            ]
        
        for desc_name in descriptor_names:
            if hasattr(Descriptors, desc_name):
                descriptor_func = getattr(Descriptors, desc_name)
                
                values = []
                for smiles in self.data['smiles']:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        try:
                            value = descriptor_func(mol)
                            values.append(value)
                        except:
                            values.append(None)
                    else:
                        values.append(None)
                
                self.data[desc_name] = values
            else:
                logger.warning(f"Unknown descriptor: {desc_name}")
        
        logger.info(f"Calculated {len(descriptor_names)} descriptors")
    
    def apply_filters(self, filters: Dict[str, Tuple[float, float]]):
        """
        Apply filters to the dataset.
        
        Args:
            filters: Dictionary of {property: (min_val, max_val)}
        """
        initial_size = len(self.data)
        
        for prop, (min_val, max_val) in filters.items():
            if prop in self.data.columns:
                mask = (
                    (self.data[prop] >= min_val) & 
                    (self.data[prop] <= max_val) &
                    (self.data[prop].notna())
                )
                self.data = self.data[mask].reset_index(drop=True)
        
        final_size = len(self.data)
        logger.info(f"Filters applied: {initial_size} -> {final_size} molecules")
    
    def get_smiles(self) -> List[str]:
        """Get list of SMILES strings."""
        return self.data['smiles'].tolist()
    
    def get_properties(self, property_names: List[str]) -> pd.DataFrame:
        """Get specified properties as DataFrame."""
        available_props = [p for p in property_names if p in self.data.columns]
        return self.data[available_props].copy()
    
    def sample(self, n: int, random_state: Optional[int] = None) -> 'MolecularDataset':
        """
        Sample n molecules from the dataset.
        
        Args:
            n: Number of molecules to sample
            random_state: Random seed
            
        Returns:
            New MolecularDataset with sampled molecules
        """
        if n >= len(self.data):
            return self.copy()
        
        sampled_data = self.data.sample(n=n, random_state=random_state)
        
        new_dataset = MolecularDataset(name=f"{self.name}_sampled_{n}")
        new_dataset.data = sampled_data.reset_index(drop=True)
        
        return new_dataset
    
    def split(
        self,
        train_size: float = 0.8,
        random_state: Optional[int] = None
    ) -> Tuple['MolecularDataset', 'MolecularDataset']:
        """
        Split dataset into train and test sets.
        
        Args:
            train_size: Fraction for training set
            random_state: Random seed
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        shuffled_data = self.data.sample(frac=1, random_state=random_state)
        
        n_train = int(len(shuffled_data) * train_size)
        
        train_data = shuffled_data.iloc[:n_train].reset_index(drop=True)
        test_data = shuffled_data.iloc[n_train:].reset_index(drop=True)
        
        train_dataset = MolecularDataset(name=f"{self.name}_train")
        train_dataset.data = train_data
        
        test_dataset = MolecularDataset(name=f"{self.name}_test")
        test_dataset.data = test_data
        
        return train_dataset, test_dataset
    
    def copy(self) -> 'MolecularDataset':
        """Create a copy of the dataset."""
        new_dataset = MolecularDataset(name=f"{self.name}_copy")
        new_dataset.data = self.data.copy()
        return new_dataset
    
    def save(self, filepath: str):
        """Save dataset to file."""
        self.data.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, name: Optional[str] = None) -> 'MolecularDataset':
        """
        Load dataset from file.
        
        Args:
            filepath: Path to CSV file
            name: Dataset name
            
        Returns:
            MolecularDataset object
        """
        data = pd.read_csv(filepath)
        
        if name is None:
            name = f"Dataset_{filepath.split('/')[-1]}"
        
        dataset = cls(name=name)
        dataset.data = data
        
        logger.info(f"Dataset loaded from {filepath}")
        return dataset
    
    def __len__(self):
        """Get dataset size."""
        return len(self.data)
    
    def __repr__(self):
        """String representation."""
        return f"MolecularDataset('{self.name}', {len(self.data)} molecules)"
