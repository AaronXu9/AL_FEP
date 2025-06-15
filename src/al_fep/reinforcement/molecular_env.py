"""
Molecular environment for reinforcement learning
"""

import gym
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import logging

from ..oracles.base_oracle import BaseOracle

logger = logging.getLogger(__name__)


class MolecularEnvironment(gym.Env):
    """
    Gym environment for molecular generation and optimization.
    
    The agent learns to generate molecules by sequentially adding atoms/bonds
    and receives rewards based on oracle evaluations.
    """
    
    def __init__(
        self,
        oracle: BaseOracle,
        max_atoms: int = 50,
        target_properties: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize molecular environment.
        
        Args:
            oracle: Oracle for molecule evaluation
            max_atoms: Maximum number of atoms in a molecule
            target_properties: Target molecular properties
            config: Configuration dictionary
        """
        super().__init__()
        
        self.oracle = oracle
        self.max_atoms = max_atoms
        self.target_properties = target_properties or {}
        self.config = config or {}
        
        # Define action space (simplified)
        # Actions: add atom type (C, N, O, S, F, Cl, Br)
        self.action_space = gym.spaces.Discrete(7)
        
        # Define observation space (molecular fingerprint + properties)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(2048 + 10,), dtype=np.float32
        )
        
        # Internal state
        self.current_mol = None
        self.current_smiles = ""
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Atom mapping
        self.atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']
        
        logger.info(f"MolecularEnvironment initialized with oracle: {oracle.name}")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Start with a simple molecule (methane)
        self.current_mol = Chem.MolFromSmiles("C")
        self.current_smiles = "C"
        self.step_count = 0
        self.episode_reward = 0.0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (atom type to add)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.step_count += 1
        
        # Try to add atom to current molecule
        new_mol, success = self._add_atom(action)
        
        if success and new_mol is not None:
            self.current_mol = new_mol
            self.current_smiles = Chem.MolToSmiles(new_mol)
        
        # Calculate reward
        reward = self._calculate_reward(success)
        self.episode_reward += reward
        
        # Check if episode is done
        done = (
            self.step_count >= self.max_atoms or
            not success or
            self._is_terminal_molecule()
        )
        
        # Get new observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {
            "smiles": self.current_smiles,
            "valid_molecule": success,
            "num_atoms": self.current_mol.GetNumAtoms() if self.current_mol else 0,
            "episode_reward": self.episode_reward
        }
        
        return observation, reward, done, info
    
    def _add_atom(self, action: int) -> Tuple[Optional[Chem.Mol], bool]:
        """
        Add an atom to the current molecule.
        
        Args:
            action: Index of atom type to add
            
        Returns:
            Tuple of (new molecule, success flag)
        """
        if self.current_mol is None:
            return None, False
        
        try:
            # Get atom type
            atom_type = self.atom_types[action]
            
            # Create editable molecule
            em = Chem.EditableMol(self.current_mol)
            
            # Add new atom
            new_atom_idx = em.AddAtom(Chem.Atom(atom_type))
            
            # Connect to a random existing atom (simplified)
            if self.current_mol.GetNumAtoms() > 0:
                # Choose random existing atom to connect to
                existing_atoms = list(range(self.current_mol.GetNumAtoms()))
                if existing_atoms:
                    connect_to = np.random.choice(existing_atoms)
                    em.AddBond(connect_to, new_atom_idx, Chem.BondType.SINGLE)
            
            # Get new molecule
            new_mol = em.GetMol()
            
            # Sanitize molecule
            try:
                Chem.SanitizeMol(new_mol)
                return new_mol, True
            except:
                return None, False
                
        except Exception as e:
            logger.debug(f"Failed to add atom: {e}")
            return None, False
    
    def _calculate_reward(self, action_success: bool) -> float:
        """
        Calculate reward for the current state.
        
        Args:
            action_success: Whether the last action was successful
            
        Returns:
            Reward value
        """
        if not action_success or self.current_mol is None:
            return -1.0  # Penalty for invalid action
        
        # Base reward for valid molecule
        reward = 0.1
        
        # Oracle-based reward (expensive, use sparingly)
        if self._should_evaluate_with_oracle():
            try:
                oracle_result = self.oracle.evaluate(self.current_smiles)
                oracle_score = oracle_result.get("score", 0)
                
                if oracle_score is not None:
                    # Normalize oracle score to reasonable range
                    reward += oracle_score / 10.0
                
            except Exception as e:
                logger.debug(f"Oracle evaluation failed: {e}")
        
        # Property-based rewards
        reward += self._calculate_property_reward()
        
        # Diversity reward (encourage exploration)
        reward += self._calculate_diversity_reward()
        
        return reward
    
    def _should_evaluate_with_oracle(self) -> bool:
        """Decide whether to evaluate current molecule with oracle."""
        # Only evaluate occasionally to save computational cost
        # Evaluate at end of episode or every 10 steps
        return (
            self.step_count % 10 == 0 or
            self._is_terminal_molecule()
        )
    
    def _calculate_property_reward(self) -> float:
        """Calculate reward based on molecular properties."""
        try:
            mol = self.current_mol
            if mol is None:
                return 0.0
            
            reward = 0.0
            
            # Molecular weight
            mw = Chem.Descriptors.MolWt(mol)
            if 200 <= mw <= 500:  # Drug-like range
                reward += 0.1
            
            # LogP
            logp = Chem.Descriptors.MolLogP(mol)
            if 0 <= logp <= 5:  # Drug-like range
                reward += 0.1
            
            # Number of rings (prefer some rings)
            num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
            if 1 <= num_rings <= 3:
                reward += 0.1
            
            return reward
            
        except Exception as e:
            logger.debug(f"Error calculating property reward: {e}")
            return 0.0
    
    def _calculate_diversity_reward(self) -> float:
        """Calculate reward for molecular diversity."""
        # Simple diversity reward based on uniqueness
        # In practice, this would compare against a database of known molecules
        return 0.05 if len(self.current_smiles) > 5 else 0.0
    
    def _is_terminal_molecule(self) -> bool:
        """Check if current molecule should terminate the episode."""
        if self.current_mol is None:
            return True
        
        # Terminate if molecule is too large
        if self.current_mol.GetNumAtoms() >= self.max_atoms:
            return True
        
        # Terminate if molecule meets certain criteria
        # (e.g., good drug-likeness score)
        return False
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (molecular fingerprint + properties).
        
        Returns:
            Observation vector
        """
        if self.current_mol is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        try:
            # Morgan fingerprint (2048 bits)
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                self.current_mol, 2, nBits=2048
            )
            fp_array = np.array(list(fp), dtype=np.float32)
            
            # Basic molecular properties (10 features)
            props = np.array([
                self.current_mol.GetNumAtoms() / self.max_atoms,  # normalized
                self.current_mol.GetNumBonds() / (self.max_atoms * 2),  # normalized
                Chem.Descriptors.MolWt(self.current_mol) / 500.0,  # normalized
                Chem.Descriptors.MolLogP(self.current_mol) / 5.0,  # normalized
                Chem.Descriptors.NumHDonors(self.current_mol) / 10.0,
                Chem.Descriptors.NumHAcceptors(self.current_mol) / 10.0,
                Chem.rdMolDescriptors.CalcNumRings(self.current_mol) / 5.0,
                Chem.rdMolDescriptors.CalcNumAromaticRings(self.current_mol) / 3.0,
                Chem.Descriptors.TPSA(self.current_mol) / 200.0,
                self.step_count / self.max_atoms  # episode progress
            ], dtype=np.float32)
            
            # Combine fingerprint and properties
            observation = np.concatenate([fp_array, props])
            
            return observation
            
        except Exception as e:
            logger.debug(f"Error getting observation: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def render(self, mode='human'):
        """Render the current molecule."""
        if mode == 'human':
            print(f"Step: {self.step_count}")
            print(f"SMILES: {self.current_smiles}")
            print(f"Atoms: {self.current_mol.GetNumAtoms() if self.current_mol else 0}")
            print(f"Reward: {self.episode_reward:.3f}")
            print("-" * 40)
    
    def close(self):
        """Clean up environment."""
        pass
