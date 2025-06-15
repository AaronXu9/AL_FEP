"""
Docking oracle using AutoDock Vina
"""

import os
import tempfile
import subprocess
from typing import Dict, Any, Optional
import logging
from rdkit import Chem
from rdkit.Chem import AllChem

from .base_oracle import BaseOracle

logger = logging.getLogger(__name__)


class DockingOracle(BaseOracle):
    """
    Oracle for molecular docking using AutoDock Vina.
    """
    
    def __init__(
        self,
        target: str,
        receptor_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the docking oracle.
        
        Args:
            target: Target identifier
            receptor_file: Path to receptor PDBQT file
            config: Configuration dictionary
        """
        super().__init__(name="Docking", target=target, config=config, **kwargs)
        
        self.receptor_file = receptor_file or self._get_default_receptor_file()
        
        # Docking parameters from config
        docking_config = self.config.get("docking", {})
        self.center_x = docking_config.get("center_x", 0.0)
        self.center_y = docking_config.get("center_y", 0.0) 
        self.center_z = docking_config.get("center_z", 0.0)
        self.size_x = docking_config.get("size_x", 20.0)
        self.size_y = docking_config.get("size_y", 20.0)
        self.size_z = docking_config.get("size_z", 20.0)
        self.exhaustiveness = docking_config.get("exhaustiveness", 8)
        self.num_poses = docking_config.get("num_poses", 9)
        
        # Check if Vina is available
        self._check_vina_installation()
        
        logger.info(f"DockingOracle initialized with receptor: {self.receptor_file}")
    
    def _get_default_receptor_file(self) -> str:
        """Get default receptor file path."""
        return f"data/targets/{self.target}/{self.target}_prepared.pdbqt"
    
    def _check_vina_installation(self):
        """Check if AutoDock Vina is installed."""
        try:
            result = subprocess.run(
                ["vina", "--help"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("Vina not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError):
            logger.warning(
                "AutoDock Vina not found or not working. "
                "Install with: conda install -c conda-forge vina"
            )
            raise RuntimeError("AutoDock Vina not available")
    
    def _smiles_to_pdbqt(self, smiles: str, output_file: str) -> bool:
        """
        Convert SMILES to PDBQT format for docking.
        
        Args:
            smiles: Input SMILES string
            output_file: Output PDBQT file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            
            if AllChem.EmbedMolecule(mol, params) != 0:
                # Try again with different method
                if AllChem.EmbedMolecule(mol) != 0:
                    logger.warning(f"Failed to generate 3D coordinates for {smiles}")
                    return False
            
            # Optimize geometry
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Write to SDF file first
            sdf_file = output_file.replace(".pdbqt", ".sdf")
            writer = Chem.SDWriter(sdf_file)
            writer.write(mol)
            writer.close()
            
            # Convert SDF to PDBQT using obabel
            try:
                subprocess.run([
                    "obabel", 
                    "-isdf", sdf_file,
                    "-opdbqt", output_file,
                    "--partialcharge", "gasteiger"
                ], check=True, capture_output=True)
                
                # Clean up SDF file
                os.remove(sdf_file)
                return True
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to convert to PDBQT: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error in SMILES to PDBQT conversion: {e}")
            return False
    
    def _run_vina_docking(self, ligand_file: str) -> Optional[float]:
        """
        Run AutoDock Vina docking.
        
        Args:
            ligand_file: Path to ligand PDBQT file
            
        Returns:
            Best docking score or None if failed
        """
        try:
            # Create output file
            output_file = ligand_file.replace(".pdbqt", "_out.pdbqt")
            log_file = ligand_file.replace(".pdbqt", "_log.txt")
            
            # Run Vina
            cmd = [
                "vina",
                "--receptor", self.receptor_file,
                "--ligand", ligand_file,
                "--out", output_file,
                "--log", log_file,
                "--center_x", str(self.center_x),
                "--center_y", str(self.center_y), 
                "--center_z", str(self.center_z),
                "--size_x", str(self.size_x),
                "--size_y", str(self.size_y),
                "--size_z", str(self.size_z),
                "--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", str(self.num_poses)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.warning(f"Vina failed: {result.stderr}")
                return None
            
            # Parse the best score from log file
            best_score = self._parse_vina_log(log_file)
            
            # Clean up files
            for f in [output_file, log_file]:
                if os.path.exists(f):
                    os.remove(f)
            
            return best_score
            
        except subprocess.TimeoutExpired:
            logger.warning("Vina docking timed out")
            return None
        except Exception as e:
            logger.error(f"Error running Vina: {e}")
            return None
    
    def _parse_vina_log(self, log_file: str) -> Optional[float]:
        """
        Parse the best docking score from Vina log file.
        
        Args:
            log_file: Path to Vina log file
            
        Returns:
            Best docking score or None if parsing failed
        """
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Look for the results table
            for i, line in enumerate(lines):
                if "mode |   affinity | dist from best mode" in line:
                    # Next line should have the best score
                    if i + 2 < len(lines):
                        score_line = lines[i + 2]
                        parts = score_line.split()
                        if len(parts) >= 2:
                            return float(parts[1])
            
            logger.warning("Could not parse Vina score from log file")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing Vina log: {e}")
            return None
    
    def _evaluate_single(self, smiles: str) -> Dict[str, Any]:
        """
        Evaluate a single molecule using docking.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary containing docking results
        """
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            ligand_file = os.path.join(temp_dir, "ligand.pdbqt")
            
            # Convert SMILES to PDBQT
            if not self._smiles_to_pdbqt(smiles, ligand_file):
                return {
                    "score": None,
                    "error": "Failed to convert SMILES to PDBQT",
                    "docking_score": None,
                    "binding_affinity": None
                }
            
            # Check if receptor file exists
            if not os.path.exists(self.receptor_file):
                return {
                    "score": None,
                    "error": f"Receptor file not found: {self.receptor_file}",
                    "docking_score": None,
                    "binding_affinity": None
                }
            
            # Run docking
            docking_score = self._run_vina_docking(ligand_file)
            
            if docking_score is None:
                return {
                    "score": None,
                    "error": "Docking failed",
                    "docking_score": None,
                    "binding_affinity": None
                }
            
            # Convert to positive score (lower is better in Vina)
            score = -docking_score
            
            return {
                "score": score,
                "docking_score": docking_score,
                "binding_affinity": docking_score,
                "error": None
            }
