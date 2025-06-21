"""
Docking oracle supporting AutoDock Vina and GNINA
"""

import os
import tempfile
import subprocess
from typing import Dict, Any, Optional
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import numpy as np

from .base_oracle import BaseOracle

logger = logging.getLogger(__name__)


class DockingOracle(BaseOracle):
    """
    Oracle for molecular docking using AutoDock Vina or GNINA.
    
    Supports both AutoDock Vina and GNINA docking engines with configurable
    scoring functions and neural network-based scoring for GNINA.
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
        
        # Get receptor file from config or parameter or default
        docking_config = self.config.get("docking", {})
        self.receptor_file = (
            receptor_file or 
            docking_config.get("receptor_file") or 
            self._get_default_receptor_file()
        )
        
        # Docking engine (vina or gnina)
        self.engine = docking_config.get("engine", "vina").lower()
        
        # Docking parameters from config
        self.center_x = docking_config.get("center_x", 0.0)
        self.center_y = docking_config.get("center_y", 0.0) 
        self.center_z = docking_config.get("center_z", 0.0)
        self.size_x = docking_config.get("size_x", 20.0)
        self.size_y = docking_config.get("size_y", 20.0)
        self.size_z = docking_config.get("size_z", 20.0)
        self.exhaustiveness = docking_config.get("exhaustiveness", 8)
        self.num_poses = docking_config.get("num_poses", 9)
        self.center_y = docking_config.get("center_y", 0.0) 
        self.center_z = docking_config.get("center_z", 0.0)
        self.size_x = docking_config.get("size_x", 20.0)
        self.size_y = docking_config.get("size_y", 20.0)
        self.size_z = docking_config.get("size_z", 20.0)
        self.exhaustiveness = docking_config.get("exhaustiveness", 8)
        self.num_poses = docking_config.get("num_poses", 9)
        
        # Mock mode for testing
        self.mock_mode = docking_config.get("mock_mode", False)
        
        # Check if docking engine is available
        if not self.mock_mode:
            self._check_docking_installation()
        
        logger.info(f"DockingOracle ({self.engine.upper()}) initialized with receptor: {self.receptor_file}")
    
    def _get_default_receptor_file(self) -> str:
        """Get default receptor file path."""
        return f"data/targets/{self.target}/{self.target}_prepared.pdbqt"
    
    def _check_docking_installation(self):
        """Check if the selected docking engine is available."""
        if self.engine == "vina":
            self._check_vina_installation()
        elif self.engine == "gnina":
            self._check_gnina_installation()
        else:
            raise ValueError(f"Unsupported docking engine: {self.engine}")
    
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
    
    def _check_gnina_installation(self):
        """Check if GNINA is installed."""
        try:
            result = subprocess.run(
                ["/home/aoxu/projects/PoseBench/forks/GNINA/gnina", "--help"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("GNINA not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError):
            logger.warning(
                "GNINA not found or not working. "
                "Please check the GNINA installation path."
            )
            raise RuntimeError("GNINA not available")
    
    def _smiles_to_pdbqt(self, smiles: str, output_file: str) -> bool:
        """
        Convert SMILES to PDBQT format using obabel directly.
        
        Args:
            smiles: SMILES string
            output_file: Output PDBQT file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use obabel directly for SMILES to PDBQT conversion
            cmd = [
                "obabel",
                "-ismi",
                "-opdbqt", 
                "--gen3d",
                "--partialcharge", "gasteiger",
                "-O", output_file
            ]
            
            # Run obabel with SMILES as input
            result = subprocess.run(
                cmd,
                input=smiles,
                text=True,
                capture_output=True,
                check=True
            )
            
            # Check if output file was created
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return True
            else:
                logger.warning(f"PDBQT file not created or empty: {output_file}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to convert SMILES to PDBQT: {e}")
            return False
        except Exception as e:
            logger.error(f"Error in SMILES to PDBQT conversion: {e}")
            return False

    def _smiles_to_pdb(self, smiles: str, output_file: str) -> bool:
        """
        Convert SMILES to PDB format using obabel directly.
        
        Args:
            smiles: SMILES string
            output_file: Output PDB file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use obabel directly for SMILES to PDB conversion
            cmd = [
                "obabel",
                "-ismi",
                "-opdb", 
                "--gen3d",
                "--partialcharge", "gasteiger",
                "-O", output_file
            ]
            
            # Run obabel with SMILES as input
            result = subprocess.run(
                cmd,
                input=smiles,
                text=True,
                capture_output=True,
                check=True
            )
            
            # Check if output file was created
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return True
            else:
                logger.warning(f"PDB file not created or empty: {output_file}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to convert SMILES to PDB: {e}")
            return False
        except Exception as e:
            logger.error(f"Error in SMILES to PDB conversion: {e}")
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
            
            # Run Vina (without --log parameter, capture stdout instead)
            cmd = [
                "vina",
                "--receptor", self.receptor_file,
                "--ligand", ligand_file,
                "--out", output_file,
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
            
            # Parse the best score from the output PDBQT file (not stdout)
            best_score = self._parse_vina_pdbqt_output(output_file)
            
            # Clean up output file
            if os.path.exists(output_file):
                os.remove(output_file)
            
            return best_score
            
        except subprocess.TimeoutExpired:
            logger.warning("Vina docking timed out")
            return None
        except Exception as e:
            logger.error(f"Error running Vina: {e}")
            return None
    
    def _parse_vina_pdbqt_output(self, pdbqt_file: str) -> Optional[float]:
        """
        Parse the best docking score from Vina output PDBQT file.
        
        Args:
            pdbqt_file: Path to Vina output PDBQT file
            
        Returns:
            Best docking score or None if parsing failed
        """
        try:
            if not os.path.exists(pdbqt_file):
                logger.warning(f"Output PDBQT file not found: {pdbqt_file}")
                return None
                
            with open(pdbqt_file, 'r') as f:
                for line in f:
                    # Look for REMARK VINA RESULT lines which contain the scores
                    if line.startswith('REMARK VINA RESULT:'):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            # The score is typically the 4th element (index 3)
                            score = float(parts[3])
                            return score
            
            logger.warning(f"Could not find REMARK VINA RESULT in {pdbqt_file}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing Vina PDBQT output: {e}")
            return None

    def _parse_vina_output(self, output_text: str) -> Optional[float]:
        """
        Parse the best docking score from Vina stdout output.
        
        Args:
            output_text: Stdout output from Vina
            
        Returns:
            Best docking score or None if parsing failed
        """
        try:
            lines = output_text.split('\n')
            
            # Look for the results table
            for i, line in enumerate(lines):
                if "mode |   affinity | dist from best mode" in line:
                    # Next line should have the best score
                    if i + 2 < len(lines):
                        score_line = lines[i + 2]
                        parts = score_line.split()
                        if len(parts) >= 2:
                            return float(parts[1])
            
            logger.warning("Could not parse Vina score from output")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing Vina output: {e}")
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
    
    def _run_gnina_docking(self, ligand_file: str) -> Optional[float]:
        """
        Run GNINA docking.
        
        Args:
            ligand_file: Path to ligand PDB file (GNINA expects PDB format)
            
        Returns:
            Best CNNscore or None if failed
        """
        try:
            # Create output file as SDF (GNINA's native format for CNN scores)
            output_file = ligand_file.replace(".pdb", "_out.sdf")
            
            # Run GNINA with SDF output to get CNN scores
            cmd = [
                "/home/aoxu/projects/PoseBench/forks/GNINA/gnina",
                "--receptor", self.receptor_file,
                "--ligand", ligand_file,  # PDB file for ligand
                "--out", output_file,     # Output as SDF for CNN scores
                "--center_x", str(self.center_x),
                "--center_y", str(self.center_y), 
                "--center_z", str(self.center_z),
                "--size_x", str(self.size_x),
                "--size_y", str(self.size_y),
                "--size_z", str(self.size_z),
                "--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", str(self.num_poses),
                "--seed", "42",  # For reproducibility
                "--cnn_scoring", "rescore"  # Use CNN for rescoring poses
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout (GNINA can be slower)
            )
            
            if result.returncode != 0:
                logger.warning(f"GNINA failed: {result.stderr}")
                return None
            
            # Parse the best CNNscore from the output SDF file
            best_score = self._parse_gnina_sdf_output(output_file)
            
            # Clean up output file
            if os.path.exists(output_file):
                os.remove(output_file)
            
            return best_score
            
        except subprocess.TimeoutExpired:
            logger.warning("GNINA docking timed out")
            return None
        except Exception as e:
            logger.error(f"Error running GNINA: {e}")
            return None

    def _parse_gnina_sdf_output(self, sdf_file: str) -> Optional[float]:
        """
        Parse the best CNNscore from GNINA output SDF file.
        
        GNINA outputs SDF files with molecular properties including CNNscore.
        Higher CNNscore values indicate better binding affinity.
        
        Args:
            sdf_file: Path to GNINA output SDF file
            
        Returns:
            Best CNNscore or None if parsing failed
        """
        try:
            if not os.path.exists(sdf_file):
                logger.warning(f"Output SDF file not found: {sdf_file}")
                return None
            
            # Use RDKit to read SDF and extract CNNscore properties
            supplier = Chem.SDMolSupplier(sdf_file)
            best_cnn_score = None
            
            for mol in supplier:
                if mol is None:
                    continue
                    
                # Look for CNNscore property
                if mol.HasProp('CNNscore'):
                    try:
                        cnn_score = float(mol.GetProp('CNNscore'))
                        # Higher CNNscore is better for GNINA (opposite of Vina)
                        if best_cnn_score is None or cnn_score > best_cnn_score:
                            best_cnn_score = cnn_score
                    except (ValueError, TypeError):
                        continue
                        
                # Also check for alternative property names just in case
                elif mol.HasProp('cnn_score'):
                    try:
                        cnn_score = float(mol.GetProp('cnn_score'))
                        if best_cnn_score is None or cnn_score > best_cnn_score:
                            best_cnn_score = cnn_score
                    except (ValueError, TypeError):
                        continue
            
            if best_cnn_score is not None:
                logger.debug(f"Best CNNscore from GNINA: {best_cnn_score}")
                return best_cnn_score
            else:
                logger.warning(f"Could not find CNNscore in {sdf_file}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing GNINA SDF output: {e}")
            return None

    def _evaluate_single(self, smiles: str) -> Dict[str, Any]:
        """
        Evaluate a single molecule using docking.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary containing docking results
        """
        # Mock mode for testing
        if self.mock_mode:
            return self._evaluate_single_mock(smiles)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Choose the appropriate file format based on engine
            if self.engine == "vina":
                ligand_file = os.path.join(temp_dir, "ligand.pdbqt")
                conversion_success = self._smiles_to_pdbqt(smiles, ligand_file)
            elif self.engine == "gnina":
                ligand_file = os.path.join(temp_dir, "ligand.pdb")
                conversion_success = self._smiles_to_pdb(smiles, ligand_file)
            else:
                return {
                    "score": None,
                    "error": f"Unsupported docking engine: {self.engine}",
                    "docking_score": None,
                    "binding_affinity": None,
                    "method": f"Docking ({self.engine.upper()})"
                }
            
            # Check if conversion was successful
            if not conversion_success:
                file_type = "PDBQT" if self.engine == "vina" else "PDB"
                return {
                    "score": None,
                    "error": f"Failed to convert SMILES to {file_type}",
                    "docking_score": None,
                    "binding_affinity": None,
                    "method": f"Docking ({self.engine.upper()})"
                }
            
            # Check if receptor file exists
            if not os.path.exists(self.receptor_file):
                return {
                    "score": None,
                    "error": f"Receptor file not found: {self.receptor_file}",
                    "docking_score": None,
                    "binding_affinity": None,
                    "method": f"Docking ({self.engine.upper()})"
                }
            
            # Run docking based on selected engine
            if self.engine == "vina":
                docking_score = self._run_vina_docking(ligand_file)
            elif self.engine == "gnina":
                docking_score = self._run_gnina_docking(ligand_file)
            else:
                return {
                    "score": None,
                    "error": f"Unsupported docking engine: {self.engine}",
                    "docking_score": None,
                    "binding_affinity": None,
                    "method": f"Docking ({self.engine.upper()})"
                }
            
            if docking_score is None:
                return {
                    "score": None,
                    "error": "Docking failed",
                    "docking_score": None,
                    "binding_affinity": None,
                    "method": f"Docking ({self.engine.upper()})"
                }
            
            # Convert to positive score (lower is better in docking)
            score = -docking_score
            
            return {
                "score": score,
                "docking_score": docking_score,
                "binding_affinity": docking_score,
                "error": None,
                "method": f"Docking ({self.engine.upper()})"
            }
    
    def _evaluate_single_mock(self, smiles: str) -> Dict[str, Any]:
        """
        Mock evaluation for testing purposes.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary containing mock docking results
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    "score": None,
                    "error": "Invalid SMILES",
                    "docking_score": None,
                    "binding_affinity": None,
                    "method": f"Docking ({self.engine.upper()} Mock)"
                }
            
            # Mock docking score based on molecular properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            
            # Simple scoring function for mock
            mock_score = -6.0 + (300 - mw) * 0.01 - logp * 0.5
            mock_score += np.random.normal(0, 0.5)  # Add some noise
            
            # Ensure reasonable range
            mock_score = max(-12.0, min(-3.0, mock_score))
            
            return {
                "score": -mock_score,  # Convert to positive score
                "docking_score": mock_score,
                "binding_affinity": mock_score,
                "error": None,
                "method": f"Docking ({self.engine.upper()} Mock)"
            }
            
        except Exception as e:
            return {
                "score": None,
                "error": f"Mock evaluation failed: {str(e)}",
                "docking_score": None,
                "binding_affinity": None,
                "method": f"Docking ({self.engine.upper()} Mock)"
            }
                           