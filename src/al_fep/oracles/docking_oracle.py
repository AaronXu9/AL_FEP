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
    
    def _smiles_to_sdf(self, smiles: str, output_file: str) -> bool:
        """
        Convert SMILES to SDF format using RDKit and obabel.
        
        Args:
            smiles: SMILES string
            output_file: Output SDF file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use RDKit for initial conversion and 3D generation
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            if AllChem.EmbedMolecule(mol) != 0:
                # If embedding fails, try with random coordinates
                AllChem.EmbedMolecule(mol, useRandomCoords=True)
            
            # Optimize geometry
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Write to SDF file
            writer = Chem.SDWriter(output_file)
            writer.write(mol)
            writer.close()
            
            return os.path.exists(output_file) and os.path.getsize(output_file) > 0
            
        except Exception as e:
            logger.error(f"Error converting SMILES to SDF: {e}")
            # Fallback to obabel method
            try:
                cmd = [
                    "obabel",
                    "-ismi",
                    "-osdf", 
                    "--gen3d",
                    "-O", output_file
                ]
                
                result = subprocess.run(
                    cmd,
                    input=smiles,
                    text=True,
                    capture_output=True,
                    check=True
                )
                
                return os.path.exists(output_file) and os.path.getsize(output_file) > 0
                
            except Exception as e2:
                logger.error(f"Fallback conversion also failed: {e2}")
                return False
                
        except Exception as e2:
            logger.error(f"Fallback conversion also failed: {e2}")
            return False
    
    def _run_vina_docking(self, ligand_file: str) -> Optional[float]:
        """
        Run AutoDock Vina docking with SDF input.
        
        Args:
            ligand_file: Path to ligand SDF file
            
        Returns:
            Best docking score or None if failed
        """
        try:
            # Create output file
            output_file = ligand_file.replace(".sdf", "_out.sdf")
            
            # Run Vina with SDF input and output
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
            
            # Parse the best score from the output SDF file
            best_score = self._parse_vina_sdf_output(output_file)
            
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
    
    def _parse_vina_sdf_output(self, sdf_file: str) -> Optional[float]:
        """
        Parse the best docking score from Vina SDF output file.
        
        Args:
            sdf_file: Path to Vina output SDF file
            
        Returns:
            Best docking score or None if parsing failed
        """
        try:
            # Use RDKit to read SDF file and extract score
            supplier = Chem.SDMolSupplier(sdf_file)
            best_score = None
            
            for mol in supplier:
                if mol is None:
                    continue
                    
                # Check for common Vina score property names
                score = None
                for prop_name in ['vina_affinity', 'VINA_AFFINITY', 'affinity', 'AFFINITY']:
                    if mol.HasProp(prop_name):
                        try:
                            score = float(mol.GetProp(prop_name))
                            break
                        except (ValueError, TypeError):
                            continue
                
                if score is not None:
                    if best_score is None or score < best_score:  # Lower is better for Vina
                        best_score = score
            
            return best_score
            
        except Exception as e:
            logger.error(f"Error parsing Vina SDF output: {e}")
            return None
    
    def _run_gnina_docking(self, ligand_file: str) -> Optional[float]:
        """
        Run GNINA docking with SDF input.
        
        Args:
            ligand_file: Path to ligand SDF file
            
        Returns:
            Best CNNscore or None if failed
        """
        try:
            # Create output file as SDF
            output_file = ligand_file.replace(".sdf", "_out.sdf")
            
            # Run GNINA with SDF input and output
            cmd = [
                "/home/aoxu/projects/PoseBench/forks/GNINA/gnina",
                "--receptor", self.receptor_file,
                "--ligand", ligand_file,  # SDF file for ligand
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

    def _evaluate_batch(self, smiles_list: list) -> list:
        """
        Evaluate a batch of molecules using docking for improved efficiency.
        
        This method processes multiple molecules in a single docking run,
        which is more efficient than individual docking calls.
        
        Args:
            smiles_list: List of SMILES strings to evaluate
            
        Returns:
            List of dictionaries containing docking results
        """
        # Mock mode for testing
        if self.mock_mode:
            return [self._evaluate_single_mock(smiles) for smiles in smiles_list]
        
        # Check if receptor file exists upfront
        if not os.path.exists(self.receptor_file):
            error_result = {
                "score": None,
                "error": f"Receptor file not found: {self.receptor_file}",
                "docking_score": None,
                "binding_affinity": None,
                "method": f"Docking ({self.engine.upper()})"
            }
            return [error_result.copy() for _ in smiles_list]
        
        # Create temporary directory for batch processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert all SMILES to SDF format and collect valid ones
            ligand_files = []
            smiles_indices = []  # Track which SMILES are valid
            
            for i, smiles in enumerate(smiles_list):
                ligand_file = os.path.join(temp_dir, f"ligand_{i}.sdf")
                conversion_success = self._smiles_to_sdf(smiles, ligand_file)
                
                if conversion_success:
                    ligand_files.append(ligand_file)
                    smiles_indices.append(i)
            
            # If no molecules were successfully converted, return errors
            if not ligand_files:
                error_result = {
                    "score": None,
                    "error": "Failed to convert any SMILES to SDF",
                    "docking_score": None,
                    "binding_affinity": None,
                    "method": f"Docking ({self.engine.upper()})"
                }
                return [error_result.copy() for _ in smiles_list]
            
            # Run batch docking - simplified approach
            docking_scores = self._run_batch_docking(ligand_files)
            
            # Process results
            results = []
            docking_idx = 0
            
            for i, smiles in enumerate(smiles_list):
                if i in smiles_indices:
                    # This molecule was successfully processed
                    docking_score = docking_scores[docking_idx] if docking_idx < len(docking_scores) else None
                    docking_idx += 1
                    
                    if docking_score is None:
                        result = {
                            "score": None,
                            "error": "Docking failed",
                            "docking_score": None,
                            "binding_affinity": None,
                            "method": f"Docking ({self.engine.upper()})"
                        }
                    else:
                        # Convert to positive score (lower is better in docking)
                        score = -docking_score
                        result = {
                            "score": score,
                            "docking_score": docking_score,
                            "binding_affinity": docking_score,
                            "error": None,
                            "method": f"Docking ({self.engine.upper()})"
                        }
                else:
                    # This molecule failed conversion
                    file_type = "PDBQT" if self.engine == "vina" else "PDB"
                    result = {
                        "score": None,
                        "error": f"Failed to convert SMILES to {file_type}",
                        "docking_score": None,
                        "binding_affinity": None,
                        "method": f"Docking ({self.engine.upper()})"
                    }
                
                results.append(result)
            
            return results
    
    def _run_batch_docking(self, ligand_files: list) -> list:
        """
        Run batch docking for multiple ligands - simplified approach.
        
        Args:
            ligand_files: List of SDF ligand file paths
            
        Returns:
            List of docking scores
        """
        scores = []
        for ligand_file in ligand_files:
            if self.engine == "vina":
                score = self._run_vina_docking(ligand_file)
            elif self.engine == "gnina":
                score = self._run_gnina_docking(ligand_file)
            else:
                score = None
            scores.append(score)
        return scores

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
            # Convert SMILES to SDF format
            ligand_file = os.path.join(temp_dir, "ligand.sdf")
            conversion_success = self._smiles_to_sdf(smiles, ligand_file)
            
            # Check if conversion was successful
            if not conversion_success:
                return {
                    "score": None,
                    "error": "Failed to convert SMILES to SDF",
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
    
    def _run_vina_batch_docking(self, ligand_files: list) -> list:
        """
        Run AutoDock Vina docking for multiple ligands efficiently.
        
        Args:
            ligand_files: List of paths to ligand PDBQT files
            
        Returns:
            List of best docking scores (same order as input files)
        """
        scores = []
        
        # Process ligands in smaller batches to manage resources
        batch_size = 10  # Adjust based on system resources
        
        for i in range(0, len(ligand_files), batch_size):
            batch_files = ligand_files[i:i + batch_size]
            batch_scores = self._run_vina_batch_chunk(batch_files)
            scores.extend(batch_scores)
        
        return scores
    
    def _run_vina_batch_chunk(self, ligand_files: list) -> list:
        """
        Run Vina docking for a chunk of ligands.
        
        Args:
            ligand_files: List of ligand PDBQT file paths (small batch)
            
        Returns:
            List of docking scores
        """
        scores = []
        
        try:
            # Create a temporary directory for batch outputs
            temp_dir = os.path.dirname(ligand_files[0])
            
            # Process each ligand in the batch
            processes = []
            output_files = []
            
            # Start all docking processes in parallel
            for ligand_file in ligand_files:
                output_file = ligand_file.replace(".pdbqt", "_out.pdbqt")
                output_files.append(output_file)
                
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
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                processes.append(process)
            
            # Wait for all processes to complete
            for i, (process, output_file) in enumerate(zip(processes, output_files)):
                try:
                    stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout per ligand
                    
                    if process.returncode == 0:
                        # Parse the score from output file
                        score = self._parse_vina_pdbqt_output(output_file)
                        scores.append(score)
                    else:
                        logger.warning(f"Vina failed for ligand {i}: {stderr}")
                        scores.append(None)
                    
                    # Clean up output file
                    if os.path.exists(output_file):
                        os.remove(output_file)
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.warning(f"Vina docking timed out for ligand {i}")
                    scores.append(None)
                    
                    # Clean up output file
                    if os.path.exists(output_file):
                        os.remove(output_file)
                
        except Exception as e:
            logger.error(f"Error in Vina batch docking: {e}")
            # Return None for all ligands in case of batch failure
            scores = [None] * len(ligand_files)
        
        return scores
    
    def _run_gnina_batch_docking(self, ligand_files: list) -> list:
        """
        Run GNINA docking for multiple ligands efficiently.
        
        GNINA doesn't support multi-ligand SDF input as effectively as expected,
        so we use parallel processing instead for better efficiency.
        
        Args:
            ligand_files: List of paths to ligand PDB files
            
        Returns:
            List of best CNNscores (same order as input files)
        """
        scores = []
        
        # Process ligands in smaller batches to manage resources
        batch_size = 5  # Smaller batches for GNINA
        
        for i in range(0, len(ligand_files), batch_size):
            batch_files = ligand_files[i:i + batch_size]
            batch_scores = self._run_gnina_batch_chunk(batch_files)
            scores.extend(batch_scores)
        
        return scores
    
    def _run_gnina_batch_chunk(self, ligand_files: list) -> list:
        """
        Run GNINA docking for a chunk of ligands in parallel.
        
        Args:
            ligand_files: List of ligand PDB file paths (small batch)
            
        Returns:
            List of docking scores
        """
        scores = []
        
        try:
            # Process each ligand in the batch
            processes = []
            output_files = []
            
            # Start all docking processes in parallel
            for ligand_file in ligand_files:
                output_file = ligand_file.replace(".pdb", "_out.sdf")
                output_files.append(output_file)
                
                cmd = [
                    "/home/aoxu/projects/PoseBench/forks/GNINA/gnina",
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
                    "--num_modes", str(self.num_poses),
                    "--seed", "42",
                    "--cnn_scoring", "rescore"
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                processes.append(process)
            
            # Wait for all processes to complete
            for i, (process, output_file) in enumerate(zip(processes, output_files)):
                try:
                    stdout, stderr = process.communicate(timeout=600)  # 10 minute timeout per ligand
                    
                    if process.returncode == 0:
                        # Parse the score from output file
                        score = self._parse_gnina_sdf_output(output_file)
                        scores.append(score)
                    else:
                        logger.warning(f"GNINA failed for ligand {i}: {stderr}")
                        scores.append(None)
                    
                    # Clean up output file
                    if os.path.exists(output_file):
                        os.remove(output_file)
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.warning(f"GNINA docking timed out for ligand {i}")
                    scores.append(None)
                    
                    # Clean up output file
                    if os.path.exists(output_file):
                        os.remove(output_file)
                
        except Exception as e:
            logger.error(f"Error in GNINA batch docking: {e}")
            # Return None for all ligands in case of batch failure  
            scores = [None] * len(ligand_files)
        
        return scores
    
    def _create_combined_sdf(self, pdb_files: list, output_sdf: str) -> bool:
        """
        Convert multiple PDB files to a single SDF file for batch processing.
        
        Args:
            pdb_files: List of PDB file paths
            output_sdf: Output SDF file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.debug(f"Creating combined SDF from {len(pdb_files)} PDB files")
            
            with open(output_sdf, 'w') as sdf_out:
                for i, pdb_file in enumerate(pdb_files):
                    logger.debug(f"Converting PDB file {i+1}/{len(pdb_files)}: {pdb_file}")
                    
                    # Check if PDB file exists and is not empty
                    if not os.path.exists(pdb_file):
                        logger.error(f"PDB file does not exist: {pdb_file}")
                        continue
                    
                    if os.path.getsize(pdb_file) == 0:
                        logger.error(f"PDB file is empty: {pdb_file}")
                        continue
                    
                    # Convert each PDB to SDF format using obabel
                    cmd = [
                        "obabel",
                        "-ipdb", pdb_file,
                        "-osdf",
                        "-O", "-"  # Output to stdout
                    ]
                    
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=30  # 30 second timeout per conversion
                        )
                        
                        if result.stdout:
                            # Add molecule index as a property for tracking
                            sdf_content = result.stdout
                            # Insert molecule index before the $$$$
                            sdf_content = sdf_content.replace(
                                "$$$$", 
                                f">  <MOLECULE_INDEX>\n{i}\n\n$$$$"
                            )
                            sdf_out.write(sdf_content)
                            logger.debug(f"Successfully converted PDB {i+1}")
                        else:
                            logger.warning(f"No output from obabel for {pdb_file}")
                    
                    except subprocess.CalledProcessError as e:
                        logger.error(f"obabel failed for {pdb_file}: {e.stderr}")
                        continue
                    except subprocess.TimeoutExpired:
                        logger.error(f"obabel timed out for {pdb_file}")
                        continue
                    except Exception as e:
                        logger.error(f"Error converting {pdb_file}: {e}")
                        continue
            
            # Check if output file was created and has content
            if os.path.exists(output_sdf) and os.path.getsize(output_sdf) > 0:
                logger.debug(f"Combined SDF created successfully: {output_sdf}")
                return True
            else:
                logger.error(f"Combined SDF file was not created or is empty: {output_sdf}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating combined SDF: {e}")
            return False
    
    def _parse_gnina_batch_output(self, sdf_file: str, num_ligands: int) -> list:
        """
        Parse GNINA batch output SDF file to extract scores for each ligand.
        
        Args:
            sdf_file: Path to GNINA output SDF file
            num_ligands: Expected number of ligands
            
        Returns:
            List of best CNNscores for each ligand
        """
        scores = [None] * num_ligands
        
        try:
            # Group molecules by their original index
            molecule_scores = {}
            
            supplier = Chem.SDMolSupplier(sdf_file)
            for mol in supplier:
                if mol is None:
                    continue
                
                # Get molecule index
                mol_idx = 0  # Default
                if mol.HasProp('MOLECULE_INDEX'):
                    try:
                        mol_idx = int(mol.GetProp('MOLECULE_INDEX'))
                    except (ValueError, TypeError):
                        pass
                
                # Get CNNscore
                cnn_score = None
                if mol.HasProp('CNNscore'):
                    try:
                        cnn_score = float(mol.GetProp('CNNscore'))
                    except (ValueError, TypeError):
                        continue
                elif mol.HasProp('cnn_score'):
                    try:
                        cnn_score = float(mol.GetProp('cnn_score'))
                    except (ValueError, TypeError):
                        continue
                
                if cnn_score is not None:
                    if mol_idx not in molecule_scores or cnn_score > molecule_scores[mol_idx]:
                        molecule_scores[mol_idx] = cnn_score
            
            # Extract scores in order
            for i in range(num_ligands):
                if i in molecule_scores:
                    scores[i] = molecule_scores[i]
                    
        except Exception as e:
            logger.error(f"Error parsing GNINA batch output: {e}")
        
        return scores
