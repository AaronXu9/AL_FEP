"""
Optimized docking oracle supporting AutoDock Vina and GNINA
"""

import os
import tempfile
import subprocess
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import concurrent.futures
from contextlib import contextmanager
import time

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    import numpy as np
    RDKIT_AVAILABLE = True
except ImportError:
    # Create placeholder for type hints
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    Descriptors = None  # type: ignore
    np = None  # type: ignore
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Using fallback methods.")

from .base_oracle import BaseOracle

logger = logging.getLogger(__name__)


class DockingEngine:
    """Base class for docking engines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def check_installation(self) -> bool:
        """Check if the docking engine is available."""
        raise NotImplementedError
        
    def dock(self, ligand_file: str, receptor_file: str, **kwargs) -> Optional[float]:
        """Perform docking and return best score."""
        raise NotImplementedError


class VinaEngine(DockingEngine):
    """AutoDock Vina docking engine."""
    
    def check_installation(self) -> bool:
        try:
            result = subprocess.run(
                ["vina", "--help"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def dock(self, ligand_file: str, receptor_file: str, **kwargs) -> Optional[float]:
        """Run Vina docking."""
        output_file = ligand_file.replace(".sdf", "_out.sdf")
        
        cmd = [
            "vina",
            "--receptor", receptor_file,
            "--ligand", ligand_file,
            "--out", output_file,
            "--center_x", str(kwargs.get("center_x", 0.0)),
            "--center_y", str(kwargs.get("center_y", 0.0)),
            "--center_z", str(kwargs.get("center_z", 0.0)),
            "--size_x", str(kwargs.get("size_x", 20.0)),
            "--size_y", str(kwargs.get("size_y", 20.0)),
            "--size_z", str(kwargs.get("size_z", 20.0)),
            "--exhaustiveness", str(kwargs.get("exhaustiveness", 8)),
            "--num_modes", str(kwargs.get("num_poses", 9))
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=False, text=True, timeout=300
            )
            
            if result.returncode != 0:
                logger.warning(f"Vina failed: {result.stderr}")
                return None
            
            score = self._parse_output(output_file)
            
            # Cleanup
            if os.path.exists(output_file):
                os.remove(output_file)
                
            return score
            
        except subprocess.TimeoutExpired:
            logger.warning("Vina docking timed out")
            return None
        except Exception as e:
            logger.error(f"Vina docking error: {e}")
            return None
    
    def _parse_output(self, sdf_file: str) -> Optional[float]:
        """Parse Vina SDF output for best score."""
        if not RDKIT_AVAILABLE or not os.path.exists(sdf_file):
            return None
            
        try:
            supplier = Chem.SDMolSupplier(sdf_file)  # type: ignore
            for mol in supplier:
                if mol is None:
                    continue
                    
                # Check for various score properties
                for prop in ['REMARK', 'vina_score', 'score']:
                    if mol.HasProp(prop):
                        try:
                            score_text = mol.GetProp(prop)
                            # Extract numeric score from text
                            import re
                            match = re.search(r'-?\d+\.?\d*', score_text)
                            if match:
                                return float(match.group())
                        except (ValueError, TypeError):
                            continue
            return None
        except Exception as e:
            logger.error(f"Error parsing Vina output: {e}")
            return None


class GninaEngine(DockingEngine):
    """GNINA docking engine."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.gnina_path = config.get("gnina_path", "gnina")
    
    def check_installation(self) -> bool:
        try:
            result = subprocess.run(
                [self.gnina_path, "--help"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def dock(self, ligand_file: str, receptor_file: str, **kwargs) -> Optional[float]:
        """Run GNINA docking."""
        output_file = ligand_file.replace(".sdf", "_gnina_out.sdf")
        
        cmd = [
            self.gnina_path,
            "--receptor", receptor_file,
            "--ligand", ligand_file,
            "--out", output_file,
            "--center_x", str(kwargs.get("center_x", 0.0)),
            "--center_y", str(kwargs.get("center_y", 0.0)),
            "--center_z", str(kwargs.get("center_z", 0.0)),
            "--size_x", str(kwargs.get("size_x", 20.0)),
            "--size_y", str(kwargs.get("size_y", 20.0)),
            "--size_z", str(kwargs.get("size_z", 20.0)),
            "--exhaustiveness", str(kwargs.get("exhaustiveness", 8)),
            "--num_modes", str(kwargs.get("num_poses", 9)),
            "--cnn_scoring", "rescore"      
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=False, text=True, timeout=600
            )
            
            if result.returncode != 0:
                logger.warning(f"GNINA failed: {result.stderr}")
                return None
            
            score = self._parse_output(output_file)
            
            # Cleanup
            if os.path.exists(output_file):
                os.remove(output_file)
                
            return score
            
        except subprocess.TimeoutExpired:
            logger.warning("GNINA docking timed out")
            return None
        except Exception as e:
            logger.error(f"GNINA docking error: {e}")
            return None
    
    def _parse_output(self, sdf_file: str) -> Optional[float]:
        """Parse GNINA SDF output for best CNNscore."""
        if not RDKIT_AVAILABLE or not os.path.exists(sdf_file):
            return None
            
        try:
            supplier = Chem.SDMolSupplier(sdf_file)  # type: ignore
            best_score = None
            
            for mol in supplier:
                if mol is None:
                    continue
                    
                # Look for CNNscore or similar properties
                for prop in ['CNNscore', 'cnn_score', 'score']:
                    if mol.HasProp(prop):
                        try:
                            score = float(mol.GetProp(prop))
                            if best_score is None or score > best_score:
                                best_score = score
                            break
                        except (ValueError, TypeError):
                            continue
                            
            return best_score
        except Exception as e:
            logger.error(f"Error parsing GNINA output: {e}")
            return None


class MolecularConverter:
    """Handle molecular format conversions."""
    
    @staticmethod
    def smiles_to_sdf(smiles: str, output_file: str) -> bool:
        """Convert SMILES to SDF format."""
        if RDKIT_AVAILABLE:
            return MolecularConverter._rdkit_conversion(smiles, output_file)
        else:
            return MolecularConverter._obabel_conversion(smiles, output_file)
    
    @staticmethod
    def is_sdf_file(input_str: str) -> bool:
        """Check if input string is a path to an SDF file."""
        if not isinstance(input_str, str):
            return False
        
        # Check if it looks like a file path and exists
        if os.path.exists(input_str) and input_str.lower().endswith('.sdf'):
            return True
        
        # Check if it's a multiline SDF content string
        lines = input_str.strip().split('\n')
        if len(lines) > 10:  # SDF files are typically multi-line
            # Look for SDF format indicators
            for line in lines:
                if line.strip() in ['$$$$', 'M  END'] or line.startswith('  '):
                    return True
        
        return False
    
    @staticmethod
    def validate_sdf_file(sdf_path: str) -> bool:
        """Validate that an SDF file is readable and contains molecules."""
        if not RDKIT_AVAILABLE:
            # Basic file existence check without RDKit
            return os.path.exists(sdf_path) and sdf_path.lower().endswith('.sdf')
        
        try:
            supplier = Chem.SDMolSupplier(sdf_path)  # type: ignore
            # Check if we can read at least one molecule
            for mol in supplier:
                if mol is not None:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error validating SDF file {sdf_path}: {e}")
            return False
    
    @staticmethod
    def extract_smiles_from_sdf(sdf_path: str) -> List[str]:
        """Extract SMILES strings from an SDF file for identification."""
        smiles_list = []
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available. Cannot extract SMILES from SDF.")
            return smiles_list
        
        try:
            supplier = Chem.SDMolSupplier(sdf_path)  # type: ignore
            for mol in supplier:
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)  # type: ignore
                    smiles_list.append(smiles)
        except Exception as e:
            logger.error(f"Error extracting SMILES from SDF {sdf_path}: {e}")
        
        return smiles_list
    
    @staticmethod
    def _rdkit_conversion(smiles: str, output_file: str) -> bool:
        """Use RDKit for conversion."""
        try:
            mol = Chem.MolFromSmiles(smiles)  # type: ignore
            if mol is None:
                return False
            
            mol = Chem.AddHs(mol)  # type: ignore
            
            # Generate 3D coordinates
            if AllChem.EmbedMolecule(mol) != 0:  # type: ignore
                AllChem.EmbedMolecule(mol, useRandomCoords=True)  # type: ignore
            
            AllChem.MMFFOptimizeMolecule(mol)  # type: ignore
            
            writer = Chem.SDWriter(output_file)  # type: ignore
            writer.write(mol)
            writer.close()
            
            return os.path.exists(output_file) and os.path.getsize(output_file) > 0
            
        except Exception as e:
            logger.error(f"RDKit conversion failed: {e}")
            return False
    
    @staticmethod
    def _obabel_conversion(smiles: str, output_file: str) -> bool:
        """Use OpenBabel for conversion."""
        try:
            cmd = ["obabel", "-ismi", "-osdf", "--gen3d", "-O", output_file]
            result = subprocess.run(
                cmd, input=smiles, text=True, capture_output=True, check=True
            )
            return os.path.exists(output_file) and os.path.getsize(output_file) > 0
        except Exception as e:
            logger.error(f"OpenBabel conversion failed: {e}")
            return False


@contextmanager
def temporary_file(suffix=".sdf"):
    """Context manager for temporary files."""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            temp_file = tf.name
        yield temp_file
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


class DockingOracle(BaseOracle):
    """
    Optimized oracle for molecular docking using AutoDock Vina or GNINA.
    """
    
    def __init__(
        self,
        target: str,
        receptor_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize the docking oracle."""
        super().__init__(name="Docking", target=target, config=config, **kwargs)
        
        # Configuration
        docking_config = self.config.get("docking", {})
        self.engine_name = docking_config.get("engine", "vina").lower()
        self.mock_mode = docking_config.get("mock_mode", False)
        self.receptor_file = (
            receptor_file or
            docking_config.get("receptor_file", "data/targets/test/test_oracle.pdb")
        )

        # Docking parameters
        self.docking_params = {
            "center_x": docking_config.get("center_x", 0.0),
            "center_y": docking_config.get("center_y", 0.0),
            "center_z": docking_config.get("center_z", 0.0),
            "size_x": docking_config.get("size_x", 20.0),
            "size_y": docking_config.get("size_y", 20.0),
            "size_z": docking_config.get("size_z", 20.0),
            "exhaustiveness": docking_config.get("exhaustiveness", 8),
            "num_poses": docking_config.get("num_poses", 9)
        }
        
        # Initialize docking engine
        if not self.mock_mode:
            self.engine = self._create_engine()
            if self.engine and not self.engine.check_installation():
                raise RuntimeError(f"{self.engine_name} not available")
        else:
            self.engine = None
            
        logger.info(f"DockingOracle ({self.engine_name.upper()}) initialized")
    
    def _create_engine(self) -> DockingEngine:
        """Create the appropriate docking engine."""
        if self.engine_name == "vina":
            return VinaEngine(self.config.get("docking", {}))
        elif self.engine_name == "gnina":
            return GninaEngine(self.config.get("docking", {}))
        else:
            raise ValueError(f"Unsupported docking engine: {self.engine_name}")
    
    def _evaluate_single(self, smiles: str) -> Dict[str, Any]:
        """Evaluate a single molecule (SMILES string or SDF file path)."""
        if self.mock_mode:
            return self._mock_evaluation(smiles)
        
        # Determine input type
        if MolecularConverter.is_sdf_file(smiles):
            return self._evaluate_sdf_file(smiles)
        else:
            return self._evaluate_smiles(smiles)
    
    def _evaluate_smiles(self, smiles: str) -> Dict[str, Any]:
        """Evaluate a single SMILES string."""
        # Convert SMILES to SDF
        with temporary_file(".sdf") as ligand_file:
            if not MolecularConverter.smiles_to_sdf(smiles, ligand_file):
                return {
                    "input": smiles,
                    "smiles": smiles,
                    "score": None,
                    "error": "SMILES conversion failed"
                }
            
            # Perform docking
            if self.engine:
                score = self.engine.dock(
                    ligand_file, self.receptor_file, **self.docking_params
                )
            else:
                score = None
            
            return {
                "input": smiles,
                "smiles": smiles,
                "score": score,
                "error": None if score is not None else "Docking failed"
            }
    
    def _evaluate_sdf_file(self, sdf_path: str) -> Dict[str, Any]:
        """Evaluate an SDF file directly."""
        # Validate the SDF file
        if not MolecularConverter.validate_sdf_file(sdf_path):
            return {
                "input": sdf_path,
                "smiles": None,
                "score": None,
                "error": "Invalid SDF file"
            }
        
        # Extract SMILES for identification (optional)
        smiles_list = MolecularConverter.extract_smiles_from_sdf(sdf_path)
        representative_smiles = smiles_list[0] if smiles_list else None
        
        # Perform docking directly with the SDF file
        if self.engine:
            score = self.engine.dock(
                sdf_path, self.receptor_file, **self.docking_params
            )
        else:
            score = None
        
        return {
            "input": sdf_path,
            "smiles": representative_smiles,
            "score": score,
            "error": None if score is not None else "Docking failed"
        }
    
    def _evaluate_batch(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Evaluate a batch of molecules (SMILES strings or SDF file paths) with parallel processing."""
        if self.mock_mode:
            return [self._mock_evaluation(smiles) for smiles in smiles_list]
        
        results = []
        max_workers = min(4, len(smiles_list))  # Limit parallel processes
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_input = {
                executor.submit(self._evaluate_single, smiles): smiles 
                for smiles in smiles_list
            }
            
            for future in concurrent.futures.as_completed(future_to_input):
                try:
                    result = future.result(timeout=600)  # 10 minute timeout
                    results.append(result)
                except Exception as e:
                    smiles = future_to_input[future]
                    results.append({
                        "input": smiles,
                        "smiles": smiles if not MolecularConverter.is_sdf_file(smiles) else None,
                        "score": None,
                        "error": f"Evaluation failed: {e}"
                    })
        
        return results
    
    def _mock_evaluation(self, smiles: str) -> Dict[str, Any]:
        """Generate mock docking scores for testing (handles both SMILES and SDF)."""
        try:
            # Handle SDF files
            if MolecularConverter.is_sdf_file(smiles):
                smiles_list = MolecularConverter.extract_smiles_from_sdf(smiles)
                if smiles_list:
                    # Use the first molecule for mock evaluation
                    actual_smiles = smiles_list[0]
                    input_data = smiles  # Keep original SDF path
                else:
                    return {
                        "input": smiles,
                        "smiles": None,
                        "score": None,
                        "error": "Cannot extract SMILES from SDF"
                    }
            else:
                actual_smiles = smiles
                input_data = smiles
            
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(actual_smiles)  # type: ignore
                if mol is None:
                    return {
                        "input": input_data,
                        "smiles": actual_smiles,
                        "score": None,
                        "error": "Invalid SMILES"
                    }
                
                # Simple score based on molecular properties
                mw = Descriptors.MolWt(mol)  # type: ignore
                logp = Descriptors.MolLogP(mol)  # type: ignore
                
                # Mock score: prefer drug-like molecules
                mock_score = -8.0  # Base score
                mock_score += max(0, (500 - mw) / 100)  # Prefer MW < 500
                mock_score += max(0, (5 - abs(logp)) / 2)  # Prefer LogP near 0-3
                
                # Add some noise
                mock_score += np.random.normal(0, 0.5)  # type: ignore
                
            else:
                # Fallback without RDKit
                mock_score = -8.0 + hash(actual_smiles) % 100 / 50.0  # Reproducible but varied
            
            return {
                "input": input_data,
                "smiles": actual_smiles,
                "score": round(mock_score, 2),
                "error": None
            }
            
        except Exception as e:
            return {
                "input": smiles,
                "smiles": smiles if not MolecularConverter.is_sdf_file(smiles) else None,
                "score": None,
                "error": f"Mock evaluation failed: {e}"
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get oracle statistics."""
        stats = super().get_statistics()
        stats.update({
            "engine": self.engine_name,
            "receptor_file": self.receptor_file,
            "mock_mode": self.mock_mode,
            "docking_params": self.docking_params
        })
        return stats
    
    def dock_sdf_file(self, sdf_path: str) -> Dict[str, Any]:
        """
        Convenience method to dock an SDF file directly.
        
        Args:
            sdf_path: Path to the SDF file
            
        Returns:
            Docking result dictionary
        """
        result = self.evaluate(sdf_path)
        return result if isinstance(result, dict) else result[0]

    def _evaluate_with_cache(self, smiles: str) -> Dict[str, Any]:
        """
        Evaluate a molecule with caching support.
        Override base class to handle SDF files.
        """
        # Check cache first
        if self.cache and smiles in self._cache:
            logger.debug(f"Cache hit for {smiles}")
            return self._cache[smiles]

        # Handle SDF files differently from SMILES
        if MolecularConverter.is_sdf_file(smiles):
            # For SDF files, skip SMILES validation and use file path as key
            start_time = time.time()
            try:
                result = self._evaluate_single(smiles)
                result["input"] = smiles
                result["oracle"] = self.name
                result["success"] = True
                
                # Update statistics
                self.call_count += 1
                evaluation_time = time.time() - start_time
                self.total_time += evaluation_time
                
                # Cache the result
                if self.cache:
                    self._cache[smiles] = result

                return result
                
            except Exception as e:
                logger.error(f"Error evaluating SDF file {smiles}: {e}")
                return {
                    "input": smiles,
                    "smiles": None,
                    "score": None,
                    "error": str(e),
                    "oracle": self.name,
                    "success": False
                }
        else:
            # Use parent method for SMILES strings
            return super()._evaluate_with_cache(smiles)
