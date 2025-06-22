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

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    import numpy as np
    RDKIT_AVAILABLE = True
except ImportError:
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
                cmd, capture_output=True, text=True, timeout=300
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
            supplier = Chem.SDMolSupplier(sdf_file)
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
            "--cnn_scoring"
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
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
            supplier = Chem.SDMolSupplier(sdf_file)
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
    def _rdkit_conversion(smiles: str, output_file: str) -> bool:
        """Use RDKit for conversion."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            if AllChem.EmbedMolecule(mol) != 0:
                AllChem.EmbedMolecule(mol, useRandomCoords=True)
            
            AllChem.MMFFOptimizeMolecule(mol)
            
            writer = Chem.SDWriter(output_file)
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
        
        # Receptor file
        self.receptor_file = (
            receptor_file or 
            docking_config.get("receptor_file") or 
            f"data/targets/{self.target}/{self.target}_prepared.pdbqt"
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
            if not self.engine.check_installation():
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
        """Evaluate a single molecule."""
        if self.mock_mode:
            return self._mock_evaluation(smiles)
        
        # Convert SMILES to SDF
        with temporary_file(".sdf") as ligand_file:
            if not MolecularConverter.smiles_to_sdf(smiles, ligand_file):
                return {
                    "smiles": smiles,
                    "score": None,
                    "error": "Conversion failed"
                }
            
            # Perform docking
            score = self.engine.dock(
                ligand_file, self.receptor_file, **self.docking_params
            )
            
            return {
                "smiles": smiles,
                "score": score,
                "error": None if score is not None else "Docking failed"
            }
    
    def _evaluate_batch(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Evaluate a batch of molecules with parallel processing."""
        if self.mock_mode:
            return [self._mock_evaluation(smiles) for smiles in smiles_list]
        
        results = []
        max_workers = min(4, len(smiles_list))  # Limit parallel processes
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_smiles = {
                executor.submit(self._evaluate_single, smiles): smiles 
                for smiles in smiles_list
            }
            
            for future in concurrent.futures.as_completed(future_to_smiles):
                try:
                    result = future.result(timeout=600)  # 10 minute timeout
                    results.append(result)
                except Exception as e:
                    smiles = future_to_smiles[future]
                    results.append({
                        "smiles": smiles,
                        "score": None,
                        "error": f"Evaluation failed: {e}"
                    })
        
        return results
    
    def _mock_evaluation(self, smiles: str) -> Dict[str, Any]:
        """Generate mock docking scores for testing."""
        try:
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return {"smiles": smiles, "score": None, "error": "Invalid SMILES"}
                
                # Simple score based on molecular properties
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                
                # Mock score: prefer drug-like molecules
                mock_score = -8.0  # Base score
                mock_score += max(0, (500 - mw) / 100)  # Prefer MW < 500
                mock_score += max(0, (5 - abs(logp)) / 2)  # Prefer LogP near 0-3
                
                # Add some noise
                mock_score += np.random.normal(0, 0.5)
                
            else:
                # Fallback without RDKit
                mock_score = -8.0 + hash(smiles) % 100 / 50.0  # Reproducible but varied
            
            return {
                "smiles": smiles,
                "score": round(mock_score, 2),
                "error": None
            }
            
        except Exception as e:
            return {
                "smiles": smiles,
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
