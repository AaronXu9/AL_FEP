"""
Boltz-2 oracle for protein-ligand affinity prediction
"""

import os
import json
import yaml
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from .base_oracle import BaseOracle

logger = logging.getLogger(__name__)


class BoltzOracle(BaseOracle):
    """
    Boltz-2 oracle for protein-ligand binding affinity prediction.
    
    Uses the Boltz-2 deep learning model to predict binding affinities
    for protein-ligand complexes.
    """
    
    def __init__(
        self,
        target: str,
        config: Optional[Dict[str, Any]] = None,
        cache: bool = True
    ):
        """
        Initialize Boltz oracle.
        
        Args:
            target: Target identifier (e.g., "7jvr")
            config: Configuration dictionary
            cache: Whether to cache results
        """
        super().__init__(name="BoltzOracle", target=target, config=config, cache=cache)
        
        # Get Boltz-specific configuration
        boltz_config = self.config.get("boltz", {})
        
        # Model parameters
        self.model = boltz_config.get("model", "boltz2")
        self.diffusion_samples = boltz_config.get("diffusion_samples", 1)
        self.recycling_steps = boltz_config.get("recycling_steps", 3)
        self.sampling_steps = boltz_config.get("sampling_steps", 200)
        self.use_msa_server = boltz_config.get("use_msa_server", True)
        self.use_potentials = boltz_config.get("use_potentials", False)
        
        # Affinity-specific parameters  
        self.predict_affinity = boltz_config.get("predict_affinity", True)
        self.affinity_mw_correction = boltz_config.get("affinity_mw_correction", False)
        self.diffusion_samples_affinity = boltz_config.get("diffusion_samples_affinity", 5)
        
        # Output format
        self.output_format = boltz_config.get("output_format", "pdb")
        
        # YAML file configuration
        self.yaml_file_path = boltz_config.get("yaml_file_path", None)  # Custom YAML file path
        self.yaml_template_dir = boltz_config.get("yaml_template_dir", None)  # Template directory
        self.preserve_yaml_files = boltz_config.get("preserve_yaml_files", False)  # Keep YAML files
        
        # Protein sequence configuration - can be provided directly or from file
        self.protein_sequence_direct = boltz_config.get("protein_sequence", None)  # Direct sequence
        
        # File paths
        self.protein_sequence_file = boltz_config.get(
            "protein_sequence_file", 
            self._get_default_protein_sequence_file()
        )
        
        # Make work_dir absolute path
        work_dir_relative = boltz_config.get("work_dir", "data/boltz_workspace")
        if os.path.isabs(work_dir_relative):
            self.work_dir = work_dir_relative
        else:
            # Get project root and make absolute path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            self.work_dir = os.path.join(project_root, work_dir_relative)
        
        # Ensure work directory exists
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Check Boltz installation
        self._check_boltz_installation()
        
        # Load protein sequence
        self.protein_sequence = self._load_protein_sequence()
        
        logger.info(f"BoltzOracle initialized for target {self.target}")
        logger.info(f"Model: {self.model}, Samples: {self.diffusion_samples}")
        logger.info(f"Affinity prediction: {self.predict_affinity}")
    
    def _get_default_protein_sequence_file(self) -> str:
        """Get default protein sequence file path."""
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        return os.path.join(project_root, "data", "targets", self.target, f"{self.target}_sequence.fasta")
    
    def _check_boltz_installation(self):
        """Check if Boltz is installed and accessible."""
        try:
            result = subprocess.run(
                ["boltz", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("Boltz command failed")
            logger.info("Boltz installation verified")
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError):
            raise ImportError(
                "Boltz not found or not working. Please install Boltz-2 following "
                "the instructions at https://github.com/jwohlwend/boltz"
            )
    
    def _load_protein_sequence(self) -> str:
        """Load protein sequence from file."""
        # If protein sequence is provided directly, use it
        if self.protein_sequence_direct:
            sequence = self.protein_sequence_direct
            logger.info(f"Using direct protein sequence input: {len(sequence)} residues")
            return sequence.strip()
        
        if not os.path.exists(self.protein_sequence_file):
            # Try to create a default sequence file
            default_seq = self._get_default_sequence()
            if default_seq:
                self._create_sequence_file(default_seq)
            else:
                raise FileNotFoundError(
                    f"Protein sequence file not found: {self.protein_sequence_file}. "
                    "Please provide a FASTA file with the protein sequence or set 'protein_sequence' in config."
                )
        
        with open(self.protein_sequence_file, 'r') as f:
            content = f.read().strip()
            
        # Extract sequence from FASTA format
        lines = content.split('\n')
        sequence = ""
        for line in lines:
            if not line.startswith('>'):
                sequence += line.strip()
        
        if not sequence:
            raise ValueError(f"No sequence found in {self.protein_sequence_file}")
        
        logger.info(f"Loaded protein sequence: {len(sequence)} residues")
        return sequence
    
    def _get_default_sequence(self) -> Optional[str]:
        """Get default protein sequence for known targets."""
        # This could be expanded with known target sequences
        default_sequences = {
            "7jvr": "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAAFAEVTPIAQAYKKLTRQEQAGGKKAQGNGGAGPRAKKRRSS",
            "test": "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCP",  # Shorter test sequence
            # Add more targets as needed
        }
        return default_sequences.get(self.target)
    
    def _create_sequence_file(self, sequence: str):
        """Create a FASTA file with the protein sequence."""
        os.makedirs(os.path.dirname(self.protein_sequence_file), exist_ok=True)
        with open(self.protein_sequence_file, 'w') as f:
            f.write(f">{self.target}_protein\n{sequence}\n")
        logger.info(f"Created sequence file: {self.protein_sequence_file}")
    
    def _create_yaml_input(self, smiles: str, output_path: str) -> str:
        """
        Create YAML input file for Boltz prediction.
        
        Args:
            smiles: SMILES string of the ligand
            output_path: Path to save the YAML file
            
        Returns:
            Path to the created YAML file
        """
        yaml_content = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": self.protein_sequence,
                    }
                },
                {
                    "ligand": {
                        "id": "B", 
                        "smiles": smiles
                    }
                }
            ]
        }
        
        # Add affinity prediction if enabled
        if self.predict_affinity:
            yaml_content["properties"] = [
                {
                    "affinity": {
                        "binder": "B"  # Ligand chain
                    }
                }
            ]
        
        with open(output_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        logger.debug(f"Created YAML input: {output_path}")
        return output_path
    
    def _run_boltz_prediction(self, yaml_file: str, output_dir: str) -> bool:
        """
        Run Boltz prediction.
        
        Args:
            yaml_file: Path to YAML input file
            output_dir: Output directory for results
            
        Returns:
            True if successful, False otherwise
        """
        cmd = [
            "boltz", "predict", yaml_file,
            "--out_dir", output_dir,
            "--model", self.model,
            "--diffusion_samples", str(self.diffusion_samples),
            "--recycling_steps", str(self.recycling_steps),
            "--sampling_steps", str(self.sampling_steps),
            "--output_format", self.output_format
        ]
        
        # Add optional flags
        if self.use_msa_server:
            cmd.append("--use_msa_server")
        
        if self.use_potentials:
            cmd.append("--use_potentials")
        
        if self.predict_affinity:
            cmd.extend([
                "--diffusion_samples_affinity", str(self.diffusion_samples_affinity)
            ])
            
            if self.affinity_mw_correction:
                cmd.append("--affinity_mw_correction")
        
        try:
            logger.debug(f"Running Boltz command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Boltz prediction failed: {result.stderr}")
                return False
            
            logger.debug("Boltz prediction completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Boltz prediction timed out")
            return False
        except Exception as e:
            logger.error(f"Error running Boltz prediction: {str(e)}")
            return False
    
    def _parse_boltz_output(self, output_dir: str, input_name: str) -> Dict[str, Any]:
        """
        Parse Boltz output files.
        
        Args:
            output_dir: Directory containing Boltz output
            input_name: Name of the input file (without extension)
            
        Returns:
            Dictionary containing parsed results
        """
        results: Dict[str, Any] = {
            "binding_affinity": None,
            "binding_probability": None,
            "confidence_score": None,
            "ptm": None,
            "iptm": None,
            "plddt": None,
            "structure_file": None,
            "error": None
        }
        
        # Find prediction directory
        pred_dir = os.path.join(output_dir, f"boltz_results_{input_name}", "predictions", input_name)
        if not os.path.exists(pred_dir):
            results["error"] = f"Prediction directory not found: {pred_dir}"
            return results
        
        try:
            # Parse confidence scores
            confidence_files = [f for f in os.listdir(pred_dir) if f.startswith("confidence_")]
            if confidence_files:
                confidence_file = os.path.join(pred_dir, confidence_files[0])
                with open(confidence_file, 'r') as f:
                    confidence_data = json.load(f)
                
                results["confidence_score"] = confidence_data.get("confidence_score")
                results["ptm"] = confidence_data.get("ptm")
                results["iptm"] = confidence_data.get("iptm")
                results["plddt"] = confidence_data.get("complex_plddt")
            
            # Parse affinity scores (if available)
            if self.predict_affinity:
                affinity_files = [f for f in os.listdir(pred_dir) if f.startswith("affinity_")]
                if affinity_files:
                    affinity_file = os.path.join(pred_dir, affinity_files[0])
                    with open(affinity_file, 'r') as f:
                        affinity_data = json.load(f)
                    
                    results["binding_affinity"] = affinity_data.get("affinity_pred_value")
                    results["binding_probability"] = affinity_data.get("affinity_probability_binary")
            
            # Find structure file
            structure_files = [f for f in os.listdir(pred_dir) 
                             if f.endswith(f".{self.output_format}")]
            if structure_files:
                # Use the first structure file (highest confidence)
                results["structure_file"] = os.path.join(pred_dir, structure_files[0])
            
        except Exception as e:
            results["error"] = f"Error parsing Boltz output: {str(e)}"
        
        return results
    
    def _evaluate_single(self, smiles: str) -> Dict[str, Any]:
        """
        Evaluate a single molecule using Boltz.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary containing evaluation results
        """
        # Determine YAML file path and working directory
        yaml_file, work_dir, cleanup_temp = self._get_yaml_file_path(smiles)
        
        try:
            # Create input YAML file
            self._create_yaml_input(smiles, yaml_file)
            
            # Run Boltz prediction
            success = self._run_boltz_prediction(yaml_file, work_dir)
            
            if not success:
                return {
                    "score": None,
                    "binding_affinity": None,
                    "binding_probability": None,
                    "error": "Boltz prediction failed",
                    "method": "Boltz-2"
                }
            
            # Parse results
            input_name = os.path.splitext(os.path.basename(yaml_file))[0]
            results = self._parse_boltz_output(work_dir, input_name)
            
            if results["error"]:
                return {
                    "score": None,
                    "binding_affinity": None,
                    "binding_probability": None,
                    "error": results["error"],
                    "method": "Boltz-2"
                }
            
            # Convert to AL_FEP format
            # Use binding affinity as the main score (lower is better)
            score = results["binding_affinity"]
            if score is not None:
                # Convert to positive score (higher is better) for consistency
                score = -score
            
            return {
                "score": score,
                "binding_affinity": results["binding_affinity"],
                "binding_probability": results["binding_probability"],
                "confidence_score": results["confidence_score"],
                "ptm": results["ptm"],
                "iptm": results["iptm"],
                "plddt": results["plddt"],
                "structure_file": results["structure_file"],
                "method": "Boltz-2",
                "model": self.model,
                "samples": self.diffusion_samples
            }
            
        finally:
            # Clean up temporary files if requested
            if cleanup_temp and os.path.exists(work_dir):
                try:
                    shutil.rmtree(work_dir)
                    logger.debug(f"Cleaned up temporary directory: {work_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory {work_dir}: {e}")
    
    def _get_yaml_file_path(self, smiles: str) -> tuple[str, str, bool]:
        """
        Determine YAML file path and working directory based on configuration.
        
        Args:
            smiles: SMILES string for the molecule
            
        Returns:
            Tuple of (yaml_file_path, work_dir, cleanup_temp)
        """
        if self.yaml_file_path:
            # Use user-specified YAML file path
            yaml_file = self.yaml_file_path
            work_dir = os.path.dirname(yaml_file)
            # Ensure directory exists
            os.makedirs(work_dir, exist_ok=True)
            cleanup_temp = False
            
        elif self.yaml_template_dir:
            # Use template directory with generated filename
            input_name = f"input_{hash(smiles) % 10000}"
            yaml_file = os.path.join(self.yaml_template_dir, f"{input_name}.yaml")
            # Ensure directory exists
            os.makedirs(self.yaml_template_dir, exist_ok=True)
            work_dir = self.yaml_template_dir
            cleanup_temp = not self.preserve_yaml_files
            
        else:
            # Create temporary directory for this prediction (default behavior)
            work_dir = tempfile.mkdtemp(dir=self.work_dir, prefix="boltz_")
            input_name = f"input_{hash(smiles) % 10000}"
            yaml_file = os.path.join(work_dir, f"{input_name}.yaml")
            cleanup_temp = not self.preserve_yaml_files
        
        return yaml_file, work_dir, cleanup_temp
    
    def __str__(self) -> str:
        return f"BoltzOracle(target={self.target}, model={self.model}, calls={self.call_count})"
    
    def __repr__(self) -> str:
        return self.__str__()
