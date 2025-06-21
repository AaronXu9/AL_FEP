"""
Molecular featurization utilities for converting molecules to numerical representations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdFingerprintGenerator
import logging

logger = logging.getLogger(__name__)


class MolecularFeaturizer:
    """
    Comprehensive molecular featurizer supporting multiple fingerprint types
    and molecular descriptors.
    """
    
    def __init__(self, 
                 fingerprint_type: str = "morgan",
                 fingerprint_radius: int = 2,
                 fingerprint_bits: int = 2048,
                 include_descriptors: bool = True,
                 include_fragments: bool = False):
        """
        Initialize the molecular featurizer.
        
        Args:
            fingerprint_type: Type of fingerprint ('morgan', 'rdkit', 'maccs', 'topological')
            fingerprint_radius: Radius for Morgan fingerprints
            fingerprint_bits: Number of bits in the fingerprint
            include_descriptors: Whether to include RDKit descriptors
            include_fragments: Whether to include fragment counts
        """
        self.fingerprint_type = fingerprint_type.lower()
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        self.include_descriptors = include_descriptors
        self.include_fragments = include_fragments
        
        # Initialize fingerprint generator
        self._init_fingerprint_generator()
        
        # Descriptor and fragment lists
        self.descriptor_names = None
        self.fragment_names = None
        self.feature_names = None
        
    def _init_fingerprint_generator(self):
        """Initialize the appropriate fingerprint generator."""
        if self.fingerprint_type == "morgan":
            self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.fingerprint_radius,
                fpSize=self.fingerprint_bits
            )
        elif self.fingerprint_type == "rdkit":
            self.fp_generator = rdFingerprintGenerator.GetRDKitFPGenerator(
                fpSize=self.fingerprint_bits
            )
        elif self.fingerprint_type == "topological":
            self.fp_generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
                fpSize=self.fingerprint_bits
            )
        else:
            # Default to Morgan
            self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.fingerprint_radius,
                fpSize=self.fingerprint_bits
            )
    
    def _get_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Get fingerprint for a molecule."""
        if self.fingerprint_type == "maccs":
            # MACCS keys are 167 bits
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            return np.array(fp)
        else:
            fp = self.fp_generator.GetFingerprint(mol)
            return np.array(fp)
    
    def _get_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate molecular descriptors."""
        descriptors = {}
        
        # Basic descriptors
        descriptors['MolWt'] = Descriptors.MolWt(mol)
        descriptors['LogP'] = Descriptors.MolLogP(mol)
        descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
        descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
        descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
        descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
        descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
        descriptors['TPSA'] = Descriptors.TPSA(mol)
        descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
        descriptors['NumChiralCenters'] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        
        # Additional descriptors
        descriptors['BertzCT'] = Descriptors.BertzCT(mol)
        descriptors['Ipc'] = Descriptors.Ipc(mol)
        descriptors['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
        descriptors['Kappa1'] = Descriptors.Kappa1(mol)
        descriptors['Kappa2'] = Descriptors.Kappa2(mol)
        descriptors['Kappa3'] = Descriptors.Kappa3(mol)
        descriptors['FractionCsp3'] = Descriptors.FractionCSP3(mol)
        descriptors['NumSaturatedCarbocycles'] = Descriptors.NumSaturatedCarbocycles(mol)
        descriptors['NumAromaticCarbocycles'] = Descriptors.NumAromaticCarbocycles(mol)
        descriptors['NumSaturatedHeterocycles'] = Descriptors.NumSaturatedHeterocycles(mol)
        descriptors['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
        
        return descriptors
    
    def _get_fragments(self, mol: Chem.Mol) -> Dict[str, int]:
        """Calculate fragment counts."""
        fragments = {}
        
        # Common fragment patterns
        fragments['fr_Al_COO'] = Fragments.fr_Al_COO(mol)
        fragments['fr_Al_OH'] = Fragments.fr_Al_OH(mol)
        fragments['fr_Ar_N'] = Fragments.fr_Ar_N(mol)
        fragments['fr_Ar_NH'] = Fragments.fr_Ar_NH(mol)
        fragments['fr_Ar_OH'] = Fragments.fr_Ar_OH(mol)
        fragments['fr_COO'] = Fragments.fr_COO(mol)
        fragments['fr_COO2'] = Fragments.fr_COO2(mol)
        fragments['fr_C_O'] = Fragments.fr_C_O(mol)
        fragments['fr_C_S'] = Fragments.fr_C_S(mol)
        fragments['fr_HOCCN'] = Fragments.fr_HOCCN(mol)
        fragments['fr_NH0'] = Fragments.fr_NH0(mol)
        fragments['fr_NH1'] = Fragments.fr_NH1(mol)
        fragments['fr_NH2'] = Fragments.fr_NH2(mol)
        fragments['fr_N_O'] = Fragments.fr_N_O(mol)
        fragments['fr_Ndealkylation1'] = Fragments.fr_Ndealkylation1(mol)
        fragments['fr_Ndealkylation2'] = Fragments.fr_Ndealkylation2(mol)
        fragments['fr_SH'] = Fragments.fr_SH(mol)
        fragments['fr_aldehyde'] = Fragments.fr_aldehyde(mol)
        fragments['fr_alkyl_halide'] = Fragments.fr_alkyl_halide(mol)
        fragments['fr_amide'] = Fragments.fr_amide(mol)
        fragments['fr_amidine'] = Fragments.fr_amidine(mol)
        fragments['fr_aniline'] = Fragments.fr_aniline(mol)
        fragments['fr_aryl_methyl'] = Fragments.fr_aryl_methyl(mol)
        fragments['fr_benzene'] = Fragments.fr_benzene(mol)
        fragments['fr_ester'] = Fragments.fr_ester(mol)
        fragments['fr_ether'] = Fragments.fr_ether(mol)
        fragments['fr_guanido'] = Fragments.fr_guanido(mol)
        fragments['fr_halogen'] = Fragments.fr_halogen(mol)
        fragments['fr_hdrzine'] = Fragments.fr_hdrzine(mol)
        fragments['fr_imidazole'] = Fragments.fr_imidazole(mol)
        fragments['fr_imide'] = Fragments.fr_imide(mol)
        fragments['fr_ketone'] = Fragments.fr_ketone(mol)
        fragments['fr_methoxy'] = Fragments.fr_methoxy(mol)
        fragments['fr_morpholine'] = Fragments.fr_morpholine(mol)
        fragments['fr_nitrile'] = Fragments.fr_nitrile(mol)
        fragments['fr_nitro'] = Fragments.fr_nitro(mol)
        fragments['fr_phenol'] = Fragments.fr_phenol(mol)
        fragments['fr_phos_acid'] = Fragments.fr_phos_acid(mol)
        fragments['fr_phos_ester'] = Fragments.fr_phos_ester(mol)
        fragments['fr_piperdine'] = Fragments.fr_piperdine(mol)
        fragments['fr_piperzine'] = Fragments.fr_piperzine(mol)
        fragments['fr_priamide'] = Fragments.fr_priamide(mol)
        fragments['fr_pyridine'] = Fragments.fr_pyridine(mol)
        fragments['fr_quatN'] = Fragments.fr_quatN(mol)
        fragments['fr_sulfide'] = Fragments.fr_sulfide(mol)
        fragments['fr_sulfonamd'] = Fragments.fr_sulfonamd(mol)
        fragments['fr_sulfone'] = Fragments.fr_sulfone(mol)
        fragments['fr_term_acetylene'] = Fragments.fr_term_acetylene(mol)
        fragments['fr_tetrazole'] = Fragments.fr_tetrazole(mol)
        fragments['fr_thiazole'] = Fragments.fr_thiazole(mol)
        fragments['fr_thiophene'] = Fragments.fr_thiophene(mol)
        fragments['fr_urea'] = Fragments.fr_urea(mol)
        
        return fragments
    
    def featurize_molecule(self, smiles: str) -> Optional[np.ndarray]:
        """
        Featurize a single molecule from SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Feature vector or None if featurization fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Could not parse SMILES: {smiles}")
                return None
            
            features = []
            
            # Get fingerprint
            fp = self._get_fingerprint(mol)
            features.extend(fp)
            
            # Get descriptors
            if self.include_descriptors:
                descriptors = self._get_descriptors(mol)
                features.extend(descriptors.values())
                
                if self.descriptor_names is None:
                    self.descriptor_names = list(descriptors.keys())
            
            # Get fragments
            if self.include_fragments:
                fragments = self._get_fragments(mol)
                features.extend(fragments.values())
                
                if self.fragment_names is None:
                    self.fragment_names = list(fragments.keys())
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error featurizing molecule {smiles}: {e}")
            return None
    
    def featurize_molecules(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Featurize a list of molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Feature matrix and list of valid SMILES
        """
        features = []
        valid_smiles = []
        
        for smiles in smiles_list:
            feat = self.featurize_molecule(smiles)
            if feat is not None:
                features.append(feat)
                valid_smiles.append(smiles)
        
        if not features:
            return np.array([]), []
        
        feature_matrix = np.vstack(features)
        return feature_matrix, valid_smiles
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        if self.feature_names is not None:
            return self.feature_names
        
        names = []
        
        # Fingerprint names
        if self.fingerprint_type == "maccs":
            names.extend([f"MACCS_{i}" for i in range(167)])
        else:
            names.extend([f"{self.fingerprint_type.upper()}_{i}" for i in range(self.fingerprint_bits)])
        
        # Descriptor names
        if self.include_descriptors and self.descriptor_names is not None:
            names.extend(self.descriptor_names)
        
        # Fragment names
        if self.include_fragments and self.fragment_names is not None:
            names.extend(self.fragment_names)
        
        self.feature_names = names
        return names
    
    def get_feature_dim(self) -> int:
        """Get the dimensionality of the feature vector."""
        dim = self.fingerprint_bits if self.fingerprint_type != "maccs" else 167
        
        if self.include_descriptors:
            dim += 22  # Number of descriptors
        
        if self.include_fragments:
            dim += 45  # Number of fragment features
        
        return dim


class DescriptorCalculator:
    """Calculate specific molecular descriptors."""
    
    @staticmethod
    def lipinski_descriptors(mol: Chem.Mol) -> Dict[str, float]:
        """Calculate Lipinski's Rule of Five descriptors."""
        return {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol)
        }
    
    @staticmethod
    def drug_like_descriptors(mol: Chem.Mol) -> Dict[str, float]:
        """Calculate drug-like descriptors."""
        descriptors = DescriptorCalculator.lipinski_descriptors(mol)
        descriptors.update({
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
            'Fsp3': Descriptors.FractionCSP3(mol),
            'NumChiralCenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        })
        return descriptors
    
    @staticmethod
    def check_lipinski(mol: Chem.Mol) -> Dict[str, bool]:
        """Check Lipinski's Rule of Five violations."""
        desc = DescriptorCalculator.lipinski_descriptors(mol)
        return {
            'MW_ok': desc['MW'] <= 500,
            'LogP_ok': desc['LogP'] <= 5,
            'HBD_ok': desc['HBD'] <= 5,
            'HBA_ok': desc['HBA'] <= 10,
            'lipinski_compliant': all([
                desc['MW'] <= 500,
                desc['LogP'] <= 5,
                desc['HBD'] <= 5,
                desc['HBA'] <= 10
            ])
        }


def batch_featurize(smiles_list: List[str], 
                   featurizer: Optional[MolecularFeaturizer] = None,
                   batch_size: int = 1000) -> Tuple[np.ndarray, List[str]]:
    """
    Featurize molecules in batches for memory efficiency.
    
    Args:
        smiles_list: List of SMILES strings
        featurizer: MolecularFeaturizer instance
        batch_size: Size of each batch
        
    Returns:
        Feature matrix and list of valid SMILES
    """
    if featurizer is None:
        featurizer = MolecularFeaturizer()
    
    all_features = []
    all_valid_smiles = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        features, valid_smiles = featurizer.featurize_molecules(batch_smiles)
        
        if len(features) > 0:
            all_features.append(features)
            all_valid_smiles.extend(valid_smiles)
        
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i + len(batch_smiles)} molecules")
    
    if all_features:
        final_features = np.vstack(all_features)
    else:
        final_features = np.array([])
    
    return final_features, all_valid_smiles
