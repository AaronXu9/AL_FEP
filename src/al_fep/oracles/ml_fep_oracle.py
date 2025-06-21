"""
ML-FEP Oracle: Machine Learning-based FEP predictions
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import joblib

from .base_oracle import BaseOracle

logger = logging.getLogger(__name__)


class MLFEPOracle(BaseOracle):
    """
    Oracle for fast ML-based FEP predictions.
    
    This oracle uses machine learning models trained on FEP data
    to provide fast, low-cost predictions with uncertainty estimates.
    """
    
    def __init__(
        self,
        target: str,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the ML-FEP oracle.
        
        Args:
            target: Target identifier
            model_path: Path to trained ML model
            config: Configuration dictionary
        """
        super().__init__(name="ML-FEP", target=target, config=config, **kwargs)
        
        self.model_path = model_path or self._get_default_model_path()
        
        # ML-FEP parameters from config
        ml_config = self.config.get("ml_fep", {})
        self.model_type = ml_config.get("model_type", "ensemble")
        self.uncertainty_threshold = ml_config.get("uncertainty_threshold", 0.5)
        self.retrain_frequency = ml_config.get("retrain_frequency", 100)
        
        # Initialize models
        self.models = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        # Load pre-trained model if available
        self._load_model()
        
        # If no model available, initialize with default
        if not self.is_trained:
            self._initialize_default_model()
        
        logger.info(f"ML-FEP Oracle initialized for target {self.target}")
    
    def _get_default_model_path(self) -> str:
        """Get default model file path."""
        return f"data/models/{self.target}_ml_fep_model.pkl"
    
    def _initialize_default_model(self):
        """Initialize a default model for demo purposes."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Create ensemble of models for uncertainty estimation
            self.models = [
                RandomForestRegressor(n_estimators=100, random_state=i)
                for i in range(5)
            ]
            self.scaler = StandardScaler()
            
            # Define feature names
            self.feature_names = [
                'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
                'NumRotatableBonds', 'TPSA', 'NumAromaticRings',
                'NumSaturatedRings', 'NumHeteroatoms', 'RingCount',
                'FractionCsp3', 'NumAliphaticCarbocycles', 'NumAromaticCarbocycles'
            ]
            
            # Generate some mock training data for demonstration
            self._generate_mock_training_data()
            
            logger.info("Initialized default ML-FEP model")
            
        except ImportError:
            logger.error("Scikit-learn not available for ML-FEP oracle")
            raise ImportError("Install scikit-learn: conda install scikit-learn")
    
    def _generate_mock_training_data(self):
        """Generate mock training data for demonstration."""
        try:
            np.random.seed(42)
            
            # Generate random molecular descriptors
            n_samples = 1000
            X = np.random.randn(n_samples, len(self.feature_names))
            
            # Create mock FEP values with some correlation to descriptors
            y = (
                -5.0 + 
                0.01 * X[:, 0] +  # MolWt
                -1.0 * X[:, 1] +  # MolLogP  
                0.5 * X[:, 2] +   # NumHDonors
                -0.3 * X[:, 3] +  # NumHAcceptors
                np.random.normal(0, 1.0, n_samples)  # noise
            )
            
            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)
            
            # Train ensemble models
            for model in self.models:
                # Add noise to create model diversity
                y_noisy = y + np.random.normal(0, 0.1, len(y))
                model.fit(X_scaled, y_noisy)
            
            self.is_trained = True
            logger.info("Generated mock training data and trained ML-FEP models")
            
        except Exception as e:
            logger.error(f"Error generating mock training data: {e}")
    
    def _load_model(self):
        """Load pre-trained model from file."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models = model_data['models']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_trained = True
                
                logger.info(f"Loaded ML-FEP model from {self.model_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
    
    def save_model(self, filepath: Optional[str] = None):
        """Save trained model to file."""
        if not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        filepath = filepath or self.model_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'target': self.target,
                'model_type': self.model_type
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved ML-FEP model to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _calculate_molecular_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        """
        Calculate molecular descriptors for ML prediction.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Feature vector or None if calculation failed
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Calculate descriptors
            features = []
            
            # Basic descriptors
            features.append(Descriptors.MolWt(mol))
            features.append(Descriptors.MolLogP(mol))
            features.append(Descriptors.NumHDonors(mol))
            features.append(Descriptors.NumHAcceptors(mol))
            features.append(Descriptors.NumRotatableBonds(mol))
            features.append(Descriptors.TPSA(mol))
            features.append(Descriptors.NumAromaticRings(mol))
            features.append(Descriptors.NumSaturatedRings(mol))
            features.append(Descriptors.NumHeteroatoms(mol))
            features.append(Descriptors.RingCount(mol))
            features.append(Descriptors.FractionCSP3(mol))
            features.append(Descriptors.NumAliphaticCarbocycles(mol))
            features.append(Descriptors.NumAromaticCarbocycles(mol))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error calculating descriptors: {e}")
            return None
    
    def _predict_with_uncertainty(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Make prediction with uncertainty estimate.
        
        Args:
            features: Molecular feature vector
            
        Returns:
            Tuple of (prediction, uncertainty)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from ensemble
        predictions = []
        for model in self.models:
            pred = model.predict(features_scaled)[0]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate mean and uncertainty (standard deviation)
        mean_pred = np.mean(predictions)
        uncertainty = np.std(predictions)
        
        return mean_pred, uncertainty
    
    def _evaluate_single(self, smiles: str) -> Dict[str, Any]:
        """
        Evaluate a single molecule using ML-FEP.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary containing ML-FEP results
        """
        if not self.is_trained:
            return {
                "score": None,
                "error": "Model not trained",
                "ml_fep_score": None,
                "uncertainty": None,
                "confidence": None
            }
        
        # Calculate molecular descriptors
        features = self._calculate_molecular_descriptors(smiles)
        
        if features is None:
            return {
                "score": None,
                "error": "Failed to calculate molecular descriptors",
                "ml_fep_score": None,
                "uncertainty": None,
                "confidence": None
            }
        
        try:
            # Make prediction with uncertainty
            ml_fep_score, uncertainty = self._predict_with_uncertainty(features)
            
            # Calculate confidence (inverse of uncertainty)
            confidence = 1.0 / (1.0 + uncertainty)
            
            # Convert to positive score (more negative is better for binding)
            score = -ml_fep_score
            
            # Check if uncertainty is above threshold
            high_uncertainty = uncertainty > self.uncertainty_threshold
            
            return {
                "score": score,
                "ml_fep_score": ml_fep_score,
                "uncertainty": uncertainty,
                "confidence": confidence,
                "high_uncertainty": high_uncertainty,
                "error": None,
                "method": "ML-FEP",
                "model_type": self.model_type
            }
            
        except Exception as e:
            logger.error(f"Error in ML-FEP prediction: {e}")
            return {
                "score": None,
                "error": str(e),
                "ml_fep_score": None,
                "uncertainty": None,
                "confidence": None
            }
    
    def retrain_model(self, smiles_list: list, fep_scores: list):
        """
        Retrain the model with new data.
        
        Args:
            smiles_list: List of SMILES strings
            fep_scores: List of corresponding FEP scores
        """
        try:
            # Calculate features for all molecules
            features_list = []
            valid_scores = []
            
            for smiles, score in zip(smiles_list, fep_scores):
                features = self._calculate_molecular_descriptors(smiles)
                if features is not None and score is not None:
                    features_list.append(features[0])
                    valid_scores.append(score)
            
            if len(features_list) == 0:
                logger.warning("No valid training data provided")
                return
            
            X = np.array(features_list)
            y = np.array(valid_scores)
            
            # Refit scaler and models
            X_scaled = self.scaler.fit_transform(X)
            
            for model in self.models:
                # Add noise for ensemble diversity
                y_noisy = y + np.random.normal(0, 0.1, len(y))
                model.fit(X_scaled, y_noisy)
            
            logger.info(f"Retrained ML-FEP model with {len(y)} samples")
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained models.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or not hasattr(self.models[0], 'feature_importances_'):
            return {}
        
        # Average feature importance across ensemble
        importance_sum = np.zeros(len(self.feature_names))
        
        for model in self.models:
            importance_sum += model.feature_importances_
        
        avg_importance = importance_sum / len(self.models)
        
        return dict(zip(self.feature_names, avg_importance))
