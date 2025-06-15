"""
Advanced active learning strategies for molecular optimization.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


class QueryStrategy(ABC):
    """Base class for active learning query strategies."""
    
    @abstractmethod
    def select_queries(self, 
                      candidates: np.ndarray,
                      candidate_ids: List[str],
                      n_queries: int,
                      model: Optional[object] = None,
                      **kwargs) -> Tuple[np.ndarray, List[str]]:
        """
        Select queries from candidate pool.
        
        Args:
            candidates: Candidate feature matrix
            candidate_ids: Candidate identifiers
            n_queries: Number of queries to select
            model: Optional trained model
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Selected candidate features and IDs
        """
        pass


class QueryByCommittee(QueryStrategy):
    """Query by Committee strategy using ensemble disagreement."""
    
    def __init__(self, 
                 ensemble_size: int = 5,
                 disagreement_measure: str = "variance",
                 diversity_weight: float = 0.1):
        """
        Initialize Query by Committee strategy.
        
        Args:
            ensemble_size: Number of models in committee
            disagreement_measure: How to measure disagreement ('variance', 'std', 'range')
            diversity_weight: Weight for diversity component
        """
        self.ensemble_size = ensemble_size
        self.disagreement_measure = disagreement_measure
        self.diversity_weight = diversity_weight
        self.committee = []
    
    def train_committee(self, X: np.ndarray, y: np.ndarray):
        """Train committee of models."""
        self.committee = []
        n_samples = len(X)
        
        for i in range(self.ensemble_size):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model with different random states
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=i,
                n_jobs=-1
            )
            model.fit(X_boot, y_boot)
            self.committee.append(model)
    
    def select_queries(self, 
                      candidates: np.ndarray,
                      candidate_ids: List[str],
                      n_queries: int,
                      model: Optional[object] = None,
                      labeled_features: Optional[np.ndarray] = None,
                      **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Select queries using committee disagreement."""
        if not self.committee:
            logger.warning("Committee not trained. Using random selection.")
            indices = np.random.choice(len(candidates), size=min(n_queries, len(candidates)), replace=False)
            return candidates[indices], [candidate_ids[i] for i in indices]
        
        # Get predictions from all committee members
        predictions = []
        for model in self.committee:
            pred = model.predict(candidates)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate disagreement
        if self.disagreement_measure == "variance":
            disagreement = np.var(predictions, axis=0)
        elif self.disagreement_measure == "std":
            disagreement = np.std(predictions, axis=0)
        elif self.disagreement_measure == "range":
            disagreement = np.max(predictions, axis=0) - np.min(predictions, axis=0)
        else:
            disagreement = np.var(predictions, axis=0)
        
        # Add diversity component
        if self.diversity_weight > 0 and labeled_features is not None:
            diversity_scores = self._calculate_diversity(candidates, labeled_features)
            total_scores = disagreement + self.diversity_weight * diversity_scores
        else:
            total_scores = disagreement
        
        # Select top queries
        top_indices = np.argsort(total_scores)[-n_queries:][::-1]
        
        return candidates[top_indices], [candidate_ids[i] for i in top_indices]
    
    def _calculate_diversity(self, candidates: np.ndarray, labeled_features: np.ndarray) -> np.ndarray:
        """Calculate diversity scores for candidates."""
        # Calculate minimum distance to labeled data
        distances = pairwise_distances(candidates, labeled_features, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        # Normalize to [0, 1]
        if np.max(min_distances) > np.min(min_distances):
            diversity_scores = (min_distances - np.min(min_distances)) / (np.max(min_distances) - np.min(min_distances))
        else:
            diversity_scores = np.ones(len(candidates))
        
        return diversity_scores


class ExpectedImprovement(QueryStrategy):
    """Expected Improvement strategy for Bayesian optimization."""
    
    def __init__(self, 
                 xi: float = 0.01,
                 use_gp: bool = True,
                 gp_kernel: str = "rbf",
                 diversity_weight: float = 0.1):
        """
        Initialize Expected Improvement strategy.
        
        Args:
            xi: Exploration parameter
            use_gp: Whether to use Gaussian Process
            gp_kernel: Kernel for GP ('rbf', 'matern')
            diversity_weight: Weight for diversity component
        """
        self.xi = xi
        self.use_gp = use_gp
        self.gp_kernel = gp_kernel
        self.diversity_weight = diversity_weight
        self.model = None
        self.y_best = None
    
    def _get_gp_model(self):
        """Get Gaussian Process model."""
        if self.gp_kernel == "rbf":
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        elif self.gp_kernel == "matern":
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        else:
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=10,
            random_state=42
        )
    
    def train_model(self, X: np.ndarray, y: np.ndarray):
        """Train the underlying model."""
        if self.use_gp:
            self.model = self._get_gp_model()
        else:
            # Use RF with uncertainty estimation
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X, y)
        self.y_best = np.max(y)
    
    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """Calculate expected improvement."""
        if self.use_gp:
            mu, sigma = self.model.predict(X, return_std=True)
        else:
            # Estimate uncertainty using RF predictions
            predictions = []
            for estimator in self.model.estimators_:
                pred = estimator.predict(X)
                predictions.append(pred)
            
            mu = np.mean(predictions, axis=0)
            sigma = np.std(predictions, axis=0)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-9)
        
        # Calculate expected improvement
        improvement = mu - self.y_best - self.xi
        Z = improvement / sigma
        
        # Using normal CDF and PDF approximations
        from scipy.stats import norm
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei
    
    def select_queries(self, 
                      candidates: np.ndarray,
                      candidate_ids: List[str],
                      n_queries: int,
                      model: Optional[object] = None,
                      labeled_features: Optional[np.ndarray] = None,
                      **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Select queries using expected improvement."""
        if self.model is None:
            logger.warning("Model not trained. Using random selection.")
            indices = np.random.choice(len(candidates), size=min(n_queries, len(candidates)), replace=False)
            return candidates[indices], [candidate_ids[i] for i in indices]
        
        # Calculate expected improvement
        ei_scores = self._expected_improvement(candidates)
        
        # Add diversity component
        if self.diversity_weight > 0 and labeled_features is not None:
            diversity_scores = self._calculate_diversity(candidates, labeled_features)
            total_scores = ei_scores + self.diversity_weight * diversity_scores
        else:
            total_scores = ei_scores
        
        # Select top queries
        top_indices = np.argsort(total_scores)[-n_queries:][::-1]
        
        return candidates[top_indices], [candidate_ids[i] for i in top_indices]
    
    def _calculate_diversity(self, candidates: np.ndarray, labeled_features: np.ndarray) -> np.ndarray:
        """Calculate diversity scores for candidates."""
        distances = pairwise_distances(candidates, labeled_features, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        if np.max(min_distances) > np.min(min_distances):
            diversity_scores = (min_distances - np.min(min_distances)) / (np.max(min_distances) - np.min(min_distances))
        else:
            diversity_scores = np.ones(len(candidates))
        
        return diversity_scores


class DiversityBasedSampling(QueryStrategy):
    """Diversity-based sampling strategy."""
    
    def __init__(self, 
                 diversity_method: str = "k_means",
                 distance_metric: str = "euclidean",
                 uncertainty_weight: float = 0.5):
        """
        Initialize diversity-based sampling.
        
        Args:
            diversity_method: Method for diversity ('k_means', 'max_min', 'core_set')
            distance_metric: Distance metric to use
            uncertainty_weight: Weight for uncertainty component
        """
        self.diversity_method = diversity_method
        self.distance_metric = distance_metric
        self.uncertainty_weight = uncertainty_weight
    
    def select_queries(self, 
                      candidates: np.ndarray,
                      candidate_ids: List[str],
                      n_queries: int,
                      model: Optional[object] = None,
                      labeled_features: Optional[np.ndarray] = None,
                      uncertainties: Optional[np.ndarray] = None,
                      **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Select queries using diversity-based sampling."""
        
        if self.diversity_method == "k_means":
            return self._k_means_selection(candidates, candidate_ids, n_queries, uncertainties)
        elif self.diversity_method == "max_min":
            return self._max_min_selection(candidates, candidate_ids, n_queries, labeled_features, uncertainties)
        elif self.diversity_method == "core_set":
            return self._core_set_selection(candidates, candidate_ids, n_queries, labeled_features, uncertainties)
        else:
            # Default to k-means
            return self._k_means_selection(candidates, candidate_ids, n_queries, uncertainties)
    
    def _k_means_selection(self, 
                          candidates: np.ndarray,
                          candidate_ids: List[str],
                          n_queries: int,
                          uncertainties: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
        """Select queries using k-means clustering."""
        n_queries = min(n_queries, len(candidates))
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_queries, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(candidates)
        
        selected_indices = []
        
        # Select one point from each cluster
        for cluster_id in range(n_queries):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # If we have uncertainties, select the most uncertain point in the cluster
            if uncertainties is not None and self.uncertainty_weight > 0:
                cluster_uncertainties = uncertainties[cluster_indices]
                best_in_cluster = cluster_indices[np.argmax(cluster_uncertainties)]
            else:
                # Select point closest to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                cluster_points = candidates[cluster_indices]
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                best_in_cluster = cluster_indices[np.argmin(distances)]
            
            selected_indices.append(best_in_cluster)
        
        return candidates[selected_indices], [candidate_ids[i] for i in selected_indices]
    
    def _max_min_selection(self, 
                          candidates: np.ndarray,
                          candidate_ids: List[str],
                          n_queries: int,
                          labeled_features: Optional[np.ndarray] = None,
                          uncertainties: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
        """Select queries using max-min diversity."""
        n_queries = min(n_queries, len(candidates))
        
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        # Initialize with labeled data if available
        if labeled_features is not None:
            all_features = np.vstack([labeled_features, candidates])
            reference_points = labeled_features
        else:
            # Start with random point
            first_idx = np.random.choice(remaining_indices)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            reference_points = candidates[selected_indices]
        
        # Iteratively select points that maximize minimum distance
        for _ in range(len(selected_indices), n_queries):
            if not remaining_indices:
                break
            
            best_idx = None
            best_score = -1
            
            for idx in remaining_indices:
                candidate_point = candidates[idx:idx+1]
                
                # Calculate minimum distance to reference points
                distances = pairwise_distances(candidate_point, reference_points, metric=self.distance_metric)
                min_distance = np.min(distances)
                
                # Add uncertainty component if available
                if uncertainties is not None and self.uncertainty_weight > 0:
                    uncertainty_score = uncertainties[idx]
                    score = min_distance + self.uncertainty_weight * uncertainty_score
                else:
                    score = min_distance
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                reference_points = np.vstack([reference_points, candidates[best_idx:best_idx+1]])
        
        return candidates[selected_indices], [candidate_ids[i] for i in selected_indices]
    
    def _core_set_selection(self, 
                           candidates: np.ndarray,
                           candidate_ids: List[str],
                           n_queries: int,
                           labeled_features: Optional[np.ndarray] = None,
                           uncertainties: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
        """Select queries using core-set approach."""
        # For simplicity, use a greedy approximation to the k-center problem
        return self._max_min_selection(candidates, candidate_ids, n_queries, labeled_features, uncertainties)


class AdaptiveStrategy(QueryStrategy):
    """Adaptive strategy that combines multiple approaches."""
    
    def __init__(self, 
                 strategies: List[QueryStrategy],
                 strategy_weights: Optional[List[float]] = None,
                 adaptation_method: str = "performance_based"):
        """
        Initialize adaptive strategy.
        
        Args:
            strategies: List of query strategies to combine
            strategy_weights: Weights for each strategy
            adaptation_method: How to adapt weights ('performance_based', 'round_robin', 'random')
        """
        self.strategies = strategies
        self.strategy_weights = strategy_weights or [1.0] * len(strategies)
        self.adaptation_method = adaptation_method
        self.strategy_performance = [0.0] * len(strategies)
        self.selection_history = []
    
    def update_performance(self, strategy_idx: int, performance: float):
        """Update performance score for a strategy."""
        self.strategy_performance[strategy_idx] = performance
        
        # Update weights based on performance
        if self.adaptation_method == "performance_based":
            total_perf = sum(self.strategy_performance)
            if total_perf > 0:
                self.strategy_weights = [p / total_perf for p in self.strategy_performance]
    
    def select_queries(self, 
                      candidates: np.ndarray,
                      candidate_ids: List[str],
                      n_queries: int,
                      model: Optional[object] = None,
                      **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Select queries using adaptive strategy combination."""
        
        if self.adaptation_method == "round_robin":
            # Round-robin selection
            strategy_idx = len(self.selection_history) % len(self.strategies)
            selected_strategy = self.strategies[strategy_idx]
        elif self.adaptation_method == "random":
            # Random selection with weights
            strategy_idx = np.random.choice(len(self.strategies), p=self.strategy_weights)
            selected_strategy = self.strategies[strategy_idx]
        else:
            # Performance-based selection
            strategy_idx = np.argmax(self.strategy_weights)
            selected_strategy = self.strategies[strategy_idx]
        
        # Record selection
        self.selection_history.append(strategy_idx)
        
        # Use selected strategy
        return selected_strategy.select_queries(candidates, candidate_ids, n_queries, model, **kwargs)


def create_query_strategy(strategy_name: str, **kwargs) -> QueryStrategy:
    """Factory function to create query strategies."""
    
    # Import here to avoid circular imports
    try:
        from .uncertainty_sampling import UncertaintySampling
    except ImportError:
        UncertaintySampling = None
    
    strategy_map = {
        'committee': lambda: QueryByCommittee(**kwargs),
        'expected_improvement': lambda: ExpectedImprovement(**kwargs),
        'diversity': lambda: DiversityBasedSampling(**kwargs),
        'adaptive': lambda: AdaptiveStrategy(**kwargs)
    }
    
    # Add uncertainty sampling if available
    if UncertaintySampling is not None:
        strategy_map['uncertainty'] = lambda: UncertaintySampling(**kwargs)
    
    if strategy_name.lower() not in strategy_map:
        available = list(strategy_map.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    
    return strategy_map[strategy_name.lower()]()
