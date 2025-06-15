"""
Tests for active learning strategies.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from al_fep.active_learning.uncertainty_sampling import UncertaintySampling
from al_fep.active_learning.advanced_strategies import (
    QueryByCommittee,
    ExpectedImprovement,
    DiversityBasedSampling,
    AdaptiveStrategy,
    create_query_strategy
)


class TestUncertaintySampling:
    """Test UncertaintySampling strategy."""
    
    def test_init(self):
        """Test initialization."""
        strategy = UncertaintySampling()
        assert strategy.uncertainty_method == "entropy"
        assert strategy.diversity_weight == 0.0
        
        strategy = UncertaintySampling(
            uncertainty_method="variance",
            diversity_weight=0.5
        )
        assert strategy.uncertainty_method == "variance"
        assert strategy.diversity_weight == 0.5
    
    def test_select_queries_random_without_model(self, sample_features):
        """Test query selection without trained model (should be random)."""
        strategy = UncertaintySampling()
        candidate_ids = [f"mol_{i}" for i in range(len(sample_features))]
        
        selected_features, selected_ids = strategy.select_queries(
            sample_features, candidate_ids, n_queries=10
        )
        
        assert len(selected_features) == 10
        assert len(selected_ids) == 10
        assert selected_features.shape[1] == sample_features.shape[1]
    
    def test_select_queries_with_model(self, sample_features, sample_targets):
        """Test query selection with trained model."""
        from tests.conftest import MockModel
        
        strategy = UncertaintySampling()
        candidate_ids = [f"mol_{i}" for i in range(len(sample_features))]
        
        # Train a mock model
        model = MockModel()
        model.fit(sample_features[:50], sample_targets[:50])
        
        selected_features, selected_ids = strategy.select_queries(
            sample_features[50:], 
            candidate_ids[50:], 
            n_queries=10,
            model=model
        )
        
        assert len(selected_features) == 10
        assert len(selected_ids) == 10
    
    def test_different_uncertainty_methods(self, sample_features):
        """Test different uncertainty calculation methods."""
        methods = ["entropy", "variance", "std", "confidence"]
        candidate_ids = [f"mol_{i}" for i in range(len(sample_features))]
        
        for method in methods:
            strategy = UncertaintySampling(uncertainty_method=method)
            selected_features, selected_ids = strategy.select_queries(
                sample_features, candidate_ids, n_queries=5
            )
            
            assert len(selected_features) == 5
            assert len(selected_ids) == 5


class TestQueryByCommittee:
    """Test QueryByCommittee strategy."""
    
    def test_init(self):
        """Test initialization."""
        strategy = QueryByCommittee()
        assert strategy.ensemble_size == 5
        assert strategy.disagreement_measure == "variance"
        assert strategy.diversity_weight == 0.1
    
    def test_train_committee(self, sample_features, sample_targets):
        """Test committee training."""
        strategy = QueryByCommittee(ensemble_size=3)
        strategy.train_committee(sample_features, sample_targets)
        
        assert len(strategy.committee) == 3
        assert all(hasattr(model, 'predict') for model in strategy.committee)
    
    def test_select_queries_without_committee(self, sample_features):
        """Test query selection without trained committee."""
        strategy = QueryByCommittee()
        candidate_ids = [f"mol_{i}" for i in range(len(sample_features))]
        
        selected_features, selected_ids = strategy.select_queries(
            sample_features, candidate_ids, n_queries=10
        )
        
        # Should fall back to random selection
        assert len(selected_features) == 10
        assert len(selected_ids) == 10
    
    def test_select_queries_with_committee(self, sample_features, sample_targets):
        """Test query selection with trained committee."""
        strategy = QueryByCommittee(ensemble_size=3)
        strategy.train_committee(sample_features[:50], sample_targets[:50])
        
        candidate_ids = [f"mol_{i}" for i in range(50, len(sample_features))]
        
        selected_features, selected_ids = strategy.select_queries(
            sample_features[50:], 
            candidate_ids, 
            n_queries=10
        )
        
        assert len(selected_features) == 10
        assert len(selected_ids) == 10
    
    def test_different_disagreement_measures(self, sample_features, sample_targets):
        """Test different disagreement measures."""
        measures = ["variance", "std", "range"]
        
        for measure in measures:
            strategy = QueryByCommittee(
                ensemble_size=3, 
                disagreement_measure=measure
            )
            strategy.train_committee(sample_features[:50], sample_targets[:50])
            
            candidate_ids = [f"mol_{i}" for i in range(50, 60)]
            selected_features, selected_ids = strategy.select_queries(
                sample_features[50:60], 
                candidate_ids, 
                n_queries=5
            )
            
            assert len(selected_features) == 5


class TestExpectedImprovement:
    """Test ExpectedImprovement strategy."""
    
    def test_init(self):
        """Test initialization."""
        strategy = ExpectedImprovement()
        assert strategy.xi == 0.01
        assert strategy.use_gp is True
        assert strategy.diversity_weight == 0.1
    
    def test_train_model(self, sample_features, sample_targets):
        """Test model training."""
        strategy = ExpectedImprovement(use_gp=False)  # Use RF for speed
        strategy.train_model(sample_features, sample_targets)
        
        assert strategy.model is not None
        assert strategy.y_best is not None
        assert strategy.y_best == np.max(sample_targets)
    
    def test_select_queries_without_model(self, sample_features):
        """Test query selection without trained model."""
        strategy = ExpectedImprovement()
        candidate_ids = [f"mol_{i}" for i in range(len(sample_features))]
        
        selected_features, selected_ids = strategy.select_queries(
            sample_features, candidate_ids, n_queries=10
        )
        
        # Should fall back to random selection
        assert len(selected_features) == 10
        assert len(selected_ids) == 10
    
    def test_select_queries_with_model(self, sample_features, sample_targets):
        """Test query selection with trained model."""
        strategy = ExpectedImprovement(use_gp=False)  # Use RF for speed
        strategy.train_model(sample_features[:50], sample_targets[:50])
        
        candidate_ids = [f"mol_{i}" for i in range(50, len(sample_features))]
        
        selected_features, selected_ids = strategy.select_queries(
            sample_features[50:], 
            candidate_ids, 
            n_queries=10
        )
        
        assert len(selected_features) == 10
        assert len(selected_ids) == 10


class TestDiversityBasedSampling:
    """Test DiversityBasedSampling strategy."""
    
    def test_init(self):
        """Test initialization."""
        strategy = DiversityBasedSampling()
        assert strategy.diversity_method == "k_means"
        assert strategy.distance_metric == "euclidean"
        assert strategy.uncertainty_weight == 0.5
    
    def test_k_means_selection(self, sample_features):
        """Test k-means based selection."""
        strategy = DiversityBasedSampling(diversity_method="k_means")
        candidate_ids = [f"mol_{i}" for i in range(len(sample_features))]
        
        selected_features, selected_ids = strategy.select_queries(
            sample_features, candidate_ids, n_queries=10
        )
        
        assert len(selected_features) == 10
        assert len(selected_ids) == 10
    
    def test_max_min_selection(self, sample_features):
        """Test max-min based selection."""
        strategy = DiversityBasedSampling(diversity_method="max_min")
        candidate_ids = [f"mol_{i}" for i in range(len(sample_features))]
        
        selected_features, selected_ids = strategy.select_queries(
            sample_features, candidate_ids, n_queries=10
        )
        
        assert len(selected_features) == 10
        assert len(selected_ids) == 10
    
    def test_with_uncertainties(self, sample_features):
        """Test selection with uncertainty scores."""
        strategy = DiversityBasedSampling(uncertainty_weight=0.5)
        candidate_ids = [f"mol_{i}" for i in range(len(sample_features))]
        uncertainties = np.random.rand(len(sample_features))
        
        selected_features, selected_ids = strategy.select_queries(
            sample_features, 
            candidate_ids, 
            n_queries=10,
            uncertainties=uncertainties
        )
        
        assert len(selected_features) == 10
        assert len(selected_ids) == 10
    
    def test_with_labeled_features(self, sample_features):
        """Test selection with labeled features for diversity."""
        strategy = DiversityBasedSampling(diversity_method="max_min")
        candidate_ids = [f"mol_{i}" for i in range(50, len(sample_features))]
        labeled_features = sample_features[:20]  # First 20 as labeled
        
        selected_features, selected_ids = strategy.select_queries(
            sample_features[50:], 
            candidate_ids, 
            n_queries=10,
            labeled_features=labeled_features
        )
        
        assert len(selected_features) == 10
        assert len(selected_ids) == 10


class TestAdaptiveStrategy:
    """Test AdaptiveStrategy."""
    
    def test_init(self):
        """Test initialization."""
        strategies = [
            UncertaintySampling(),
            QueryByCommittee(),
            DiversityBasedSampling()
        ]
        adaptive = AdaptiveStrategy(strategies)
        
        assert len(adaptive.strategies) == 3
        assert len(adaptive.strategy_weights) == 3
        assert adaptive.adaptation_method == "performance_based"
    
    def test_select_queries_round_robin(self, sample_features):
        """Test round-robin strategy selection."""
        strategies = [
            UncertaintySampling(),
            DiversityBasedSampling()
        ]
        adaptive = AdaptiveStrategy(strategies, adaptation_method="round_robin")
        candidate_ids = [f"mol_{i}" for i in range(len(sample_features))]
        
        # First call should use strategy 0
        selected_features, selected_ids = adaptive.select_queries(
            sample_features, candidate_ids, n_queries=5
        )
        assert len(adaptive.selection_history) == 1
        assert adaptive.selection_history[0] == 0
        
        # Second call should use strategy 1
        selected_features, selected_ids = adaptive.select_queries(
            sample_features, candidate_ids, n_queries=5
        )
        assert len(adaptive.selection_history) == 2
        assert adaptive.selection_history[1] == 1
    
    def test_update_performance(self):
        """Test performance update."""
        strategies = [UncertaintySampling(), DiversityBasedSampling()]
        adaptive = AdaptiveStrategy(strategies)
        
        # Update performance for strategy 0
        adaptive.update_performance(0, 0.8)
        adaptive.update_performance(1, 0.6)
        
        assert adaptive.strategy_performance[0] == 0.8
        assert adaptive.strategy_performance[1] == 0.6
        
        # Weights should be updated based on performance
        total_perf = 0.8 + 0.6
        expected_weight_0 = 0.8 / total_perf
        expected_weight_1 = 0.6 / total_perf
        
        assert abs(adaptive.strategy_weights[0] - expected_weight_0) < 1e-6
        assert abs(adaptive.strategy_weights[1] - expected_weight_1) < 1e-6


class TestCreateQueryStrategy:
    """Test query strategy factory function."""
    
    def test_create_uncertainty_strategy(self):
        """Test creating uncertainty sampling strategy."""
        strategy = create_query_strategy("uncertainty")
        assert isinstance(strategy, UncertaintySampling)
    
    def test_create_committee_strategy(self):
        """Test creating committee strategy."""
        strategy = create_query_strategy("committee", ensemble_size=3)
        assert isinstance(strategy, QueryByCommittee)
        assert strategy.ensemble_size == 3
    
    def test_create_expected_improvement_strategy(self):
        """Test creating expected improvement strategy."""
        strategy = create_query_strategy("expected_improvement", xi=0.05)
        assert isinstance(strategy, ExpectedImprovement)
        assert strategy.xi == 0.05
    
    def test_create_diversity_strategy(self):
        """Test creating diversity strategy."""
        strategy = create_query_strategy("diversity", diversity_method="max_min")
        assert isinstance(strategy, DiversityBasedSampling)
        assert strategy.diversity_method == "max_min"
    
    def test_create_adaptive_strategy(self):
        """Test creating adaptive strategy."""
        strategies = [UncertaintySampling(), QueryByCommittee()]
        strategy = create_query_strategy("adaptive", strategies=strategies)
        assert isinstance(strategy, AdaptiveStrategy)
        assert len(strategy.strategies) == 2
    
    def test_create_unknown_strategy(self):
        """Test creating unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_query_strategy("unknown_strategy")
    
    def test_case_insensitive(self):
        """Test that strategy names are case insensitive."""
        strategy1 = create_query_strategy("UNCERTAINTY")
        strategy2 = create_query_strategy("Uncertainty")
        strategy3 = create_query_strategy("uncertainty")
        
        assert all(isinstance(s, UncertaintySampling) for s in [strategy1, strategy2, strategy3])


@pytest.mark.parametrize("strategy_name,strategy_class", [
    ("uncertainty", UncertaintySampling),
    ("committee", QueryByCommittee),
    ("expected_improvement", ExpectedImprovement),
    ("diversity", DiversityBasedSampling),
])
def test_strategy_interface_consistency(strategy_name, strategy_class, sample_features):
    """Test that all strategies have consistent interface."""
    strategy = create_query_strategy(strategy_name)
    candidate_ids = [f"mol_{i}" for i in range(len(sample_features))]
    
    # All strategies should be able to select queries
    selected_features, selected_ids = strategy.select_queries(
        sample_features, candidate_ids, n_queries=5
    )
    
    assert len(selected_features) == 5
    assert len(selected_ids) == 5
    assert isinstance(selected_features, np.ndarray)
    assert isinstance(selected_ids, list)
    assert selected_features.shape[1] == sample_features.shape[1]


def test_query_selection_edge_cases(sample_features):
    """Test edge cases in query selection."""
    strategy = UncertaintySampling()
    candidate_ids = [f"mol_{i}" for i in range(len(sample_features))]
    
    # Request more queries than available candidates
    selected_features, selected_ids = strategy.select_queries(
        sample_features[:5], candidate_ids[:5], n_queries=10
    )
    assert len(selected_features) == 5  # Should return all available
    assert len(selected_ids) == 5
    
    # Request zero queries
    selected_features, selected_ids = strategy.select_queries(
        sample_features, candidate_ids, n_queries=0
    )
    assert len(selected_features) == 0
    assert len(selected_ids) == 0
    
    # Empty candidate pool
    selected_features, selected_ids = strategy.select_queries(
        np.array([]).reshape(0, sample_features.shape[1]), [], n_queries=5
    )
    assert len(selected_features) == 0
    assert len(selected_ids) == 0
