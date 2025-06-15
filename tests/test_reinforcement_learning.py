"""
Tests for reinforcement learning components.
"""

import pytest
import torch
import numpy as np
import gym
from unittest.mock import Mock, patch

from al_fep.reinforcement.ppo_agent import PPOAgent, ActorCritic, PPOTrainer
from al_fep.reinforcement.molecular_env import MolecularEnvironment


class TestActorCritic:
    """Test ActorCritic network."""
    
    def test_init(self):
        """Test network initialization."""
        network = ActorCritic(state_dim=10, action_dim=5)
        assert network.state_dim == 10
        assert network.action_dim == 5
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        network = ActorCritic(state_dim=10, action_dim=5)
        state = torch.randn(1, 10)
        
        action_probs, state_value = network(state)
        
        assert action_probs.shape == (1, 5)
        assert state_value.shape == (1, 1)
        assert torch.allclose(action_probs.sum(dim=1), torch.ones(1))  # Probabilities sum to 1
    
    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        network = ActorCritic(state_dim=10, action_dim=5)
        state = torch.randn(1, 10)
        
        action, log_prob, state_value = network.get_action(state, deterministic=True)
        
        assert isinstance(action, int)
        assert 0 <= action < 5
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(state_value, torch.Tensor)
    
    def test_get_action_stochastic(self):
        """Test stochastic action selection."""
        network = ActorCritic(state_dim=10, action_dim=5)
        state = torch.randn(1, 10)
        
        action, log_prob, state_value = network.get_action(state, deterministic=False)
        
        assert isinstance(action, int)
        assert 0 <= action < 5
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(state_value, torch.Tensor)
    
    def test_custom_hidden_dims(self):
        """Test custom hidden dimensions."""
        network = ActorCritic(
            state_dim=20, 
            action_dim=10, 
            hidden_dims=[128, 64, 32]
        )
        state = torch.randn(1, 20)
        
        action_probs, state_value = network(state)
        
        assert action_probs.shape == (1, 10)
        assert state_value.shape == (1, 1)


class TestPPOAgent:
    """Test PPO agent."""
    
    def test_init(self):
        """Test agent initialization."""
        agent = PPOAgent(state_dim=10, action_dim=5)
        
        assert agent.network is not None
        assert agent.optimizer is not None
        assert agent.gamma == 0.99
        assert agent.clip_epsilon == 0.2
        assert len(agent.states) == 0
    
    def test_get_action(self):
        """Test action selection."""
        agent = PPOAgent(state_dim=10, action_dim=5)
        state = np.random.randn(10)
        
        action, log_prob, value = agent.get_action(state)
        
        assert isinstance(action, int)
        assert 0 <= action < 5
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_store_experience(self):
        """Test experience storage."""
        agent = PPOAgent(state_dim=10, action_dim=5)
        
        state = np.random.randn(10)
        agent.store_experience(state, 2, 0.5, 1.0, 0.8, False)
        
        assert len(agent.states) == 1
        assert len(agent.actions) == 1
        assert len(agent.log_probs) == 1
        assert len(agent.rewards) == 1
        assert len(agent.values) == 1
        assert len(agent.dones) == 1
    
    def test_compute_gae(self):
        """Test GAE computation."""
        agent = PPOAgent(state_dim=10, action_dim=5)
        
        # Add some experience
        for i in range(5):
            state = np.random.randn(10)
            agent.store_experience(state, i % 5, 0.5, 1.0, 0.8, False)
        
        returns, advantages = agent.compute_gae(next_value=0.0)
        
        assert len(returns) == 5
        assert len(advantages) == 5
        assert isinstance(returns, list)
        assert isinstance(advantages, list)
    
    def test_update_empty_buffer(self):
        """Test update with empty buffer."""
        agent = PPOAgent(state_dim=10, action_dim=5)
        
        stats = agent.update()
        
        assert isinstance(stats, dict)
        assert len(stats) == 0  # Should return empty dict
    
    def test_update_with_experience(self):
        """Test update with experience."""
        agent = PPOAgent(state_dim=10, action_dim=5, mini_batch_size=2)
        
        # Add some experience
        for i in range(10):
            state = np.random.randn(10)
            agent.store_experience(state, i % 5, 0.5, 1.0, 0.8, i == 9)
        
        stats = agent.update()
        
        assert isinstance(stats, dict)
        assert 'policy_loss' in stats
        assert 'value_loss' in stats
        assert 'entropy' in stats
        assert 'total_loss' in stats
        assert 'grad_norm' in stats
        
        # Buffer should be reset after update
        assert len(agent.states) == 0
    
    def test_save_load(self, tmp_path):
        """Test saving and loading agent."""
        agent1 = PPOAgent(state_dim=10, action_dim=5)
        
        # Store some experience and update to change network weights
        for i in range(5):
            state = np.random.randn(10)
            agent1.store_experience(state, i % 5, 0.5, 1.0, 0.8, False)
        agent1.update()
        
        # Save agent
        save_path = tmp_path / "agent.pt"
        agent1.save(str(save_path))
        
        # Create new agent and load
        agent2 = PPOAgent(state_dim=10, action_dim=5)
        agent2.load(str(save_path))
        
        # Test that loaded agent produces same outputs
        state = np.random.randn(10)
        action1, _, _ = agent1.get_action(state, deterministic=True)
        action2, _, _ = agent2.get_action(state, deterministic=True)
        
        assert action1 == action2
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        agent = PPOAgent(state_dim=10, action_dim=5)
        
        # Add some training statistics
        agent.training_stats['policy_loss'].extend([0.1, 0.2, 0.3])
        agent.training_stats['value_loss'].extend([0.05, 0.06, 0.07])
        
        stats = agent.get_stats()
        
        assert 'policy_loss_mean' in stats
        assert 'policy_loss_std' in stats
        assert 'value_loss_mean' in stats
        assert 'value_loss_std' in stats
        
        assert abs(stats['policy_loss_mean'] - 0.2) < 1e-6
    
    def test_device_setting(self):
        """Test device setting."""
        # Test auto device selection
        agent = PPOAgent(state_dim=10, action_dim=5, device="auto")
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert agent.device == expected_device
        
        # Test explicit CPU
        agent = PPOAgent(state_dim=10, action_dim=5, device="cpu")
        assert agent.device == torch.device("cpu")


class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self, state_dim=10, action_dim=5, max_steps=20):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.observation_space = Mock()
        self.observation_space.shape = (state_dim,)
        self.action_space = Mock()
        self.action_space.n = action_dim
    
    def reset(self):
        """Reset environment."""
        self.current_step = 0
        return np.random.randn(self.state_dim)
    
    def step(self, action):
        """Take step in environment."""
        self.current_step += 1
        next_state = np.random.randn(self.state_dim)
        reward = np.random.randn()  # Random reward
        done = self.current_step >= self.max_steps
        info = {}
        return next_state, reward, done, info


class TestPPOTrainer:
    """Test PPO trainer."""
    
    def test_init(self):
        """Test trainer initialization."""
        agent = PPOAgent(state_dim=10, action_dim=5)
        env = MockEnvironment()
        trainer = PPOTrainer(agent, env)
        
        assert trainer.agent is agent
        assert trainer.env is env
        assert len(trainer.episode_rewards) == 0
        assert len(trainer.episode_lengths) == 0
    
    def test_train_episode(self):
        """Test training single episode."""
        agent = PPOAgent(state_dim=10, action_dim=5)
        env = MockEnvironment(max_steps=10)
        trainer = PPOTrainer(agent, env)
        
        stats = trainer.train_episode(max_steps=20)
        
        assert isinstance(stats, dict)
        assert 'episode_reward' in stats
        assert 'episode_length' in stats
        assert 'avg_reward' in stats
        assert 'avg_length' in stats
        
        # Check that episode was recorded
        assert len(trainer.episode_rewards) == 1
        assert len(trainer.episode_lengths) == 1
    
    def test_train_multiple_episodes(self):
        """Test training multiple episodes."""
        agent = PPOAgent(state_dim=10, action_dim=5)
        env = MockEnvironment(max_steps=5)
        trainer = PPOTrainer(agent, env)
        
        history = trainer.train(num_episodes=3, save_freq=10, eval_freq=10)
        
        assert len(history) == 3
        assert all(isinstance(stats, dict) for stats in history)
        assert len(trainer.episode_rewards) == 3
    
    def test_evaluate(self):
        """Test agent evaluation."""
        agent = PPOAgent(state_dim=10, action_dim=5)
        env = MockEnvironment(max_steps=5)
        trainer = PPOTrainer(agent, env)
        
        eval_stats = trainer.evaluate(num_episodes=3)
        
        assert isinstance(eval_stats, dict)
        assert 'eval_reward_mean' in eval_stats
        assert 'eval_reward_std' in eval_stats
        assert 'eval_length_mean' in eval_stats
        assert 'eval_length_std' in eval_stats


class TestMolecularEnvironment:
    """Test MolecularEnvironment."""
    
    def test_init(self):
        """Test environment initialization."""
        from tests.conftest import MockOracle
        
        oracle = MockOracle()
        env = MolecularEnvironment(oracle=oracle)
        
        assert env.oracle is oracle
        assert env.max_length > 0
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
    
    @patch('al_fep.reinforcement.molecular_env.MolecularEnvironment._get_valid_actions')
    def test_reset(self, mock_get_valid_actions):
        """Test environment reset."""
        from tests.conftest import MockOracle
        
        mock_get_valid_actions.return_value = [0, 1, 2, 3, 4]
        oracle = MockOracle()
        env = MolecularEnvironment(oracle=oracle)
        
        state = env.reset()
        
        assert isinstance(state, np.ndarray)
        assert len(state) == env.observation_space.shape[0]
    
    @patch('al_fep.reinforcement.molecular_env.MolecularEnvironment._is_valid_molecule')
    @patch('al_fep.reinforcement.molecular_env.MolecularEnvironment._get_valid_actions')
    def test_step(self, mock_get_valid_actions, mock_is_valid):
        """Test environment step."""
        from tests.conftest import MockOracle
        
        mock_get_valid_actions.return_value = [0, 1, 2, 3, 4]
        mock_is_valid.return_value = True
        
        oracle = MockOracle()
        env = MolecularEnvironment(oracle=oracle)
        
        state = env.reset()
        next_state, reward, done, info = env.step(0)
        
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)


@pytest.mark.parametrize("state_dim,action_dim", [
    (5, 3),
    (20, 10),
    (100, 50)
])
def test_agent_different_sizes(state_dim, action_dim):
    """Test agent with different state and action dimensions."""
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    
    state = np.random.randn(state_dim)
    action, log_prob, value = agent.get_action(state)
    
    assert 0 <= action < action_dim
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_ppo_training_convergence():
    """Test that PPO training shows some convergence behavior."""
    # Simple environment where the optimal action is always 0
    class SimpleEnv:
        def __init__(self):
            self.observation_space = Mock()
            self.observation_space.shape = (5,)
            self.action_space = Mock()
            self.action_space.n = 3
            self.step_count = 0
        
        def reset(self):
            self.step_count = 0
            return np.ones(5)  # Always return same state
        
        def step(self, action):
            self.step_count += 1
            reward = 1.0 if action == 0 else -1.0  # Reward action 0
            done = self.step_count >= 10
            return np.ones(5), reward, done, {}
    
    agent = PPOAgent(state_dim=5, action_dim=3, lr=0.01)
    env = SimpleEnv()
    trainer = PPOTrainer(agent, env)
    
    # Train for a few episodes
    history = trainer.train(num_episodes=5)
    
    # Check that we got some training statistics
    assert len(history) == 5
    assert all('episode_reward' in stats for stats in history)
    
    # The agent should learn to prefer action 0 over time
    # (This is a weak test due to stochasticity, but should generally hold)
    state = np.ones(5)
    action_counts = {0: 0, 1: 0, 2: 0}
    
    for _ in range(100):
        action, _, _ = agent.get_action(state, deterministic=False)
        action_counts[action] += 1
    
    # Action 0 should be selected more often than others
    # (With some tolerance for randomness)
    assert action_counts[0] >= action_counts[1]
    assert action_counts[0] >= action_counts[2]


def test_memory_management():
    """Test that PPO doesn't leak memory during training."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    agent = PPOAgent(state_dim=50, action_dim=10)
    env = MockEnvironment(state_dim=50, action_dim=10, max_steps=100)
    trainer = PPOTrainer(agent, env)
    
    # Train for multiple episodes
    trainer.train(num_episodes=10)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (less than 50MB)
    assert memory_increase < 50 * 1024 * 1024
