"""
PPO (Proximal Policy Optimization) agent for molecular generation and optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from collections import deque
import gym

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = "relu"):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # Shared layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            self.activation,
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            self.activation,
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action probabilities and state value
        """
        shared_features = self.shared_layers(state)
        action_probs = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        
        return action_probs, state_value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Get action from the policy.
        
        Args:
            state: Input state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action, log probability, and state value
        """
        action_probs, state_value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)))
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, state_value


class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 mini_batch_size: int = 64,
                 hidden_dims: List[int] = [256, 256],
                 device: str = "auto"):
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize network
        self.network = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # PPO parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # Experience buffer
        self.reset_buffer()
        
        # Statistics
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'total_loss': deque(maxlen=100),
            'grad_norm': deque(maxlen=100)
        }
    
    def reset_buffer(self):
        """Reset the experience buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Get action from the agent.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action, log probability, and value estimate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor, deterministic)
        
        return action, log_prob.item(), value.item()
    
    def store_experience(self, 
                        state: np.ndarray,
                        action: int,
                        log_prob: float,
                        reward: float,
                        value: float,
                        done: bool):
        """Store experience in the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            next_value: Value of the next state (0 if terminal)
            
        Returns:
            Returns and advantages
        """
        advantages = []
        gae = 0
        
        # Add next value for GAE calculation
        values = self.values + [next_value]
        
        # Calculate advantages in reverse order
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * values[i + 1] * (1 - self.dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
        
        # Calculate returns
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return returns, advantages
    
    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """
        Update the agent using PPO.
        
        Args:
            next_value: Value of the next state
            
        Returns:
            Training statistics
        """
        if len(self.states) == 0:
            return {}
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        dataset_size = len(states)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_loss = 0
        update_count = 0
        
        for _ in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                action_probs, values = self.network(batch_states)
                dist = Categorical(action_probs)
                
                # Calculate losses
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Policy loss (PPO clipping)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
                update_count += 1
        
        # Store statistics
        if update_count > 0:
            self.training_stats['policy_loss'].append(total_policy_loss / update_count)
            self.training_stats['value_loss'].append(total_value_loss / update_count)
            self.training_stats['entropy'].append(total_entropy / update_count)
            self.training_stats['total_loss'].append(total_loss / update_count)
            self.training_stats['grad_norm'].append(grad_norm.item())
        
        # Reset buffer
        self.reset_buffer()
        
        return {
            'policy_loss': total_policy_loss / update_count if update_count > 0 else 0,
            'value_loss': total_value_loss / update_count if update_count > 0 else 0,
            'entropy': total_entropy / update_count if update_count > 0 else 0,
            'total_loss': total_loss / update_count if update_count > 0 else 0,
            'grad_norm': grad_norm.item() if update_count > 0 else 0
        }
    
    def save(self, filepath: str):
        """Save the agent."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': dict(self.training_stats)
        }, filepath)
    
    def load(self, filepath: str):
        """Load the agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {}
        for key, values in self.training_stats.items():
            if len(values) > 0:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
        return stats


class PPOTrainer:
    """Trainer for PPO agent."""
    
    def __init__(self, agent: PPOAgent, env: gym.Env, logger_name: str = __name__):
        self.agent = agent
        self.env = env
        self.logger = logging.getLogger(logger_name)
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def train_episode(self, max_steps: int = 1000) -> Dict[str, float]:
        """Train for one episode."""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Get action
            action, log_prob, value = self.agent.get_action(state)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.store_experience(state, action, log_prob, reward, value, done)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update agent
        next_value = 0.0 if done else self.agent.get_action(state)[2]  # Get value of next state
        train_stats = self.agent.update(next_value)
        
        # Store episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        
        # Combine statistics
        episode_stats = {
            'episode_reward': total_reward,
            'episode_length': steps,
            'avg_reward': np.mean(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths)
        }
        
        return {**episode_stats, **train_stats}
    
    def train(self, 
              num_episodes: int,
              save_freq: int = 100,
              eval_freq: int = 50,
              save_path: str = None) -> List[Dict[str, float]]:
        """
        Train the agent for multiple episodes.
        
        Args:
            num_episodes: Number of episodes to train
            save_freq: Frequency to save the model
            eval_freq: Frequency to evaluate the model
            save_path: Path to save the model
            
        Returns:
            List of training statistics
        """
        training_history = []
        
        for episode in range(num_episodes):
            # Train episode
            stats = self.train_episode()
            training_history.append(stats)
            
            # Logging
            if episode % 10 == 0:
                self.logger.info(
                    f"Episode {episode}: "
                    f"Reward={stats['episode_reward']:.2f}, "
                    f"Length={stats['episode_length']}, "
                    f"AvgReward={stats['avg_reward']:.2f}"
                )
            
            # Save model
            if save_path and episode % save_freq == 0:
                self.agent.save(f"{save_path}_episode_{episode}.pt")
            
            # Evaluation
            if episode % eval_freq == 0:
                eval_stats = self.evaluate(num_episodes=5)
                self.logger.info(f"Evaluation at episode {episode}: {eval_stats}")
        
        return training_history
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate the agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            
        Returns:
            Evaluation statistics
        """
        rewards = []
        lengths = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            done = False
            while not done:
                action, _, _ = self.agent.get_action(state, deterministic=deterministic)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1
            
            rewards.append(total_reward)
            lengths.append(steps)
        
        return {
            'eval_reward_mean': np.mean(rewards),
            'eval_reward_std': np.std(rewards),
            'eval_length_mean': np.mean(lengths),
            'eval_length_std': np.std(lengths)
        }
